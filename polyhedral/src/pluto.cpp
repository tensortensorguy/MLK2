// Pluto-style scheduler implementation (see pluto.h).
//
// Per schedule row r the LP column layout is
//   [ S0: c0..cd | S1: ... | M over live deps | multipliers... ]
// where the c columns are free schedule coefficients and M_d are distance
// epigraphs. Every "for all v in polyhedron: LHS(v) >= 0" is encoded via
// Farkas' lemma (see pluto.h). Blocks appended per row, per remaining
// dependence d, per live slice disjunct:
//   (1) dist_d(v) >= 0               (last row: dist_d(v) - 1 >= 0)
//   (2) M_d - dist_d(v) >= 0
// per statement: theta_s(v) >= 0 over the statement domain; progress:
// sum_d M_d >= 1 (non-final rows); objective: minimize sum_d M_d.
// After each row: integer scaling via LCM of denominators + gcd
// normalization; resolution test (min distance over the slice >= 1 via a
// small LP); surviving slices gain the equality "distance_r == 0". A dep
// whose slice vanishes is resolved (fully zero distance — its ordering is
// fixed by statement order in codegen).
#include "mlk/poly/pluto.h"

#include <optional>
#include <utility>

#include "schedule_common.h"
#include "mlk/poly/checked.h"
#include "mlk/poly/lp.h"

namespace mlk::poly {

namespace {

/// LHS form of a Farkas block: for each polyhedron variable j (plus the
/// constant slot nVars), a Rational row over the LP columns AT THE TIME
/// THE BLOCK IS BUILT (before this block's multipliers are appended).
using FormParts = SmallVector<SmallVector<Rational, 8>, 8>;

/// Distance form of dependence d under schedule coefficients: fills
/// parts[j] (j <= 2*depth) where parts[j][col] is the LP-column expression
/// of the v_j coefficient of theta_dst(v') - theta_src(v).
[[nodiscard]] Result<FormParts> distanceParts(uint32_t depth,
                                              uint32_t coeffsPerStmt,
                                              uint32_t nLpCols,
                                              uint32_t srcStmt,
                                              uint32_t dstStmt) {
    const uint32_t nPolyVars = depth * 2;
    const uint32_t srcBase = srcStmt * coeffsPerStmt;
    const uint32_t dstBase = dstStmt * coeffsPerStmt;
    FormParts parts;
    for (uint32_t j = 0; j <= nPolyVars; ++j) {
        SmallVector<Rational, 8> row(nLpCols, Rational{});
        if (j < depth) {
            row[srcBase + 1 + j] = Rational{-1, 1};
        } else if (j < nPolyVars) {
            row[dstBase + 1 + (j - depth)] = Rational{1, 1};
        } else {
            row[dstBase] = Rational{1, 1};
            row[srcBase] = Rational{-1, 1};
        }
        parts.push_back(std::move(row));
    }
    return parts;
}

/// Appends a Farkas block for "LHS(v) + constantShift >= 0 for all v in
/// poly". One free multiplier column per poly row; inequality-row
/// multipliers get a >= 0 row; coefficient matching yields one equality
/// per poly variable (rhs 0) plus the constant slot (rhs constantShift).
[[nodiscard]] Result<void> appendFarkas(const Polyhedron& poly,
                                        const FormParts& parts,
                                        int64_t constantShift,
                                        LinearProgram& lp) {
    const uint32_t nPolyVars = poly.space.totalVars();
    const uint32_t nRows = static_cast<uint32_t>(poly.rows.size());
    const uint32_t base = lp.nVars;
    lp.nVars = base + nRows;
    if (lp.nVars > kLpMaxVars) {
        return err(ErrorCode::ResourceExhausted,
                   "pluto LP exceeds column budget");
    }
    for (uint32_t k = 0; k < nRows; ++k) {
        if (poly.rows[k].isEquality) continue;
        SmallVector<Rational, 8> ge(lp.nVars, Rational{});
        ge[base + k] = Rational{1, 1};
        lp.geRows.push_back(std::move(ge));
        lp.geRhs.push_back(Rational{});
    }
    for (uint32_t j = 0; j <= nPolyVars; ++j) {
        SmallVector<Rational, 8> eq(lp.nVars, Rational{});
        for (uint32_t col = 0; col < parts[j].size() && col < lp.nVars;
             ++col) {
            eq[col] = parts[j][col];
        }
        for (uint32_t k = 0; k < nRows; ++k) {
            const ConstraintRow& row = poly.rows[k];
            const int64_t rk = j < nPolyVars ? row.coeffOf(j)
                                             : row.constant;
            if (rk == 0) continue;
            eq[base + k] = Rational{-rk, 1};
        }
        lp.eqRows.push_back(std::move(eq));
        lp.eqRhs.push_back(j == nPolyVars ? Rational{constantShift, 1}
                                          : Rational{});
    }
    return {};
}

/// Scales one realized row: LCM of all coefficient denominators, then a
/// whole-row gcd normalization; verifies the coefficient bound. The
/// result is the canonical, deterministic integer form (Rule 24).
[[nodiscard]] Result<void> normalizeRow(ScheduleRow& row, int64_t lcm) {
    for (auto& sc : row.stmtCoeffs) {
        for (int64_t& c : sc) {
            MLK_TRY_VAR(scaled, checked::mulLimited(c, lcm));
            c = scaled;
        }
    }
    int64_t g = 0;
    for (const auto& sc : row.stmtCoeffs) {
        for (const int64_t c : sc) g = ratGcd(g, c);
    }
    if (g > 1) {
        for (auto& sc : row.stmtCoeffs) {
            for (int64_t& c : sc) c /= g;
        }
    }
    for (const auto& sc : row.stmtCoeffs) {
        for (const int64_t c : sc) {
            if (c > constants::kPolyMaxScheduleCoeff ||
                c < -constants::kPolyMaxScheduleCoeff) {
                return err(ErrorCode::ResourceExhausted,
                           "schedule coefficients exceed bound");
            }
        }
    }
    return {};
}

}  // namespace

Result<PolySchedule> computePlutoSchedule(
    const Scop& scop, const SmallVector<Dependence, 16>& deps) {
    const uint32_t depth = scop.depth;
    const uint32_t nStmts = static_cast<uint32_t>(scop.statements.size());
    if (depth == 0 || nStmts == 0) {
        return err(ErrorCode::InvalidArgument, "empty SCoP");
    }
    const uint32_t coeffsPerStmt = depth + 1;

    PolySchedule out;
    out.depth = depth;
    out.nStmts = nStmts;

    struct DepState {
        PresburgerSet slice{};
        bool resolved{false};
    };
    SmallVector<DepState, 16> states;
    for (const Dependence& d : deps) {
        DepState st;
        st.slice = d.relation;
        st.resolved = false;
        states.push_back(std::move(st));
    }

    // Post-row bookkeeping shared by both phases: resolution (min distance
    // >= 1 on the slice), parallel marking (every live dependence distance
    // identically zero), and slice refinement (dist == 0 equalities).
    auto processRow = [&](const ScheduleRow& row,
                          const SmallVector<uint32_t, 16>& live,
                          SmallVector<bool, 16>& resolvesNow) -> Result<void> {
        bool rowParallel = true;
        for (std::size_t li = 0; li < live.size(); ++li) {
            const uint32_t di = live[li];
            DepState& st = states[di];
            const Dependence& d = deps[di];
            SmallVector<int64_t, 8> distCoeffs;
            int64_t distConst = 0;
            MLK_TRYV(detail::distanceForm(row, d, depth, distCoeffs,
                                          distConst));
            bool anyLive = false;
            Rational minDist{};
            for (const Polyhedron& disjunct : st.slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                MLK_TRY_VAR(mn, detail::minOverPoly(disjunct, distCoeffs,
                                                distConst));
                if (!anyLive || ratLess(mn, minDist)) minDist = mn;
                anyLive = true;
            }
            resolvesNow[di] =
                anyLive && minDist.isInt() && minDist.num >= 1;
            // Parallel only when this dep's distance is identically zero
            // on its slice (the dim carries nothing). A dep resolved by
            // this row is exactly the one being CARRIED here.
            if (anyLive) {
                SmallVector<int64_t, 8> negCoeffs;
                for (const int64_t c : distCoeffs) negCoeffs.push_back(-c);
                Rational maxDist{};
                bool anyMax = false;
                for (const Polyhedron& disjunct : st.slice.disjuncts) {
                    if (!detail::polyFeasible(disjunct)) continue;
                    MLK_TRY_VAR(mx, detail::minOverPoly(
                                        disjunct, negCoeffs, -distConst));
                    if (!anyMax || ratLess(maxDist, mx)) maxDist = mx;
                    anyMax = true;
                }
                if (anyMax && !(maxDist.isInt() && maxDist.num == 0)) {
                    rowParallel = false;
                }
            }
        }
        out.parallel.push_back(rowParallel);
        for (std::size_t li = 0; li < live.size(); ++li) {
            const uint32_t di = live[li];
            DepState& st = states[di];
            if (resolvesNow[di]) {
                st.resolved = true;
                continue;
            }
            const Dependence& d = deps[di];
            SmallVector<int64_t, 8> distCoeffs;
            int64_t distConst = 0;
            MLK_TRYV(detail::distanceForm(row, d, depth, distCoeffs,
                                          distConst));
            PresburgerSet next;
            next.space = st.slice.space;
            bool anyLiveDisjunct = false;
            for (const Polyhedron& disjunct : st.slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                Polyhedron updated = disjunct;
                ConstraintRow eq;
                eq.isEquality = true;
                for (const int64_t c : distCoeffs) eq.coeffs.push_back(c);
                eq.constant = -distConst;
                updated.addRow(std::move(eq));
                if (detail::polyFeasible(updated)) {
                    next.disjuncts.push_back(std::move(updated));
                    anyLiveDisjunct = true;
                }
            }
            if (!anyLiveDisjunct) {
                st.resolved = true;
            } else {
                st.slice = std::move(next);
            }
        }
        return {};
    };

    // Phase 1 — identity prefix: theta_r = v_r reproduces the ORIGINAL
    // program order for extraction-produced SCoPs, which is legal by
    // construction (the original execution is the legality reference).
    // Rows parallel-marked when no dependence is carried at them.
    for (uint32_t r = 0; r < depth; ++r) {
        SmallVector<uint32_t, 16> live;
        for (uint32_t di = 0; di < states.size(); ++di) {
            if (!states[di].resolved) live.push_back(di);
        }
        if (live.empty()) break;
        ScheduleRow row;
        for (uint32_t st = 0; st < nStmts; ++st) {
            SmallVector<int64_t, 8> cs(coeffsPerStmt, 0);
            cs[r + 1] = 1;
            row.stmtCoeffs.push_back(std::move(cs));
        }
        SmallVector<bool, 16> resolvesNow(states.size(), false);
        MLK_TRYV(processRow(row, live, resolvesNow));
        out.rows.push_back(std::move(row));
    }

    // Phase 2 — separator rows for dependences the identity never
    // strictly separates (cross-statement same-position pairs, e.g. an
    // init statement vs its accumulator): Feautrier-style hard primary
    // resolution, one dependence per row, intra-statement primaries
    // first. These rows are statement-order separators (constants) for
    // the synth kernel class.
    while (true) {
        SmallVector<uint32_t, 16> live;
        for (uint32_t di = 0; di < states.size(); ++di) {
            if (!states[di].resolved) live.push_back(di);
        }
        if (live.empty() ||
            out.rows.size() >= constants::kPolyMaxScheduleRows) {
            break;
        }
        SmallVector<uint32_t, 16> candidates;
        for (const uint32_t di : live) {
            if (deps[di].srcStmt == deps[di].dstStmt) {
                candidates.push_back(di);
            }
        }
        for (const uint32_t di : live) {
            if (deps[di].srcStmt != deps[di].dstStmt) {
                candidates.push_back(di);
            }
        }
        const uint32_t baseCols = nStmts * coeffsPerStmt;
        std::optional<ScheduleRow> realized;
        SmallVector<bool, 16> primaryResolved(states.size(), false);
        for (const uint32_t primary : candidates) {
            LinearProgram lp;
            lp.nVars = baseCols + static_cast<uint32_t>(live.size());
            if (lp.nVars > kLpMaxVars) {
                return err(ErrorCode::ResourceExhausted,
                           "pluto LP exceeds column budget");
            }
            // Statement non-negativity: theta_s(v) >= 0 over the domain.
            for (const Statement& st : scop.statements) {
                for (const Polyhedron& dom : st.domain.disjuncts) {
                    if (!detail::polyFeasible(dom)) continue;
                    FormParts parts;
                    for (uint32_t j = 0; j <= st.space.nDims; ++j) {
                        SmallVector<Rational, 8> row(lp.nVars, Rational{});
                        if (j < st.space.nDims) {
                            row[st.id * coeffsPerStmt + 1 + j] =
                                Rational{1, 1};
                        } else {
                            row[st.id * coeffsPerStmt] = Rational{1, 1};
                        }
                        parts.push_back(std::move(row));
                    }
                    MLK_TRYV(appendFarkas(dom, parts, 0, lp));
                }
            }
            for (uint32_t li = 0; li < live.size(); ++li) {
                const uint32_t di = live[li];
                DepState& st = states[di];
                const Dependence& d = deps[di];
                const uint32_t mCol = baseCols + li;
                const int64_t shift = (di == primary) ? 1 : 0;
                MLK_TRY_VAR(parts,
                            distanceParts(depth, coeffsPerStmt, lp.nVars,
                                          d.srcStmt, d.dstStmt));
                for (const Polyhedron& disjunct : st.slice.disjuncts) {
                    if (!detail::polyFeasible(disjunct)) continue;
                    FormParts plain;
                    for (uint32_t j = 0; j <= depth * 2; ++j) {
                        SmallVector<Rational, 8> row(lp.nVars, Rational{});
                        row = parts[j];
                        plain.push_back(std::move(row));
                    }
                    MLK_TRYV(appendFarkas(disjunct, plain, shift, lp));
                    if (shift != 0) continue;
                    FormParts epi;
                    for (uint32_t j = 0; j <= depth * 2; ++j) {
                        SmallVector<Rational, 8> row(lp.nVars, Rational{});
                        for (uint32_t col = 0; col < parts[j].size(); ++col) {
                            Result<Rational> n = ratNeg(parts[j][col]);
                            if (!n.has_value()) {
                                return err(ErrorCode::ResourceExhausted,
                                           "pluto farkas overflow");
                            }
                            row[col] = *n;
                        }
                        if (j == depth * 2) row[mCol] = Rational{1, 1};
                        epi.push_back(std::move(row));
                    }
                    MLK_TRYV(appendFarkas(disjunct, epi, 0, lp));
                }
            }
            lp.objective = SmallVector<Rational, 8>(lp.nVars, Rational{});
            for (uint32_t li = 0; li < live.size(); ++li) {
                if (live[li] != primary) {
                    lp.objective[baseCols + li] = Rational{1, 1};
                }
            }
            MLK_TRY_VAR(sol, solveLp(lp));
            if (sol.status != LpStatus::Optimal) continue;
            ScheduleRow row;
            int64_t lcm = 1;
            for (uint32_t st2 = 0; st2 < nStmts; ++st2) {
                SmallVector<int64_t, 8> cs;
                for (uint32_t j = 0; j < coeffsPerStmt; ++j) {
                    const Rational& v = sol.x[st2 * coeffsPerStmt + j];
                    if (!v.isInt() && v.den != 0) {
                        const int64_t g = ratGcd(lcm, v.den);
                        MLK_TRY_VAR(q, checked::divLimited(v.den, g));
                        MLK_TRY_VAR(nl, checked::mulLimited(lcm, q));
                        lcm = nl;
                    }
                    cs.push_back(v.num);
                }
                row.stmtCoeffs.push_back(std::move(cs));
            }
            for (uint32_t st2 = 0; st2 < nStmts; ++st2) {
                for (uint32_t j = 0; j < coeffsPerStmt; ++j) {
                    const Rational& v = sol.x[st2 * coeffsPerStmt + j];
                    MLK_TRY_VAR(factor, checked::divLimited(lcm, v.den));
                    MLK_TRY_VAR(scaled, checked::mulLimited(v.num, factor));
                    row.stmtCoeffs[st2][j] = scaled;
                }
            }
            MLK_TRYV(normalizeRow(row, 1));
            primaryResolved[primary] = true;
            realized = std::move(row);
            break;
        }
        if (!realized.has_value()) {
            return err(ErrorCode::UnsupportedCapability,
                       "pluto: unresolved dependences cannot be separated");
        }
        SmallVector<bool, 16> resolvesNow = primaryResolved;
        MLK_TRYV(processRow(*realized, live, resolvesNow));
        out.rows.push_back(std::move(*realized));
    }

    // Vectorizable: the innermost (last) parallel row, if any.
    out.vectorizable = SmallVector<bool, 12>(out.rows.size(), false);
    for (int32_t rr = static_cast<int32_t>(out.rows.size()) - 1; rr >= 0;
         --rr) {
        if (out.parallel[rr]) {
            out.vectorizable[rr] = true;
            break;
        }
    }
    return out;
}

Result<bool> verifyScheduleLegality(
    const Scop& scop, const SmallVector<Dependence, 16>& deps,
    const PolySchedule& sched) {
    if (sched.nStmts != scop.statements.size() ||
        sched.depth != scop.depth) {
        return err(ErrorCode::InvalidArgument,
                   "schedule/SCoP shape mismatch");
    }
    for (const Dependence& d : deps) {
        PresburgerSet slice = d.relation;
        bool resolved = false;
        for (uint32_t r = 0; r < sched.rows.size() && !resolved; ++r) {
            MLK_TRY_VAR(f, slice.feasibility());
            if (f == Feasibility::Empty) {
                resolved = true;
                break;
            }
            const ScheduleRow& row = sched.rows[r];
            SmallVector<int64_t, 8> distCoeffs;
            int64_t distConst = 0;
            MLK_TRYV(detail::distanceForm(row, d, scop.depth, distCoeffs,
                                          distConst));
            Rational minDist{};
            bool anyLive = false;
            for (const Polyhedron& disjunct : slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                MLK_TRY_VAR(mn, detail::minOverPoly(disjunct, distCoeffs,
                                                distConst));
                if (!anyLive || ratLess(mn, minDist)) minDist = mn;
                anyLive = true;
            }
            if (!anyLive) {
                resolved = true;
                break;
            }
            if (minDist.isInt()) {
                if (minDist.num >= 1) {
                    resolved = true;  // strictly ordered at this row
                    break;
                }
                if (minDist.num < 0) {
                    return false;  // order violated
                }
            }
            // min == 0 (or fractional): refine the slice and continue.
            PresburgerSet next;
            next.space = slice.space;
            for (const Polyhedron& disjunct : slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                Polyhedron updated = disjunct;
                ConstraintRow eq;
                eq.isEquality = true;
                for (const int64_t c : distCoeffs) eq.coeffs.push_back(c);
                eq.constant = -distConst;
                updated.addRow(std::move(eq));
                if (!updated.isEmptyFlag) {
                    next.disjuncts.push_back(std::move(updated));
                }
            }
            slice = std::move(next);
        }
        if (!resolved) {
            MLK_TRY_VAR(f, slice.feasibility());
            if (f != Feasibility::Empty) return false;
        }
    }
    return true;
}

}  // namespace mlk::poly
