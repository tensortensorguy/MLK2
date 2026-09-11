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

    for (uint32_t r = 0; r < constants::kPolyMaxScheduleRows; ++r) {
        // Live (unresolved) dependence list; compact M-column positions.
        SmallVector<uint32_t, 16> live;
        for (uint32_t di = 0; di < states.size(); ++di) {
            if (!states[di].resolved) live.push_back(di);
        }
        if (live.empty()) break;
        const uint32_t baseCols = nStmts * coeffsPerStmt;

        // Deterministic "resolve one dependence per row": the primary dep
        // is forced to strictly positive distance (dist >= 1 everywhere on
        // its slice); all other live deps keep dist >= 0 with an epigraph
        // objective that keeps their distances small (the fusion driver).
        // The hard primary constraint makes progress mandatory — no
        // degenerate rows. Intra-statement deps are tried first: their
        // slices pin instance coordinates, so satisfying them requires a
        // real dimension row (never a bare statement-order constant),
        // which keeps dimension rows ahead of separator rows.
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
            bool lpOk = true;
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
                    Result<void> fr =
                        appendFarkas(dom, parts, 0, lp);
                    if (!fr.has_value()) {
                        return err(fr.error().code, fr.error().message);
                    }
                }
            }
            // Per-dependence blocks.
            for (uint32_t li = 0; li < live.size(); ++li) {
                const uint32_t di = live[li];
                DepState& st = states[di];
                const Dependence& d = deps[di];
                const uint32_t mCol = baseCols + li;
                const int64_t shift = (di == primary) ? 1 : 0;
                Result<FormParts> pr = distanceParts(
                    depth, coeffsPerStmt, lp.nVars, d.srcStmt, d.dstStmt);
                if (!pr.has_value()) {
                    return err(pr.error().code, pr.error().message);
                }
                const FormParts& parts = *pr;
                for (const Polyhedron& disjunct : st.slice.disjuncts) {
                    if (!detail::polyFeasible(disjunct)) continue;
                    // Block 1: dist - shift >= 0.
                    FormParts plain;
                    for (uint32_t j = 0; j <= depth * 2; ++j) {
                        SmallVector<Rational, 8> row(lp.nVars, Rational{});
                        row = parts[j];
                        plain.push_back(std::move(row));
                    }
                    Result<void> f1 =
                        appendFarkas(disjunct, plain, shift, lp);
                    if (!f1.has_value()) {
                        return err(f1.error().code, f1.error().message);
                    }
                    if (shift != 0) continue;  // primary: no epigraph
                    // Block 2 (non-primary): M_d - dist >= 0.
                    FormParts epi;
                    for (uint32_t j = 0; j <= depth * 2; ++j) {
                        SmallVector<Rational, 8> row(lp.nVars, Rational{});
                        for (uint32_t col = 0; col < parts[j].size();
                             ++col) {
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
                    Result<void> f2 =
                        appendFarkas(disjunct, epi, 0, lp);
                    if (!f2.has_value()) {
                        return err(f2.error().code, f2.error().message);
                    }
                }
                (void)lpOk;
            }
            // Objective: minimize sum of NON-primary M_d.
            lp.objective = SmallVector<Rational, 8>(lp.nVars, Rational{});
            for (uint32_t li = 0; li < live.size(); ++li) {
                if (live[li] != primary) {
                    lp.objective[baseCols + li] = Rational{1, 1};
                }
            }
            MLK_TRY_VAR(sol, solveLp(lp));
            if (sol.status != LpStatus::Optimal) continue;  // try next dep

            // Realize the row as canonical integers.
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
                       "pluto: no dependence can make progress (row " +
                           std::to_string(r) + ")");
        }
        const ScheduleRow& row = *realized;

        // Post-row: distance forms with the fixed integer row.
        SmallVector<bool, 16> resolvesNow = primaryResolved;
        bool rowParallel = true;
        for (uint32_t li = 0; li < live.size(); ++li) {
            const uint32_t di = live[li];
            DepState& st = states[di];
            const Dependence& d = deps[di];
            SmallVector<int64_t, 8> distCoeffs;
            int64_t distConst = 0;
            MLK_TRYV(detail::distanceForm(row, d, depth, distCoeffs,
                                          distConst));
            // Resolution: min distance over every live disjunct >= 1.
            bool anyLive = false;
            Rational minDist{};
            for (const Polyhedron& disjunct : st.slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                MLK_TRY_VAR(mn, detail::minOverPoly(disjunct, distCoeffs,
                                                distConst));
                if (!anyLive || ratLess(mn, minDist)) minDist = mn;
                anyLive = true;
            }
            const bool resolved =
                anyLive && minDist.isInt() && minDist.num >= 1;
            resolvesNow[di] = resolved;
            // Parallel test: the dim is parallel only when this dep's
            // distance is IDENTICALLY zero on its slice (it carries
            // nothing). A dep resolved by this row (dist >= 1) is exactly
            // the one being CARRIED here — it forces sequential order.
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

        // Slice updates: resolved deps leave; others gain dist == 0.
        for (uint32_t li = 0; li < live.size(); ++li) {
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
                st.resolved = true;  // slice vanished: fully zero-dist
            } else {
                st.slice = std::move(next);
            }
        }

        out.rows.push_back(std::move(row));
    }

    // Every dependence must be resolved by the final row (the last row
    // enforces dist >= 1 on all remaining slices).
    for (const DepState& st : states) {
        if (!st.resolved) {
            return err(ErrorCode::UnsupportedCapability,
                       "pluto: row budget exhausted with live dependences");
        }
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
            // Slice emptiness (resolved by earlier rows).
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
            // Min distance over the live slice.
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
