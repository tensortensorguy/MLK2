// Pluto-style scheduler implementation (see pluto.h).
//
// Per schedule row r the LP column layout is
//   [ S0: c0..cd | S1: ... | M over live deps | multipliers... ]
// where the c columns are free schedule coefficients and M_d are distance
// epigraphs. Every "for all v in polyhedron: LHS(v) >= 0" is encoded via
// Farkas' lemma (see pluto.h). Blocks appended per row, per remaining
// dependence d, per live slice disjunct:
//   (1) dist_d(v) >= 0               (validity)
//   (2) M_d - dist_d(v) >= 0         (epigraph; sequential search only)
// and for the PARALLEL search both (1) and its negation (dist_d == 0).
// per statement: theta_s(v) >= 0 over the statement domain; shape:
// coefficient of every unspent dim other than the pivot == 0, pivot
// coefficient >= 0, anchor statement's pivot coefficient == 1; bounds:
// |c| <= kRowCoeffBound; progress (sequential): sum_d M_d >= 1;
// objective: minimize sum_d M_d.
//
// Row selection runs in three deterministic stages, first feasible wins:
//   1. PARALLEL search — a row under which every live dependence has
//      distance identically zero (feasibility LP). Such a row fuses
//      statement nests and carries nothing; parallel marks follow.
//   2. SEQUENTIAL search — minimize the epigraph sum subject to validity
//      and progress; exact post-classification (min distance per dep on
//      its slice) resolves deps with min >= 1 and refines the rest with
//      "distance == 0" equalities.
//   3. IDENTITY fallback — theta = e_pivot for every statement.
//      Original-order purification orients every dependence forward in
//      the original dim order, so this row is always valid and always
//      codegen-shape compatible; it guarantees progress (one pivot dim
//      spent per row) even when both LP searches fail or reject.
// Rows are emitted until every varying dimension is spent (totality: each
// statement instance gets a distinct schedule vector — required by
// codegen, which replays each payload exactly once) or no pivot candidate
// remains. Dependences still live at that point have all-zero row
// distances: they are schedule-tied and resolve by statement order
// (origOrder) in codegen.
//
// Skewing: spent-dim (outer-loop) coefficients are FREE in the LP — a row
// like theta = v_1 + 3*v_0 legalizes loop permutations the naive swap
// breaks (classic skew-permute). Skew coefficients never reach loop
// bounds (the emitted loop variable stays the pivot dim with its box
// bounds); they only steer legality and statement classification.
//
// Determinism (Rule 53): exact rational simplex with Bland's rule, fixed
// pivot/anchor enumeration order, integer scaling by LCM + gcd per row.
// Failure mode (Rule 62): budget trips yield an error; the baseline kernel
// stays untouched.
#include "mlk/poly/pluto.h"

#include <optional>
#include <utility>

#include "schedule_common.h"
#include "mlk/poly/checked.h"
#include "mlk/poly/lp.h"

namespace mlk::poly {

namespace {

/// LP-side coefficient bound (keeps Farkas blocks small; the post-scale
/// bound is constants::kPolyMaxScheduleCoeff).
inline constexpr int64_t kRowCoeffBound = 32;

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

/// Negates every entry of a Farkas form (encodes dist == 0 as the pair
/// dist >= 0 and -dist >= 0 in the parallel search).
[[nodiscard]] Result<FormParts> negateParts(const FormParts& parts) {
    FormParts out;
    for (const auto& row : parts) {
        SmallVector<Rational, 8> neg;
        for (const Rational& v : row) {
            Result<Rational> n = ratNeg(v);
            if (!n.has_value()) {
                return err(ErrorCode::ResourceExhausted,
                           "pluto farkas negation overflow");
            }
            neg.push_back(*n);
        }
        out.push_back(std::move(neg));
    }
    return out;
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

/// Extracts an integer schedule row from an LP solution: LCM scaling of
/// the rational coefficients, then gcd normalization.
[[nodiscard]] Result<ScheduleRow> realizeRow(const LpSolution& sol,
                                             uint32_t nStmts,
                                             uint32_t coeffsPerStmt) {
    ScheduleRow row;
    int64_t lcm = 1;
    for (uint32_t s = 0; s < nStmts; ++s) {
        SmallVector<int64_t, 8> cs;
        for (uint32_t j = 0; j < coeffsPerStmt; ++j) {
            const Rational& v = sol.x[s * coeffsPerStmt + j];
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
    for (uint32_t s = 0; s < nStmts; ++s) {
        for (uint32_t j = 0; j < coeffsPerStmt; ++j) {
            const Rational& v = sol.x[s * coeffsPerStmt + j];
            MLK_TRY_VAR(factor, checked::divLimited(lcm, v.den));
            MLK_TRY_VAR(scaled, checked::mulLimited(v.num, factor));
            row.stmtCoeffs[s][j] = scaled;
        }
    }
    MLK_TRYV(normalizeRow(row, 1));
    return row;
}

/// Codegen-shape compatibility of a realized row (see pluto.h): every
/// statement is either a foldable constant (all nonzero coefficients sit
/// on pinned dims) or varies on the pivot with a positive coefficient.
/// Rows violating this would make codegen replay instances ambiguously or
/// strand finished statements inside deeper loops — rejected here so the
/// search falls through to the next candidate (identity always passes).
[[nodiscard]] bool rowCodegenCompatible(const Scop& scop,
                                        const ScheduleRow& row,
                                        uint32_t pivot) {
    for (const Statement& s : scop.statements) {
        int64_t folded = 0;
        if (detail::rowEffectiveConst(s, row, &folded)) continue;
        const auto& cs = row.stmtCoeffs[s.id];
        if (pivot + 1 < cs.size() && cs[pivot + 1] >= 1 &&
            detail::dimVaries(s, pivot)) {
            continue;
        }
        return false;
    }
    return true;
}

/// Per-dependence scheduling state: the slice is the dependence relation
/// refined by "distance == 0" equalities of all previous rows.
struct DepState {
    PresburgerSet slice{};
    bool resolved{false};
};

/// Builds the LP for one (pivot, anchor) candidate. Column layout: stmt
/// coefficient blocks, then (sequential search only) one epigraph M
/// column per live dependence; Farkas multiplier columns are appended per
/// block. Returns nullopt when the LP is infeasible (candidate rejected).
[[nodiscard]] Result<std::optional<ScheduleRow>> tryRowLp(
    const Scop& scop,
    const SmallVector<Dependence, 16>& deps,
    const SmallVector<DepState, 16>& states,
    const SmallVector<uint32_t, 16>& live,
    const SmallVector<bool, 8>& spent, uint32_t pivot, uint32_t anchor,
    bool parallelSearch) {
    const uint32_t nStmts = static_cast<uint32_t>(scop.statements.size());
    const uint32_t depth = scop.depth;
    const uint32_t coeffsPerStmt = depth + 1;
    const uint32_t baseCols = nStmts * coeffsPerStmt;

    LinearProgram lp;
    lp.nVars = baseCols;
    if (!parallelSearch) lp.nVars += static_cast<uint32_t>(live.size());
    if (lp.nVars > kLpMaxVars) {
        return err(ErrorCode::ResourceExhausted,
                   "pluto LP exceeds column budget");
    }

    // Shape: coefficient of every unspent dim other than the pivot == 0.
    for (uint32_t s = 0; s < nStmts; ++s) {
        for (uint32_t d = 0; d < depth; ++d) {
            if (spent[d] || d == pivot) continue;
            SmallVector<Rational, 8> eq(lp.nVars, Rational{});
            eq[s * coeffsPerStmt + 1 + d] = Rational{1, 1};
            lp.eqRows.push_back(std::move(eq));
            lp.eqRhs.push_back(Rational{});
        }
    }
    // Pivot non-negativity: order-preserving pivots only (original-order
    // purification makes negative pivots useless — see pluto.h).
    for (uint32_t s = 0; s < nStmts; ++s) {
        SmallVector<Rational, 8> ge(lp.nVars, Rational{});
        ge[s * coeffsPerStmt + 1 + pivot] = Rational{1, 1};
        lp.geRows.push_back(std::move(ge));
        lp.geRhs.push_back(Rational{});
    }
    // Anchor: the pivot direction is nonzero (unit coefficient).
    {
        SmallVector<Rational, 8> eq(lp.nVars, Rational{});
        eq[anchor * coeffsPerStmt + 1 + pivot] = Rational{1, 1};
        lp.eqRows.push_back(std::move(eq));
        lp.eqRhs.push_back(Rational{1, 1});
    }
    // Coefficient bounds |c| <= kRowCoeffBound (constant slot included).
    for (uint32_t s = 0; s < nStmts; ++s) {
        for (uint32_t j = 0; j < coeffsPerStmt; ++j) {
            const uint32_t col = s * coeffsPerStmt + j;
            // c >= -B  (the system is >= -form)
            SmallVector<Rational, 8> loRow(lp.nVars, Rational{});
            loRow[col] = Rational{1, 1};
            lp.geRows.push_back(std::move(loRow));
            lp.geRhs.push_back(Rational{-kRowCoeffBound, 1});
            // c <= B  ->  -c >= -B
            SmallVector<Rational, 8> hiRow(lp.nVars, Rational{});
            hiRow[col] = Rational{-1, 1};
            lp.geRows.push_back(std::move(hiRow));
            lp.geRhs.push_back(Rational{-kRowCoeffBound, 1});
        }
    }
    // Statement non-negativity: theta_s(v) >= 0 over each domain disjunct
    // (canonical form: schedule values start at 0).
    for (const Statement& st : scop.statements) {
        for (const Polyhedron& dom : st.domain.disjuncts) {
            if (!detail::polyFeasible(dom)) continue;
            FormParts parts;
            for (uint32_t j = 0; j <= st.space.nDims; ++j) {
                SmallVector<Rational, 8> row(lp.nVars, Rational{});
                if (j < st.space.nDims) {
                    row[st.id * coeffsPerStmt + 1 + j] = Rational{1, 1};
                } else {
                    row[st.id * coeffsPerStmt] = Rational{1, 1};
                }
                parts.push_back(std::move(row));
            }
            MLK_TRYV(appendFarkas(dom, parts, 0, lp));
        }
    }
    // Dependence blocks over the live slices.
    for (uint32_t li = 0; li < live.size(); ++li) {
        const uint32_t di = live[li];
        const DepState& st = states[di];
        const Dependence& dep = deps[di];
        const uint32_t mCol = parallelSearch
                                  ? 0
                                  : baseCols + li;
        for (const Polyhedron& disjunct : st.slice.disjuncts) {
            if (!detail::polyFeasible(disjunct)) continue;
            MLK_TRY_VAR(parts,
                        distanceParts(depth, coeffsPerStmt, lp.nVars,
                                      dep.srcStmt, dep.dstStmt));
            // Validity: dist >= 0 on the slice.
            MLK_TRYV(appendFarkas(disjunct, parts, 0, lp));
            if (parallelSearch) {
                // Parallel row: dist == 0 on the slice.
                MLK_TRY_VAR(neg, negateParts(parts));
                MLK_TRYV(appendFarkas(disjunct, neg, 0, lp));
                continue;
            }
            // Epigraph: M_d >= dist on the slice.
            FormParts epi;
            for (uint32_t j = 0; j <= depth * 2; ++j) {
                SmallVector<Rational, 8> row(lp.nVars, Rational{});
                for (uint32_t col = 0; col < parts[j].size() &&
                                         col < lp.nVars; ++col) {
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
    if (!parallelSearch) {
        if (live.empty()) {
            // Progress with no live deps: keep the row varying on the
            // pivot only (anchor already ensures a nonzero direction).
        } else {
            // Progress: at least one epigraph reaches 1.
            SmallVector<Rational, 8> ge(lp.nVars, Rational{});
            for (uint32_t li = 0; li < live.size(); ++li) {
                ge[baseCols + li] = Rational{1, 1};
            }
            lp.geRows.push_back(std::move(ge));
            lp.geRhs.push_back(Rational{1, 1});
            // Objective: minimize the epigraph sum (distance-minimal
            // rows — the fusion/skewing driver).
            lp.objective = SmallVector<Rational, 8>(lp.nVars, Rational{});
            for (uint32_t li = 0; li < live.size(); ++li) {
                lp.objective[baseCols + li] = Rational{1, 1};
            }
        }
    }
    MLK_TRY_VAR(sol, solveLp(lp));
    if (sol.status != LpStatus::Optimal) return std::nullopt;
    ScheduleRow row;
    MLK_TRY_VAR(r, realizeRow(sol, nStmts, coeffsPerStmt));
    row = std::move(r);
    return row;
}

/// Post-row bookkeeping shared by all stages: exact per-dependence
/// classification on the realized integer row (resolution when the min
/// distance reaches 1), parallel marking (every live dependence distance
/// identically zero on its slice), and slice refinement with
/// "distance == 0" equalities. Returns the row's parallel mark.
[[nodiscard]] Result<bool> processRow(
    const SmallVector<Dependence, 16>& deps,
    SmallVector<DepState, 16>& states, const ScheduleRow& row,
    uint32_t depth) {
    bool rowParallel = true;
    for (uint32_t di = 0; di < states.size(); ++di) {
        DepState& st = states[di];
        if (st.resolved) continue;
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
        if (anyLive && minDist.isInt() && minDist.num < 0) {
            return err(ErrorCode::Internal,
                       "pluto: realized row violates dependence validity");
        }
        if (anyLive && minDist.isInt() && minDist.num >= 1) {
            st.resolved = true;  // strictly ordered at this row
            rowParallel = false;  // the row carries this dependence
            continue;
        }
        if (anyLive) {
            // Parallel only when the distance is identically zero on the
            // slice (the dim carries nothing of this dependence).
            SmallVector<int64_t, 8> negCoeffs;
            for (const int64_t c : distCoeffs) negCoeffs.push_back(-c);
            Rational maxDist{};
            bool anyMax = false;
            for (const Polyhedron& disjunct : st.slice.disjuncts) {
                if (!detail::polyFeasible(disjunct)) continue;
                MLK_TRY_VAR(mx, detail::minOverPoly(disjunct, negCoeffs,
                                                    -distConst));
                if (!anyMax || ratLess(maxDist, mx)) maxDist = mx;
                anyMax = true;
            }
            if (anyMax && !(maxDist.isInt() && maxDist.num == 0)) {
                rowParallel = false;
            }
        }
        // Refine the slice with "distance == 0" for the next row.
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
            st.resolved = true;  // slice vanished: nothing left to order
        } else {
            st.slice = std::move(next);
        }
    }
    return rowParallel;
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

    SmallVector<DepState, 16> states;
    for (const Dependence& d : deps) {
        DepState st;
        st.slice = d.relation;
        st.resolved = false;
        states.push_back(std::move(st));
    }
    SmallVector<bool, 8> spent(depth, false);

    while (out.rows.size() < constants::kPolyMaxScheduleRows) {
        // Pivot candidates: unspent dims that vary for at least one
        // statement (a pivot nobody varies on would emit an empty loop).
        SmallVector<uint32_t, 8> candidates;
        for (uint32_t d = 0; d < depth; ++d) {
            if (spent[d]) continue;
            for (const Statement& s : scop.statements) {
                if (detail::dimVaries(s, d)) {
                    candidates.push_back(d);
                    break;
                }
            }
        }
        if (candidates.empty()) break;  // totality reached

        SmallVector<uint32_t, 16> live;
        for (uint32_t di = 0; di < states.size(); ++di) {
            if (!states[di].resolved) live.push_back(di);
        }

        // Totality rows (no dependences left to order): the plain
        // identity row on the first candidate pivot. Deterministic and
        // minimal — with nothing to carry there is nothing to optimize.
        if (live.empty()) {
            const uint32_t p = candidates[0];
            ScheduleRow row;
            for (uint32_t s = 0; s < nStmts; ++s) {
                SmallVector<int64_t, 8> cs(coeffsPerStmt, 0);
                cs[p + 1] = 1;
                row.stmtCoeffs.push_back(std::move(cs));
            }
            MLK_TRY_VAR(par, processRow(deps, states, row, depth));
            out.rows.push_back(std::move(row));
            out.pivotDim.push_back(p);
            out.parallel.push_back(par);
            spent[p] = true;
            continue;
        }

        // Stage 1 — parallel rows: every live dependence distance
        // identically zero (feasibility). Tried for every (pivot,
        // anchor) pair before any sequential row: Pluto's parallelism
        // preference.
        std::optional<ScheduleRow> realized;
        uint32_t pivotUsed = candidates[0];
        for (const uint32_t p : candidates) {
            for (uint32_t a = 0; a < nStmts && !realized.has_value(); ++a) {
                if (!detail::dimVaries(scop.statements[a], p)) continue;
                auto par = tryRowLp(scop, deps, states, live, spent,
                                    p, a, true);
                // Budget/overflow trips on a candidate LP reject the
                // candidate (Rule 10); the identity fallback below keeps
                // the scheduler total, and processRow validates whatever
                // row is realized.
                if (par.has_value() && par->has_value() &&
                    rowCodegenCompatible(scop, **par, p)) {
                    realized = std::move(*par);
                    pivotUsed = p;
                }
            }
            if (realized.has_value()) break;
        }
        // Stage 2 — sequential rows: validity + progress, minimizing the
        // epigraph sum (distance-minimal carrying).
        if (!realized.has_value()) {
            for (const uint32_t p : candidates) {
                for (uint32_t a = 0; a < nStmts && !realized.has_value();
                     ++a) {
                    if (!detail::dimVaries(scop.statements[a], p)) continue;
                    auto seq = tryRowLp(scop, deps, states, live,
                                        spent, p, a, false);
                    if (seq.has_value() && seq->has_value() &&
                        rowCodegenCompatible(scop, **seq, p)) {
                        realized = std::move(*seq);
                        pivotUsed = p;
                    }
                }
                if (realized.has_value()) break;
            }
        }
        // Stage 3 — identity fallback (always valid, always compatible).
        if (!realized.has_value()) {
            const uint32_t p = candidates[0];
            ScheduleRow row;
            for (uint32_t s = 0; s < nStmts; ++s) {
                SmallVector<int64_t, 8> cs(coeffsPerStmt, 0);
                cs[p + 1] = 1;
                row.stmtCoeffs.push_back(std::move(cs));
            }
            realized = std::move(row);
            pivotUsed = p;
        }

        MLK_TRY_VAR(par, processRow(deps, states, *realized, depth));
        out.rows.push_back(std::move(*realized));
        out.pivotDim.push_back(pivotUsed);
        out.parallel.push_back(par);
        spent[pivotUsed] = true;
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
            if (f == Feasibility::Empty) continue;
            // Schedule-tied: every row distance is zero on the surviving
            // slice, so codegen orders the pair by statement order —
            // legal exactly when the source precedes the sink (an
            // intra-statement tie would be the same instance, which a
            // live dependence never is).
            if (d.srcStmt == d.dstStmt) return false;
            if (scop.statements[d.srcStmt].origOrder >=
                scop.statements[d.dstStmt].origOrder) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace mlk::poly
