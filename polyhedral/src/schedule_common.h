// MLK+ polyhedral engine — shared scheduler internals (src-private).
//
// Helpers shared by pluto.cpp (row synthesis) and tile.cpp (band
// detection): integer distance forms under a fixed schedule row, exact
// linear-form minimization over a polyhedron, and rational-feasibility
// filtering of disjuncts. Not part of the public engine API.
#pragma once

#include "mlk/poly/checked.h"
#include "mlk/poly/lp.h"
#include "mlk/poly/pluto.h"

namespace mlk::poly::detail {

using CoeffRow = SmallVector<int64_t, 8>;  // [c0, c1..cd] for one statement

/// Integer distance form of dependence d under a FIXED integer schedule
/// row: coefficients over the 2*depth product vars (src i | dst j) +
/// constant. dist = theta_dst(v') - theta_src(v).
[[nodiscard]] inline Result<void> distanceForm(const ScheduleRow& row,
                                               const Dependence& dep,
                                               uint32_t depth,
                                               CoeffRow& outCoeffs,
                                               int64_t& outConst) {
    outCoeffs.clear();
    for (uint32_t i = 0; i < depth * 2; ++i) outCoeffs.push_back(0);
    const auto& src = row.stmtCoeffs[dep.srcStmt];
    const auto& dst = row.stmtCoeffs[dep.dstStmt];
    for (uint32_t d = 0; d < depth; ++d) {
        MLK_TRY_VAR(s, checked::subLimited(0, src[d + 1]));
        outCoeffs[d] = s;
        MLK_TRY_VAR(t, checked::addLimited(0, dst[d + 1]));
        outCoeffs[depth + d] = t;
    }
    MLK_TRY_VAR(k, checked::subLimited(dst[0], src[0]));
    outConst = k;
    return {};
}

/// Minimizes a linear form over a polyhedron (bounded by construction;
/// instance slices come from bounded statement domains).
[[nodiscard]] inline Result<Rational> minOverPoly(
    const Polyhedron& poly, const SmallVector<int64_t, 8>& formCoeffs,
    int64_t formConst) {
    LinearProgram lp;
    lp.nVars = poly.space.totalVars();
    if (lp.nVars == 0) {
        return err(ErrorCode::InvalidArgument, "min over empty space");
    }
    for (const ConstraintRow& r : poly.rows) {
        SmallVector<Rational, 8> coeffs(lp.nVars, Rational{});
        for (uint32_t j = 0; j < lp.nVars; ++j) {
            coeffs[j] = Rational{r.coeffOf(j), 1};
        }
        const Rational rhs{-r.constant, 1};
        if (r.isEquality) {
            lp.eqRows.push_back(std::move(coeffs));
            lp.eqRhs.push_back(rhs);
        } else {
            lp.geRows.push_back(std::move(coeffs));
            lp.geRhs.push_back(rhs);
        }
    }
    for (uint32_t j = 0; j < lp.nVars; ++j) {
        lp.objective.push_back(j < formCoeffs.size()
                                   ? Rational{formCoeffs[j], 1}
                                   : Rational{});
    }
    MLK_TRY_VAR(sol, solveLp(lp));
    if (sol.status == LpStatus::Unbounded) {
        return err(ErrorCode::InvalidArgument,
                   "min over an unbounded polyhedron");
    }
    if (sol.status != LpStatus::Optimal) {
        return err(ErrorCode::Internal, "min over an infeasible polyhedron");
    }
    MLK_TRY_VAR(total, ratAdd(sol.objective, Rational{formConst, 1}));
    return total;
}

/// True when the disjunct can hold a (rational) point. Disjuncts that are
/// FM-emptied without a direct contradiction flag are filtered here so
/// Farkas blocks (vacuous on empty sets) and minimizations never see them.
[[nodiscard]] inline bool polyFeasible(const Polyhedron& p) {
    if (p.isEmptyFlag) return false;
    if (p.rowsOverflow) return true;  // Unknown → treat as feasible
    auto f = detail::systemFeasibility(p);
    return f.has_value() && *f != Feasibility::Empty;
}

/// True when the statement's dim d is pinned to a compile-time constant:
/// singleton own-bounds, or a deep dim beyond the statement's own depth
/// (pinned to 0 by extraction — see scop.h).
[[nodiscard]] inline bool dimPinnedAt(const Statement& s, uint32_t d,
                                      int64_t* value) {
    if (d < s.ownLower.size() && d < s.ownUpper.size()) {
        if (s.ownLower[d] == s.ownUpper[d]) {
            *value = s.ownLower[d];
            return true;
        }
        return false;
    }
    if (d >= s.depth) {
        *value = 0;
        return true;
    }
    return false;
}

/// True when the statement's dim d actually varies over its domain.
[[nodiscard]] inline bool dimVaries(const Statement& s, uint32_t d) {
    int64_t pin = 0;
    return !dimPinnedAt(s, d, &pin);
}

/// Row-constant fold for one statement: true with *value = constant +
/// sum(coeff_d * pin_d) when every nonzero row coefficient sits on a dim
/// pinned for this statement; false when any nonzero coefficient
/// references a varying (loop-carried) dim. Magnitudes are bounded by
/// construction (kPolyMaxScheduleCoeff coefficients, workload-specialized
/// singleton bounds), so plain arithmetic is safe here.
[[nodiscard]] inline bool rowEffectiveConst(const Statement& s,
                                            const ScheduleRow& row,
                                            int64_t* value) {
    const auto& cs = row.stmtCoeffs[s.id];
    int64_t folded = cs[0];
    for (uint32_t d = 0; d + 1 < cs.size(); ++d) {
        const int64_t c = cs[d + 1];
        if (c == 0) continue;
        int64_t pin = 0;
        if (!dimPinnedAt(s, d, &pin)) return false;
        folded += c * pin;
    }
    *value = folded;
    return true;
}

}  // namespace mlk::poly::detail
