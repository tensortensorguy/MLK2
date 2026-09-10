// MLK+ polyhedral engine — Pluto-style affine scheduling.
//
// Synthesizes a legal (dependence-preserving) affine schedule row by row
// (Bondhugula et al., PLDI 2008, sequential formulation adapted to the
// engine's exact Presburger relations):
//
//   For schedule row r, every not-yet-resolved dependence d contributes
//   its SLICE polyhedron: relation_d intersected with the equality rows
//   "earlier-row distance == 0" (earlier rows are already fixed numbers,
//   so the slice stays a Presburger polyhedron). Farkas' lemma turns
//     "distance_r(v) >= 0 for all v in slice"
//   into linear constraints over the schedule coefficients plus
//   non-negative multipliers. An epigraph variable M_d (the maximum
//   distance of d on the slice) carries the objective:
//     minimize  sum_d M_d   subject to   sum_d M_d >= 1 (progress)
//   which keeps dependence distances small — the fusion/skewing driver.
//   The final row forces strictness (every remaining dependence carried
//   with distance >= 1) so the lexicographic order is total.
//
// Determinism (Rule 53): exact rational simplex with Bland's rule, fixed
// variable ordering, integer scaling by LCM + gcd normalization per row.
// Failure mode (Rule 62): any infeasibility/budget trip yields an error;
// the baseline kernel stays untouched.
#pragma once

#include <cstdint>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/dependence.h"
#include "mlk/poly/scop.h"

namespace mlk::poly {

/// One schedule row: per statement [constant, coeff(d0), ..., coeff(d_{n-1})].
struct ScheduleRow {
    SmallVector<SmallVector<int64_t, 8>, 8> stmtCoeffs{};
};

/// The synthesized affine schedule.
struct PolySchedule {
    uint32_t depth{0};
    uint32_t nStmts{0};
    SmallVector<ScheduleRow, 12> rows{};
    SmallVector<bool, 12> parallel{};      // no dep carried at/after r
    SmallVector<bool, 12> vectorizable{};  // innermost parallel row
};

/// Computes a Pluto-style schedule. Errors mean "no schedule found" —
/// callers keep the baseline kernel (Rules 62/102).
[[nodiscard]] Result<PolySchedule> computePlutoSchedule(
    const Scop& scop, const SmallVector<Dependence, 16>& deps);

/// Verifies schedule legality: for every dependence, the lexicographic
/// distance theta(v') - theta(v) over its relation must be positive at
/// some row prefix (with zero distances before). Returns true only for a
/// proven-legal schedule; errors mean "cannot verify" (conservative).
[[nodiscard]] Result<bool> verifyScheduleLegality(
    const Scop& scop, const SmallVector<Dependence, 16>& deps,
    const PolySchedule& sched);

}  // namespace mlk::poly
