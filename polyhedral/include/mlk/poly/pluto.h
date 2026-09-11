// MLK+ polyhedral engine — Pluto-style affine scheduling.
//
// Synthesizes a legal (dependence-preserving) affine schedule in two
// layers (Bondhugula et al., PLDI 2008, sequential formulation adapted
// to the engine's exact Presburger relations):
//
// ORDER SEARCH (outer layer). The scheduler does not assume the original
// loop order: every pivot ORDER (a permutation of the varying dims) is a
// candidate transformation. Depth <= 4 enumerates all orders; deeper
// SCoPs extend a prefix greedily one pivot at a time. Each candidate
// order is synthesized row by row (inner layer) and scored EXACTLY:
//   1. parallel rows   (more is better — ML kernels want concurrency),
//   2. innermost unit-stride fit (the last row parallel AND every active
//      statement's accesses stride 0/1 along its pivot — SIMD-able),
//   3. total carried distance (exact rational sum of the per-dependence
//      minimum distances at their resolving rows — smaller skews/fuses),
//   4. lexicographic pivot order (determinism; keeps the identity order
//      on ties).
// The first order with the best score wins; the identity order is always
// synthesized (it is valid by original-order purification), so the
// search is total: it degrades to the round-11 scheduler on ties.
//
// INNER LAYER (per candidate order). Every row is SELECTED by an LP over
// the schedule coefficients — nothing is copied from the original loop:
//
//   For schedule row r, every not-yet-resolved dependence d contributes
//   its SLICE polyhedron: relation_d intersected with the equality rows
//   "earlier-row distance == 0" (earlier rows are already fixed numbers,
//   so the slice stays a Presburger polyhedron). Farkas' lemma turns
//     "distance_r(v) >= 0 for all v in slice"
//   into linear constraints over the schedule coefficients plus
//   non-negative multipliers. Three deterministic stages, first feasible
//   wins:
//     1. PARALLEL search — a row under which every live dependence
//        distance is identically zero (feasibility LP with both
//        dist >= 0 and -dist >= 0 Farkas blocks). Such rows FUSE
//        statement nests and carry nothing.
//     2. SEQUENTIAL search — an epigraph variable M_d per live dep (the
//        maximum distance of d on the slice) carries the objective
//          minimize  sum_d M_d   subject to   sum_d M_d >= 1 (progress)
//        which keeps dependence distances small — the fusion/skewing
//        driver. Spent-dim (outer-loop) coefficients stay free, so skew
//        rows (theta = v_inner + k*v_outer) are synthesized when they
//        legalize better orders.
//     3. IDENTITY fallback — theta = e_pivot. Original-order
//        purification orients every dependence forward, so this row is
//        always valid; it guarantees termination (one pivot dim spent
//        per row).
//   Rows are emitted until every varying dimension is spent (totality:
//   every statement instance maps to a distinct schedule vector, so
//   codegen can replay each payload exactly once). Dependences still
//   live then are schedule-tied (all row distances zero) and resolve by
//   statement order (origOrder) — see verifyScheduleLegality.
//
// Shape contract (codegen compatibility, checked per realized row): each
// statement is either a foldable constant (nonzero coefficients only on
// pinned dims) or varies on the row's pivot dim with a positive
// coefficient. The loop variable emitted by codegen IS the pivot dim
// (box bounds); skew coefficients never reach loop bounds. With any
// pivot SEQUENCE the nest enumerates instances in exactly the schedule's
// lexicographic order (equal prefixes force equal spent coords by
// induction), which is what makes arbitrary permutations codegen-safe.
//
// Budgets (Rule 10): the order search and every candidate LP run under
// kPolyMaxSchedulerLps / kPolyMaxOrderLps; a tripped budget rejects the
// candidate order, never the pass. A candidate whose synthesis errors
// (overflow, validity trip) is skipped identically.
//
// Determinism (Rule 53): exact rational simplex with Bland's rule, fixed
// enumeration order, integer scaling by LCM + gcd per row, exact scores.
// Failure mode (Rule 62): any error yields "no schedule found"; the
// baseline kernel stays untouched.
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
    /// The pivot (loop-carried) dimension of each row: the unspent dim
    /// the row varies on. Codegen emits the pivot dim as the loop
    /// variable; tiling legality is checked on pivot-coordinate
    /// distances (skew coefficients cancel in loop bounds).
    SmallVector<uint32_t, 12> pivotDim{};
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
