// MLK+ polyhedral engine — exact rational LP solver.
//
// Dense tableau simplex over Rationals with Bland's pivot rule (Rule 53:
// deterministic — the same input always yields the same schedule; Bland's
// rule also guarantees termination, no cycling). Free variables are split
// into non-negative differences; Phase 1 finds a feasible basis with
// artificials, Phase 2 minimizes the objective.
//
// Cold-path component (compile time only). Sizes are budgeted (Rule 10).
// Infeasibility is REPORTED, never approximated (Rule 33: legality
// decisions never rest on rounded values).
#pragma once

#include <cstdint>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/rational.h"

namespace mlk::poly {

/// LP size budgets (Rule 10).
inline constexpr std::size_t kLpMaxVars = 1024;      // after splitting
inline constexpr std::size_t kLpMaxRows = 512;

/// A linear program:
///   minimize    c . x
///   subject to  A_eq . x == b_eq
///               A_ge . x >= b_ge
///               x free
/// Rows are dense Rational vectors over nVars.
struct LinearProgram {
    uint32_t nVars{0};
    SmallVector<SmallVector<Rational, 8>, 8> eqRows{};
    SmallVector<Rational, 8> eqRhs{};
    SmallVector<SmallVector<Rational, 8>, 8> geRows{};
    SmallVector<Rational, 8> geRhs{};
    SmallVector<Rational, 8> objective{};
};

enum class LpStatus : uint8_t { Optimal = 0, Infeasible, Unbounded };

struct LpSolution {
    LpStatus status{LpStatus::Infeasible};
    Rational objective{};
    SmallVector<Rational, 8> x{};
};

/// Solves the LP exactly. Deterministic (Rule 53).
[[nodiscard]] Result<LpSolution> solveLp(const LinearProgram& lp);

}  // namespace mlk::poly
