// MLK+ polyhedral engine — Presburger integer sets over affine constraints.
//
// Representation: a set is a DISJUNCTION of conjunctive systems (polyhedra).
// Each row is coeffs*x + constant >= 0 (inequality) or == 0 (equality) over
// the flat variable space [dims, symbols] (affine_expr.h VarSpace).
//
// Soundness contract (documented for reviewers; see docs/polyhedral_spec.md):
//  - isEmpty() applies Fourier-Motzkin elimination. Rational emptiness
//    implies integer emptiness (sound in the safe direction).
//  - Feasibility is TRI-STATE (Rule 22 discipline): Empty / NonEmpty
//    (witness-verified or rationally non-empty) / Unknown (elimination
//    budget exceeded). Legality callers must treat Unknown as NonEmpty
//    (conservative: keeps dependences, loses optimization, never breaks
//    correctness — Rules 62/102 fallback discipline).
//  - lexMin()/lexMax() enumerate candidates in ascending/descending
//    lexicographic order with rational-feasibility pruning (sound) and
//    verify the final integer point against the ORIGINAL constraints
//    (witness check). The first verified point is the exact optimum.
//    Budget exhaustion yields an error, never a wrong optimum (Rule 53:
//    determinism; Rule 10: bounded passes).
//
// Symbols (parameters) must be specialized to constants before lexMin /
// lexMax (the polyhedral pipeline specializes workload sizes first; see
// poly.scop_detect). Parametric emptiness (dependence legality) is
// supported directly by FM over dims+symbols.
#pragma once

#include <cstdint>
#include <optional>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/affine_expr.h"

namespace mlk::poly {

/// Row budget per polyhedron (Rule 10: explicit budgets).
inline constexpr std::size_t kPolyMaxRowsPerSystem = 256;
/// Disjunct budget per set (complement of a k-row system has k disjuncts).
inline constexpr std::size_t kPolyMaxDisjuncts = 64;
/// FM intermediate row cap before declaring Unknown (blow-up guard).
inline constexpr std::size_t kPolyFmMaxIntermediateRows = 512;
/// Branch-and-bound node budget for lexmin/lexmax (Rule 131).
inline constexpr uint32_t kPolyLexBbNodeBudget = 4096;

/// Tri-state feasibility (Rule 22: Unknown never silently becomes a
/// definite answer; Unknown is treated as NonEmpty by legality callers).
enum class Feasibility : uint8_t { Empty = 0, NonEmpty = 1, Unknown = 2 };

/// One conjunctive constraint system: coeffs*x + c >= 0 / == 0.
struct ConstraintRow {
    SmallVector<int64_t, 8> coeffs{};
    int64_t constant{0};
    bool isEquality{false};

    [[nodiscard]] bool isTriviallyTrue() const noexcept {
        if (isEquality) return constant == 0 && allCoeffsZero();
        return constant >= 0 && allCoeffsZero();
    }
    [[nodiscard]] bool isTriviallyFalse() const noexcept {
        if (isEquality) return constant != 0 && allCoeffsZero();
        return constant < 0 && allCoeffsZero();
    }
    [[nodiscard]] int64_t coeffOf(uint32_t var) const noexcept {
        return var < coeffs.size() ? coeffs[var] : 0;
    }

private:
    [[nodiscard]] bool allCoeffsZero() const noexcept {
        for (const int64_t c : coeffs) {
            if (c != 0) return false;
        }
        return true;
    }
};

/// A conjunctive polyhedron (conjunction of rows over one VarSpace).
struct Polyhedron {
    VarSpace space{};
    SmallVector<ConstraintRow, 8> rows{};
    bool isEmptyFlag{false};    // set when a contradiction row was added
    bool rowsOverflow{false};   // row budget exceeded => feasibility Unknown

    void addRow(ConstraintRow row);
    void addInequality(SmallVector<int64_t, 8> coeffs, int64_t constant);
    void addEquality(SmallVector<int64_t, 8> coeffs, int64_t constant);

    /// Exact integer point check (all rows evaluated; no approximation).
    [[nodiscard]] bool containsPoint(
        const SmallVector<int64_t, 8>& vars) const noexcept;
};

/// Disjunctive Presburger set.
struct PresburgerSet {
    VarSpace space{};
    SmallVector<Polyhedron, 4> disjuncts{};

    [[nodiscard]] static PresburgerSet empty(const VarSpace& space) noexcept;
    /// Full-dimensional box: dim d in [lb[d], ub[d]] (inclusive, constants).
    [[nodiscard]] static Result<PresburgerSet> box(
        const VarSpace& space, const SmallVector<int64_t, 8>& lower,
        const SmallVector<int64_t, 8>& upper) noexcept;
    /// Box with affine bounds: lb_d(x) <= x_d <= ub_d(x).
    [[nodiscard]] static Result<PresburgerSet> affineBox(
        const VarSpace& space, const SmallVector<AffineExpr, 8>& lower,
        const SmallVector<AffineExpr, 8>& upper) noexcept;

    [[nodiscard]] static Result<PresburgerSet> intersect(
        const PresburgerSet& a, const PresburgerSet& b) noexcept;
    [[nodiscard]] static Result<PresburgerSet> unite(
        const PresburgerSet& a, const PresburgerSet& b) noexcept;
    [[nodiscard]] static Result<PresburgerSet> complement(
        const PresburgerSet& a) noexcept;
    [[nodiscard]] static Result<PresburgerSet> subtract(
        const PresburgerSet& a, const PresburgerSet& b) noexcept;

    /// Tri-state emptiness via Fourier-Motzkin elimination.
    [[nodiscard]] Result<Feasibility> feasibility() const noexcept;
    /// True only when PROVABLY empty (safe direction for legality).
    [[nodiscard]] Result<bool> provablyEmpty() const noexcept;

    /// Project one variable out of every disjunct (FM; may report Unknown
    /// through the row budget as over-approximation — documented).
    [[nodiscard]] Result<PresburgerSet> projectOut(uint32_t var) const noexcept;

    /// Existential image projection: keep only vars in keepVars (dims with
    /// a true flag); all other dims are eliminated (symbols are KEPT —
    /// they are parameters, not quantified variables).
    [[nodiscard]] Result<PresburgerSet> existsProject(
        const SmallVector<bool, 8>& keepDims) const noexcept;

    /// Exact integer lexmin over dims (order 0,1,2,...; symbols must be
    /// specialized: nSyms > 0 is an InvalidArgument error). Returns the
    /// point (dims only) or nullopt when the set is integer-empty.
    [[nodiscard]] Result<std::optional<SmallVector<int64_t, 8>>> lexMin()
        const noexcept;
    /// Exact integer lexmax (descending lex order; same contract).
    [[nodiscard]] Result<std::optional<SmallVector<int64_t, 8>>> lexMax()
        const noexcept;

    /// Exact integer point containment across all disjuncts.
    [[nodiscard]] bool containsPoint(
        const SmallVector<int64_t, 8>& vars) const noexcept;

    /// Number of disjuncts that are not flagged empty.
    [[nodiscard]] std::size_t liveDisjuncts() const noexcept;
};

namespace detail {

/// Eliminates one variable from a system via FM. Returns nullopt when the
/// system is (rationally) empty, a row budget overflow reports
/// ResourceExhausted, otherwise the projected system.
[[nodiscard]] Result<std::optional<ConstraintRow>> fmEliminateVar(
    Polyhedron sys, uint32_t var, SmallVector<ConstraintRow, 8>& outRows);

/// Rational-only feasibility of one system (used internally by lexmin).
[[nodiscard]] Result<Feasibility> systemFeasibility(const Polyhedron& sys);

}  // namespace detail

}  // namespace mlk::poly
