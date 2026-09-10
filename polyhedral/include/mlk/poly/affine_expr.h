// MLK+ polyhedral engine — integer affine expressions.
//
// An AffineExpr is sum_i coeff[i] * x[i] + constant over the engine's
// flat variable space: [0, nDims) are iteration dimensions, [nDims,
// nDims + nSyms) are symbolic parameters (workload sizes). Coefficients
// are int64 and bounded by kRationalMagnitudeLimit (Rule 27: named
// limits; Rule 10: bounded passes).
//
// All arithmetic is overflow-checked and returns Result (Rule 6). The
// engine never wraps silently (Rule 73).
#pragma once

#include <cstdint>
#include <string>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/rational.h"

namespace mlk::poly {

/// Variable space descriptor shared by expressions, systems, and maps.
struct VarSpace {
    uint32_t nDims{0};  // iteration dimensions
    uint32_t nSyms{0};  // symbolic parameters (workload sizes)

    [[nodiscard]] uint32_t totalVars() const noexcept { return nDims + nSyms; }
    [[nodiscard]] bool sameAs(const VarSpace& o) const noexcept {
        return nDims == o.nDims && nSyms == o.nSyms;
    }
};

/// Integer affine expression over a VarSpace.
struct AffineExpr {
    VarSpace space{};
    SmallVector<int64_t, 8> coeffs{};  // one per totalVars()
    int64_t constant{0};

    [[nodiscard]] static AffineExpr fromConstant(const VarSpace& space,
                                             int64_t v) noexcept;
    [[nodiscard]] static AffineExpr variable(const VarSpace& space,
                                             uint32_t varIndex) noexcept;

    [[nodiscard]] bool isConstant(int64_t* value = nullptr) const noexcept;
    /// Coefficient of var; 0 when out of range (expressions are dense).
    [[nodiscard]] int64_t coeffOf(uint32_t var) const noexcept;

    /// Evaluates at an integer point (vars.size() == totalVars()).
    /// Fails on overflow (Rule 6).
    [[nodiscard]] Result<int64_t> evalAt(
        const SmallVector<int64_t, 8>& vars) const noexcept;

    /// gcd-normalizes the coefficient vector (all coeffs and constant
    /// divided by their gcd). Used to canonicalize equality rows.
    void gcdNormalize() noexcept;

    /// True when every coefficient is within the engine magnitude limit.
    [[nodiscard]] bool withinLimits() const noexcept;
};

[[nodiscard]] Result<AffineExpr> exprAdd(const AffineExpr& a,
                                         const AffineExpr& b) noexcept;
[[nodiscard]] Result<AffineExpr> exprSub(const AffineExpr& a,
                                         const AffineExpr& b) noexcept;
[[nodiscard]] Result<AffineExpr> exprScale(const AffineExpr& a,
                                           int64_t k) noexcept;
/// Same space required; fails with InvalidArgument otherwise (Rule 67).
[[nodiscard]] bool exprSpacesMatch(const AffineExpr& a,
                                   const AffineExpr& b) noexcept;

/// Stable structural equality (Rule 24).
[[nodiscard]] bool exprEqual(const AffineExpr& a,
                             const AffineExpr& b) noexcept;

/// Pretty form for tools/diagnostics only (cold path, never stored in IR).
[[nodiscard]] std::string exprToString(const AffineExpr& e,
                                       const SmallVector<std::string, 8>& varNames);

}  // namespace mlk::poly
