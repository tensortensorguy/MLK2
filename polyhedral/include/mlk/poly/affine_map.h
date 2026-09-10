// MLK+ polyhedral engine — affine maps (multi-output, integer).
//
// An AffineMap is a tuple of AffineExprs over a shared input space:
//   map: [dims_in] + [syms] -> [nOut affine outputs]
// Maps model memory access functions (iteration -> address), schedules
// (iteration -> time), and domain embeddings. Symbols are parameters
// shared across the pipeline and are never eliminated by image/preimage
// (they ride through; see int_set.h for the symbol specialization rule).
#pragma once

#include <cstdint>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/affine_expr.h"
#include "mlk/poly/int_set.h"

namespace mlk::poly {

struct AffineMap {
    VarSpace inSpace{};  // nDims inputs + nSyms parameters
    uint32_t nOut{0};
    SmallVector<AffineExpr, 4> outputs{};

    [[nodiscard]] static AffineMap identity(const VarSpace& space) noexcept;
    /// Constant embedding: all outputs constant.
    [[nodiscard]] static AffineMap constantMap(const VarSpace& space,
                                               int64_t v) noexcept;

    /// Evaluates the map at an input point.
    [[nodiscard]] Result<SmallVector<int64_t, 8>> evalAt(
        const SmallVector<int64_t, 8>& vars) const noexcept;

    /// Structural equality (Rule 24).
    [[nodiscard]] bool sameAs(const AffineMap& o) const noexcept;
};

/// Composes outer ∘ inner: applies `outer` to each output of `inner`.
/// inner.inSpace defines the domain; outer must be defined over a space
/// with nDims == inner.nOut (symbols must match: inner.nSyms).
[[nodiscard]] Result<AffineMap> mapCompose(const AffineMap& outer,
                                           const AffineMap& inner) noexcept;

/// Image of a set under a map: { f(d) | d in set }.
/// Implementation: product space (d, x) with x_j - f_j(d) == 0, then
/// existentially eliminate d (FM; symbols kept). Result space: nOut dims
/// + same symbols. Rational-projection caveat applies (int_set.h doc):
/// callers needing exact integer images (dependence analysis) pass sets
/// whose maps are unimodular or where the caveat is conservative-safe.
[[nodiscard]] Result<PresburgerSet> mapImage(const PresburgerSet& set,
                                             const AffineMap& map) noexcept;

/// Preimage of a set under a map: { d | f(d) in set }.
/// Product space with set constraints on x, x_j - f_j(d) == 0, eliminate x.
[[nodiscard]] Result<PresburgerSet> mapPreimage(const PresburgerSet& set,
                                                const AffineMap& map) noexcept;

}  // namespace mlk::poly
