// Affine map implementation (see affine_map.h).
//
// Product-space convention (documented; see docs/polyhedral_spec.md §maps):
//   product dims = [ map inputs d (nIn) | map outputs x (nOut) | symbols ]
//   symbol slots always sit AFTER both dim blocks, matching the engine's
//   VarSpace layout. embedSet() shifts a source system's dim columns by
//   dimOffset and its symbol columns to symBase; remapToDimFirst() compacts
//   a projected product back into a [dims | symbols] result space.
#include "mlk/poly/affine_map.h"

#include <utility>

namespace mlk::poly {

namespace {

/// Copies a single-disjunct system into `target`, shifting dims by
/// dimOffset and placing symbols at symBase + s.
[[nodiscard]] Result<void> embedSystem(const Polyhedron& src,
                                       uint32_t dimOffset, uint32_t symBase,
                                       Polyhedron& target) {
    if (src.isEmptyFlag) {
        target.isEmptyFlag = true;
        return {};
    }
    if (src.rowsOverflow) {
        return err(ErrorCode::ResourceExhausted,
                   "embedSystem on over-budget system");
    }
    for (const ConstraintRow& r : src.rows) {
        ConstraintRow shifted;
        shifted.isEquality = r.isEquality;
        shifted.constant = r.constant;
        shifted.coeffs = SmallVector<int64_t, 8>(target.space.totalVars(), 0);
        for (uint32_t d = 0; d < src.space.nDims && d < r.coeffs.size(); ++d) {
            const uint32_t t = dimOffset + d;
            if (t >= shifted.coeffs.size()) {
                return err(ErrorCode::InvalidArgument,
                           "embedSystem: dim offset out of product space");
            }
            shifted.coeffs[t] = r.coeffs[d];
        }
        for (uint32_t s = 0; s < src.space.nSyms; ++s) {
            const uint32_t srcIdx = src.space.nDims + s;
            const uint32_t t = symBase + s;
            if (srcIdx < r.coeffs.size() && t < shifted.coeffs.size()) {
                shifted.coeffs[t] = r.coeffs[srcIdx];
            }
        }
        target.addRow(std::move(shifted));
        if (target.isEmptyFlag) return {};
    }
    return {};
}

/// Compacts a projected product system back to [dims | symbols]: kept dim
/// columns are listed in `keptDimSlots` (product slot per result dim);
/// symbols move from symBase to the trailing nSyms slots.
[[nodiscard]] Polyhedron remapToDimFirst(const Polyhedron& sys,
                                         const SmallVector<uint32_t, 8>& keptDimSlots,
                                         uint32_t symBase, uint32_t nSyms,
                                         uint32_t nOutDims) {
    Polyhedron out;
    out.space = VarSpace{nOutDims, nSyms};
    out.isEmptyFlag = sys.isEmptyFlag;
    for (const ConstraintRow& r : sys.rows) {
        ConstraintRow nr;
        nr.isEquality = r.isEquality;
        nr.constant = r.constant;
        nr.coeffs = SmallVector<int64_t, 8>(out.space.totalVars(), 0);
        for (uint32_t d = 0; d < nOutDims && d < keptDimSlots.size(); ++d) {
            if (keptDimSlots[d] < r.coeffs.size()) {
                nr.coeffs[d] = r.coeffs[keptDimSlots[d]];
            }
        }
        for (uint32_t s = 0; s < nSyms; ++s) {
            const uint32_t srcSlot = symBase + s;
            if (srcSlot < r.coeffs.size()) {
                nr.coeffs[nOutDims + s] = r.coeffs[srcSlot];
            }
        }
        out.addRow(std::move(nr));
    }
    return out;
}

/// Adds tie rows x_j - f_j(d) == 0 (x at xBase, symbols at symBase).
[[nodiscard]] Result<void> addTieRows(const AffineMap& map, uint32_t xBase,
                                      uint32_t symBase, Polyhedron& target) {
    for (uint32_t j = 0; j < map.nOut; ++j) {
        const AffineExpr& f = map.outputs[j];
        if (!f.space.sameAs(map.inSpace)) {
            return err(ErrorCode::InvalidArgument,
                       "map tie row: expression space mismatch");
        }
        ConstraintRow tie;
        tie.isEquality = true;
        tie.coeffs = SmallVector<int64_t, 8>(target.space.totalVars(), 0);
        tie.coeffs[xBase + j] = 1;
        for (uint32_t d = 0; d < map.inSpace.nDims; ++d) {
            tie.coeffs[d] -= f.coeffs[d];
        }
        for (uint32_t s = 0; s < map.inSpace.nSyms; ++s) {
            tie.coeffs[symBase + s] -= f.coeffs[map.inSpace.nDims + s];
        }
        tie.constant = -f.constant;
        target.addRow(std::move(tie));
        if (target.isEmptyFlag) return {};
    }
    return {};
}

}  // namespace

AffineMap AffineMap::identity(const VarSpace& space) noexcept {
    AffineMap m;
    m.inSpace = space;
    m.nOut = space.nDims;
    for (uint32_t d = 0; d < space.nDims; ++d) {
        m.outputs.push_back(AffineExpr::variable(space, d));
    }
    return m;
}

AffineMap AffineMap::constantMap(const VarSpace& space, int64_t v) noexcept {
    AffineMap m;
    m.inSpace = space;
    m.nOut = 1;
    m.outputs.push_back(AffineExpr::fromConstant(space, v));
    return m;
}

Result<SmallVector<int64_t, 8>> AffineMap::evalAt(
    const SmallVector<int64_t, 8>& vars) const noexcept {
    SmallVector<int64_t, 8> out;
    out.reserve(outputs.size());
    for (const AffineExpr& e : outputs) {
        MLK_TRY_VAR(v, e.evalAt(vars));
        out.push_back(v);
    }
    return out;
}

bool AffineMap::sameAs(const AffineMap& o) const noexcept {
    if (!inSpace.sameAs(o.inSpace) || nOut != o.nOut) return false;
    if (outputs.size() != o.outputs.size()) return false;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!exprEqual(outputs[i], o.outputs[i])) return false;
    }
    return true;
}

Result<AffineMap> mapCompose(const AffineMap& outer,
                             const AffineMap& inner) noexcept {
    if (outer.inSpace.nDims != inner.nOut) {
        return err(ErrorCode::InvalidArgument,
                   "mapCompose: outer domain rank != inner output rank");
    }
    if (outer.inSpace.nSyms != inner.inSpace.nSyms) {
        return err(ErrorCode::InvalidArgument,
                   "mapCompose: symbol counts differ");
    }
    AffineMap out;
    out.inSpace = inner.inSpace;
    out.nOut = outer.nOut;
    for (const AffineExpr& oe : outer.outputs) {
        AffineExpr acc = AffineExpr::fromConstant(inner.inSpace, oe.constant);
        for (uint32_t j = 0; j < inner.nOut && j < inner.outputs.size(); ++j) {
            const AffineExpr& ie = inner.outputs[j];
            MLK_TRY_VAR(scaled, exprScale(ie, oe.coeffOf(j)));
            MLK_TRY_VAR(sum, exprAdd(acc, scaled));
            acc = std::move(sum);
        }
        if (!acc.withinLimits()) {
            return err(ErrorCode::ResourceExhausted,
                       "mapCompose overflows engine magnitude limit");
        }
        out.outputs.push_back(std::move(acc));
    }
    return out;
}

Result<PresburgerSet> mapImage(const PresburgerSet& set,
                               const AffineMap& map) noexcept {
    if (set.space.nDims != map.inSpace.nDims ||
        set.space.nSyms != map.inSpace.nSyms) {
        return err(ErrorCode::InvalidArgument,
                   "mapImage: set space != map domain space");
    }
    VarSpace prod;
    prod.nDims = map.inSpace.nDims + map.nOut;
    prod.nSyms = map.inSpace.nSyms;
    const uint32_t symBase = map.inSpace.nDims + map.nOut;
    PresburgerSet prodSet;
    prodSet.space = prod;
    Polyhedron p;
    p.space = prod;
    bool anyLive = false;
    bool empty = false;
    for (const Polyhedron& src : set.disjuncts) {
        if (src.isEmptyFlag) continue;
        MLK_TRYV(embedSystem(src, 0, symBase, p));
        anyLive = true;
        if (p.isEmptyFlag) {
            empty = true;
            break;
        }
    }
    if (!anyLive || empty || p.isEmptyFlag) {
        return PresburgerSet::empty(VarSpace{map.nOut, map.inSpace.nSyms});
    }
    MLK_TRYV(addTieRows(map, map.inSpace.nDims, symBase, p));
    if (p.isEmptyFlag) {
        return PresburgerSet::empty(VarSpace{map.nOut, map.inSpace.nSyms});
    }
    prodSet.disjuncts.push_back(std::move(p));
    SmallVector<bool, 8> keep(prod.nDims, true);
    for (uint32_t d = 0; d < map.inSpace.nDims; ++d) keep[d] = false;
    MLK_TRY_VAR(projected, prodSet.existsProject(keep));
    // Compact: x_j lives at slot nIn + j.
    SmallVector<uint32_t, 8> keptDimSlots;
    for (uint32_t j = 0; j < map.nOut; ++j) {
        keptDimSlots.push_back(map.inSpace.nDims + j);
    }
    Polyhedron finalSys = remapToDimFirst(projected.disjuncts[0], keptDimSlots,
                                          symBase, map.inSpace.nSyms,
                                          map.nOut);
    PresburgerSet image;
    image.space = VarSpace{map.nOut, map.inSpace.nSyms};
    image.disjuncts.push_back(std::move(finalSys));
    return image;
}

Result<PresburgerSet> mapPreimage(const PresburgerSet& set,
                                  const AffineMap& map) noexcept {
    if (set.space.nDims != map.nOut ||
        set.space.nSyms != map.inSpace.nSyms) {
        return err(ErrorCode::InvalidArgument,
                   "mapPreimage: set space != map range space");
    }
    VarSpace prod;
    prod.nDims = map.inSpace.nDims + map.nOut;
    prod.nSyms = map.inSpace.nSyms;
    const uint32_t symBase = map.inSpace.nDims + map.nOut;
    PresburgerSet prodSet;
    prodSet.space = prod;
    Polyhedron p;
    p.space = prod;
    bool anyLive = false;
    bool empty = false;
    for (const Polyhedron& src : set.disjuncts) {
        if (src.isEmptyFlag) continue;
        MLK_TRYV(embedSystem(src, map.inSpace.nDims, symBase, p));
        anyLive = true;
        if (p.isEmptyFlag) {
            empty = true;
            break;
        }
    }
    if (!anyLive || empty || p.isEmptyFlag) {
        return PresburgerSet::empty(VarSpace{map.inSpace.nDims,
                                             map.inSpace.nSyms});
    }
    MLK_TRYV(addTieRows(map, map.inSpace.nDims, symBase, p));
    if (p.isEmptyFlag) {
        return PresburgerSet::empty(VarSpace{map.inSpace.nDims,
                                             map.inSpace.nSyms});
    }
    prodSet.disjuncts.push_back(std::move(p));
    SmallVector<bool, 8> keep(prod.nDims, true);
    for (uint32_t d = 0; d < map.nOut; ++d) {
        keep[map.inSpace.nDims + d] = false;
    }
    MLK_TRY_VAR(projected, prodSet.existsProject(keep));
    // Compact: d_j stays at slot j (already first).
    SmallVector<uint32_t, 8> keptDimSlots;
    for (uint32_t d = 0; d < map.inSpace.nDims; ++d) {
        keptDimSlots.push_back(d);
    }
    Polyhedron finalSys = remapToDimFirst(projected.disjuncts[0], keptDimSlots,
                                          symBase, map.inSpace.nSyms,
                                          map.inSpace.nDims);
    PresburgerSet pre;
    pre.space = VarSpace{map.inSpace.nDims, map.inSpace.nSyms};
    pre.disjuncts.push_back(std::move(finalSys));
    return pre;
}

}  // namespace mlk::poly
