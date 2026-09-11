// MLK+ polyhedral engine — permutable-band detection and tiling.
//
// A PREFIX of schedule rows [0, bandEnd) forms a tileable band when, for
// every dependence still live inside the prefix, its distance stays
// non-negative on the refined slice at every band row (the Pluto LP
// already guarantees this — the check here is defense in depth) and the
// row is non-degenerate (the schedule value actually varies over at least
// one statement domain — tiling a constant row is meaningless).
//
// Tiling itself is realized at codegen time (poly.codegen): band row r is
// split into a TILE loop over floor(theta_r / t) and a POINT loop over the
// residual [tile*t, tile*t + t - 1] clipped to the statement's constant
// domain bounds. Floor monotonicity keeps tile(dst) >= tile(src) for
// every dependence with non-negative band distances, which is exactly the
// rectangular-tiling legality condition (Rules 33/36).
#pragma once

#include <cstdint>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/poly/dependence.h"
#include "mlk/poly/pluto.h"
#include "mlk/poly/scop.h"

namespace mlk::poly {

struct TiledInfo {
    bool tiled{false};
    uint32_t bandEnd{0};                 // rows [0, bandEnd) form the band
    SmallVector<int64_t, 12> tileSizes{};  // one per band row
};

/// Detects the maximal tileable band prefix. `tileSize` comes from the
/// poly_tile_size knob / kPolyDefaultTileSize (Rule 29).
[[nodiscard]] Result<TiledInfo> computeTiling(
    const Scop& scop, const SmallVector<Dependence, 16>& deps,
    const PolySchedule& sched, int64_t tileSize);

}  // namespace mlk::poly
