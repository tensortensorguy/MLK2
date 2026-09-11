// MLK+ polyhedral engine — schedule → Kernel IR code generation.
//
// Emits a new KernelModule node tree from the Scop, the affine schedule
// and the tiling record (see docs/polyhedral_spec.md §codegen).
//
// Supported schedule class (CLAST-lite; conservative bail otherwise):
//   - levels where every alive statement is CONSTANT for the level are
//     separators or hoisting/guard points: statements emit before/after
//     the level's loop when their folded value falls outside the range,
//     and RE-ENTER the loop under an affine-equality Guard node when it
//     falls inside (the statement's remaining rows continue in the
//     fused deeper loops — the guarded init in a fused GEMM),
//   - levels where statements vary emit ONE shared loop over the row's
//     pivot dim (fusion); every statement is either pivot-varying or
//     row-constant with the pivot PINNED for it (its instances occupy
//     exactly one coordinate — anything else bails),
//   - tiled band rows emit a TILE loop over floor(value / t) and a
//     POINT loop with bounds clipped to the statement box (affine in
//     the tile index through KernelNode's affine-bounds extension).
// This covers fusion, fission, interchange, skew-shaped schedules, the
// parallel / vector markings produced by the scheduler, and guarded
// statement re-entry for the poly.synth kernel class. Non-realizable
// rows (a live pair whose pivot-coordinate order runs backward) are
// rejected by the scheduler's realizability gate — the baseline kernel
// stays in place otherwise (Rules 62/102).
//
// Emission is level-by-level over the statement group:
//   - constant statements (domain fully pinned at this level) with value
//     < the loop lower bound are emitted BEFORE the loop, values > the
//     upper bound AFTER it (our synth kernels pin reduced dims to 0, so
//     the before-case is the one that occurs; the inside case bails),
//   - varying statements sharing the same ±e_d row share ONE loop
//     (fusion); statements differing in the row's dimension FISSION into
//     separate sequential nests,
//   - tiled band rows emit a TILE loop over floor(value / t) and a POINT
//     loop with bounds clipped to the statement box (affine in the tile
//     index through KernelNode's affine-bounds extension).
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/poly/scop.h"
#include "mlk/poly/tile.h"
#include "mlk/support/kernel_ir.h"

namespace mlk::poly {

/// Rebuilds the kernel node tree from the schedule. The returned module
/// keeps the baseline's buffers and schedule params; only the node forest
/// is regenerated. Errors leave the baseline untouched.
[[nodiscard]] Result<KernelModule> emitScheduledKernel(
    const Scop& scop, const PolySchedule& sched, const TiledInfo& tiled,
    const KernelModule& baseline, SymbolTable& symbols);

}  // namespace mlk::poly
