// MLK+ polyhedral engine — schedule → Kernel IR code generation.
//
// Emits a new KernelModule node tree from the Scop, the affine schedule
// and the tiling record (see docs/polyhedral_spec.md §codegen).
//
// Supported schedule class (documented; conservative bail otherwise):
//   each schedule row used by a statement group is either CONSTANT for
//   every statement in the group (a separator level: statement-order
//   emission) or a pure permutation ±e_d of one dimension (a loop level).
// This covers fusion, fission, interchange, tiling and the parallel /
// vector markings produced by the scheduler for the poly.synth kernel
// class. Non-unimodular rows (skewed schedules) bail — the baseline
// kernel stays in place (Rules 62/102).
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
