// MLK+ polyhedral engine — SCoP model and KernelModule extraction.
//
// A SCoP (Static Control Part) is a loop nest where every bound is a
// compile-time constant (after workload specialization) with unit step and
// every memory access is an affine function of the nesting loop variables.
// The extractor walks a KernelModule and lifts its affine region into:
//   statements    — one per Compute+Store pair (the MLK kernel statement
//                   convention; see docs/kernel_abi.md),
//   domains       — full-rank Presburger boxes over the scoping depth;
//                   statements at shallower depth pin deeper dims to 0
//                   (init-then-accumulate fusion semantics, standard in
//                   polyhedral frameworks),
//   accesses      — flat (row-major) affine index maps per buffer.
//
// Eligibility contract (documented, conservative):
//   - unit steps, constant bounds within kRationalMagnitudeLimit,
//   - only Loop/Compute/Store nodes inside the region (Call/Barrier/Guard/
//     Trace/Alloc/Copy make the kernel poly-ineligible),
//   - statement/access/depth budgets from core/constants.h (Rule 10),
//   - one affine region per module: the WHOLE node tree must qualify
//     (poly.synth produces exactly that; mixed kernels run the baseline).
// Non-eligible kernels are never modified — the baseline kernel remains
// the fallback (Rules 62/102/115).
#pragma once

#include <string>

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/poly/affine_map.h"
#include "mlk/poly/int_set.h"
#include "mlk/support/kernel_ir.h"

namespace mlk::poly {

/// One memory reference: flat = flatIndex applied to the statement's dims.
struct MemoryAccess {
    uint32_t bufferId{constants::kInvalidId};
    bool isWrite{false};
    AffineMap flatIndex{};  // inSpace {scopDepth, 0}, nOut == 1
};

/// One statement instance space + kernel payload.
struct Statement {
    uint32_t id{0};
    VarSpace space{};  // {scopDepth, 0}
    PresburgerSet domain{};
    SmallVector<MemoryAccess, 4> accesses{};
    /// Kernel payload replayed by codegen (Rule 24: serializable form).
    SmallVector<KernelExpr, 8> exprs{};
    uint32_t storeBuffer{constants::kInvalidId};
    SmallVector<int64_t, 4> storeCoeffs{};  // flat target coefficients
    int64_t storeOffset{0};
    AccumMode accum{AccumMode::None};  // store update mode (Add/Max are
                                       // read-modify-write reductions)
    uint32_t origOrder{0};  // program order tie-break (determinism, 53)
    uint32_t depth{0};      // own nest depth (dims that actually vary)
    /// Constant bounds per OWN dim [lower, upper] (inclusive), recorded at
    /// extraction; domains are rebuilt at full scoping rank from these.
    SmallVector<int64_t, 4> ownLower{};
    SmallVector<int64_t, 4> ownUpper{};
};

/// The extracted static control part of a kernel module.
struct Scop {
    VarSpace space{};  // {scopDepth, 0}; symbols must be specialized
    SmallVector<Statement, 4> statements{};
    SmallVector<SymbolId, 8> dimVars{};  // loop var symbol per dim
    uint32_t depth{0};
};

/// Extracts the SCoP of `kernel`. Errors carry an actionable reason
/// (Rule 67); callers keep the baseline kernel on any error.
[[nodiscard]] Result<Scop> extractScop(const KernelModule& kernel,
                                       SymbolTable& symbols);

}  // namespace mlk::poly
