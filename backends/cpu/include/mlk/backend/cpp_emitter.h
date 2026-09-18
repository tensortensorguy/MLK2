// MLK+ C++ source emitter (spec §12: KernelIR -> C++ template/source for
// clang/gcc compilation). Emits the FULL KernelModule forest as standalone
// C++ — legacy 1-D elementwise kernels and polyhedral multi-dim nests alike
// — with the exact buffer-executor semantics (docs/kernel_abi.md,
// docs/polyhedral_spec.md §backend):
//   - loop bounds: constant, affine (beginCoeffs/endCoeffs over the
//     enclosing var stack), and buffer-dim bounds (endBuf/endDim),
//   - fused Compute chains with ElemIdx affine addressing (row-major flat
//     = offset + sum coeffs[p]*var[p]) and ScalarParam runtime scalars,
//   - affine stores (overwrite and accumulate) and affine-equality guards,
//   - parallel/vectorHint marks as OpenMP pragmas under #ifdef _OPENMP
//     (inert without the flag — Rule 148: features are gated and recorded;
//     the marks are proven by the scheduler, so OpenMP's dynamic schedule
//     choice cannot change results: parallel rows write disjoint slabs),
//   - math families: "poly7" is emitted as an inlined snapshot of
//     mlk/support/math_families.h (the artifact must be standalone; the
//     header remains the single source the snapshot is taken from).
//
// Two ABI forms, chosen by the same multi-dim detection the buffer
// executor uses (executeKernelOnBuffers):
//
//  1. Multi-dim/polyhedral modules — returns an error code so structural
//     ABI violations surface like the executor's Result (no exceptions):
//
//       extern "C" int mlk_kernel(
//           const double* A, const int64_t* A_dims,     // per buffer,
//           ...                                          // table order,
//           double* C, const int64_t* C_dims,            // writable ptr
//                                                        // iff stored,
//           const double* mlk_scalars,                   // FIXED ABI:
//           int64_t mlk_n_scalars)                       // always last
//
//     Buffer pointers are `const double*` exactly when no Store targets
//     the buffer; `_dims` arrays carry the buffer's logical dims. The
//     scalars pair is part of the fixed signature (pass an empty table
//     when the kernel has no ScalarParam — reads degrade to 0.0 exactly
//     like the walker's empty scalar vector); a fixed shape removes the
//     whole "did the tail make it into the call" mismatch class. The
//     dims of buffers with static dims are validated against the emitted
//     affine forms only structurally (the executor trusts the binder).
//
//  2. Legacy 1-D modules — the historical ABI extended with the same
//     fixed scalars tail (n = the shared dynamic bound io.elements):
//
//       extern "C" void mlk_kernel(const double* x, const double* y,
//                                  double* out, int64_t n,
//                                  const double* mlk_scalars,
//                                  int64_t mlk_n_scalars)
//
// Unsupported nodes fail the emission honestly (Result error): Call
// (lowered by poly.synth first), speculative (non-affine) Guards
// (ExecutionEngine's Rule-5 domain), AllocBuffer/CopyBuffer (not in the
// lowering/polyhedral emission alphabet), bare Stores, temp chains over
// kMaxTempSlots, and operands referencing unbound buffers.
#pragma once

#include "mlk/backend/slab_plan.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/kernel_ir.h"

namespace mlk {

/// Emits standalone C++ source for a KernelModule (AOT artifact form).
[[nodiscard]] Result<std::string> emitCppSource(const KernelModule& kernel,
                                                SymbolTable& symbols);

/// Slab-mirror form (docs/polyhedral_spec.md #GPU-backend, round 21):
/// the same emission with slab preload blocks — for every root with a
/// plan, the root's read-only buffers are copied into a heap scratch
/// block (RAII guard, leak-free on every exit) before the root's
/// loops and all their reads are redirected to the copy. This is the
/// BEHAVIORAL verification vehicle for the slab value logic: the
/// artifact is differentially tested bit-exact against the walker
/// (the CUDA text shares the plan and the redirect, so the value
/// logic is pinned by construction; the device-specific prologue is
/// structure-verified separately). With `opts.sharedMemSlabs == false`
/// this is byte-identical to the two-argument form. ABI code 5 =
/// slab allocation failure (recorded in the artifact header when the
/// option is on).
[[nodiscard]] Result<std::string> emitCppSource(
    const KernelModule& kernel, SymbolTable& symbols,
    const SlabEmitOptions& opts);

/// The buffer executor's multi-dim routing decision
/// (executeKernelOnBuffers): selects the ABI form of an emitted artifact
/// AND the invocation form of the backend driver. Shared by both
/// emitters and the driver so the artifact, the driver, and the walker
/// can never disagree about which contract a module executes under.
[[nodiscard]] bool isMultiDimModule(const KernelModule& kernel) noexcept;

}  // namespace mlk
