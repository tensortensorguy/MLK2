// MLK+ x86-64 assembly emitter (spec §12 backend path 3; see ADR-0003 as
// amended by ADR-0006). Emits the FULL KernelModule forest as standalone
// AT&T-syntax x86-64 assembly — legacy 1-D elementwise kernels and
// polyhedral multi-dim nests alike — with the exact buffer-executor
// semantics (docs/kernel_abi.md, docs/polyhedral_spec.md §backend):
//   - loop bounds: constant, affine (beginCoeffs/endCoeffs over the
//     enclosing var stack), and buffer-dim bounds (endBuf/endDim),
//   - fused Compute chains with ElemIdx affine addressing (row-major flat
//     = offset + sum coeffs[p]*var[p]) and ScalarParam runtime scalars,
//   - affine stores (overwrite and accumulate) and affine-equality guards,
//   - SSE2 scalar IEEE-754 double arithmetic (addsd/subsd/mulsd/divsd/
//     sqrtsd) plus libm PLT calls for transcendentals — the same
//     operations, in the same order, the buffer executor performs, so a
//     differential test against the walker is bit-exact by construction
//     (no FMA contraction is ever emitted; Rules 33/90).
//   - parallel/vectorHint marks are RECORDED in the header (Rule 148);
//     the assembly artifact executes sequentially. Parallel rows write
//     disjoint slabs, so any single-thread interleaving of the proven
//     schedule is deterministic and bit-exact.
//
// The ABI is the multi-form contract documented in cpp_emitter.h (the two
// emitters produce interchangeable artifacts):
//
//  1. Multi-dim/polyhedral modules:
//       extern "C" int mlk_kernel(
//           const double* A, const int64_t* A_dims,   // per bindable
//           ...                                        // buffer, table
//                                                      // order
//           const double* mlk_scalars,                 // FIXED ABI:
//           int64_t mlk_n_scalars)                     // always last
//     returning 0 on success and 1 when a store address form evaluates
//     negative (the walker rejects negative store flats with
//     InvalidGraph; the artifact reports the same violation through the
//     ABI int — the driver maps it back to an error Result).
//  2. Legacy 1-D modules:
//       extern "C" void mlk_kernel(...buffers..., int64_t n,
//                                  const double* mlk_scalars,
//                                  int64_t mlk_n_scalars)
//
// In BOTH forms the scalars pair is part of the fixed signature (pass an
// empty table when the kernel has no ScalarParam — reads degrade to 0.0
// exactly like the walker's empty scalar vector).
//
// Target framing: System V AMD64, position-independent (RIP-relative
// rodata, PLT calls), .note.GNU-stack emitted (non-executable stack).
// The artifact is assembled out-of-process by the backend driver
// (backend_driver.h) — no in-process machine codegen exists anywhere in
// this backend (ADR-0003/0006, Rule 116).
//
// Unsupported nodes fail the emission honestly (Result error): Call
// (lowered by poly.synth first — Rule 121), speculative (non-affine)
// Guards (ExecutionEngine's Rule-5 domain), AllocBuffer/CopyBuffer,
// non-unit loop steps, the "poly7" Sin family (inline family bodies are
// a roadmap item; use the C++ emitter for family snapshots), temp chains
// over kMaxTempSlots, legacy dynamic bounds inside multi-dim modules,
// and operands referencing unbound buffers.
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/kernel_ir.h"

namespace mlk {

/// Emits standalone x86-64 AT&T assembly for a KernelModule (AOT
/// artifact form; assemble out-of-process with the backend driver).
[[nodiscard]] Result<std::string> emitAsmSource(const KernelModule& kernel,
                                                SymbolTable& symbols);

}  // namespace mlk
