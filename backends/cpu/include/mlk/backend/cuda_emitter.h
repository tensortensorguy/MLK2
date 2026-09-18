// MLK+ CUDA source emitter (docs/polyhedral_spec.md #GPU-backend; ADR-0008,
// amending ADR-0006). Emits the FULL KernelModule forest as standalone CUDA
// C++ (.cu text) — legacy 1-D elementwise kernels and polyhedral multi-dim
// nests alike — under the same artifact contract as the C++/assembly
// emitters: the exact buffer-executor semantics, compiled out-of-process,
// loaded with dlopen, invoked through the SAME mlk_kernel ABI.
//
// Publication model (ADR-0003/0006, unchanged): there is NO in-process
// machine codegen anywhere — nvcc is spawned exactly like `cc` today and
// the artifact is a plain .so mapped by the system loader.
//
// Artifact shape (multi-dim modules):
//   - ONE __global__ device kernel per forest ROOT (unreferenced top-level
//     node), named mlk_dev_<idx>, each taking the FULL bindable table
//     (ptr + dims per buffer, temps included) + the scalars pair + an
//     int* status slot. Unused parameters are (void)-cast, never dropped.
//   - The host wrapper `extern "C" int mlk_kernel(...)` manages device
//     memory explicitly behind the same Result-mapped ABI: cudaMalloc per
//     bindable buffer + per dims array + scalars + status; H2D for inputs
//     AND outputs (accumulate/Max stores read the output slot — walker
//     semantics); device temps materialized from the module's STATIC temp
//     dims, zero-initialized (cudaMemset, the walker's temp model, the
//     same kKernelTempElementsLimit enforced on the host side); one launch
//     per root IN ROOT ORDER with cudaDeviceSynchronize between launches
//     (the walker executes roots sequentially in id order — the launch
//     sequence preserves that program order exactly); status read back and
//     mapped to the ABI int (code 1 = negative store flat, the walker's
//     InvalidGraph); D2H for outputs; cudaFree of everything allocated.
//   - TEMP table entries stay in the HOST signature for ABI uniformity
//     (the driver's exactly-typed dispatch is arity-based) but are IGNORED
//     by the artifact: device scratch is materialized internally from the
//     module's static temp dims (documented boundary, not a silent drop).
//
// Thread mapping from proven marks (spec #GPU-backend): the outermost
// chain of parallel-marked loops with RECTANGULAR bounds (constants or
// buffer-dim refs only — a grid launch needs constant trip counts) is
// collapsed into a 1-D grid, one thread per instance tuple, decomposed
// row-major (outermost first — the walker's stack order). Every deeper
// level — serial, carried, tile/point, guards, split segments — executes
// as a serial loop inside the thread in the walker's exact order, so the
// differential is bit-exact BY CONSTRUCTION for the ops the device
// computes identically. A root whose first loop is not parallel-marked
// (or a root that is a Guard/compute region) launches with ONE thread and
// records the boundary in the header (Rule 148: recorded, never silent).
//   - Affine-equality guards and piecewise-split segment bounds evaluate
//     per-thread with the thread's own coordinates: segments that exclude
//     it have empty ranges and cost zero iterations — the split form
//     needs no special handling on the device path.
//   - Negative store flats are checked at the store site exactly like the
//     assembly artifact (status = 1; the driver maps nonzero to an error
//     Result and the caller discards partial outputs — same observable
//     contract).
//
// Bit-exactness boundary (declared, never silently relaxed — spec
// #GPU-backend): the build requires `--fmad=false` (separate mul/add,
// matching the walker and the asm emitter) and each Compute op is emitted
// as its own expression, so modules over {Add, Sub, Mul, Div, Neg, Sqrt,
// Rsqrt, poly7-Sin} are bit-exact vs the walker (IEEE-754 correctly
// rounded mul/add/div/sqrt; floor/fmod exact; the poly7 snapshot is pure
// mul/add). Modules touching device libm transcendentals (Exp, Log, Sin
// under libm family, Cos, Tan, Tanh, Erf, Gelu, Pow's non-fast form) are
// NOT claimed bit-exact: device libm does not guarantee host libm bit
// patterns, so those modules are recorded with an ULP-BOUNDED policy —
// the differential for them is measured and reported (bench --paths=cuda
// records the max ULP distance), never asserted and never silently
// relaxed (Rule 90 honesty). cudaArtifactBitExactPolicy() exposes the
// emitter's classification so drivers/tests/bench can gate honestly.
//
// Unsupported nodes fail the emission honestly (Result error): Call
// (lowered by poly.synth first — Rule 121), speculative (non-affine)
// Guards, AllocBuffer/CopyBuffer, non-unit loop steps, legacy dynamic
// bounds inside multi-dim modules, dynamic-sized temp buffers, more
// roots than kCudaMaxDeviceKernels, operands referencing unbound
// buffers, and fused chains over the temp-slot cap.
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/kernel_ir.h"

namespace mlk {

/// Emits standalone CUDA C++ for a KernelModule (AOT artifact form;
/// compile out-of-process with nvcc through the GPU driver config —
/// backend_driver.h buildGpuKernelArtifact).
[[nodiscard]] Result<std::string> emitCudaSource(const KernelModule& kernel,
                                                 SymbolTable& symbols);

/// The emitter's exactness classification for the module (see the header
/// contract above): true when every Compute op is in the device-exact set
/// (IEEE mul/add/div/sqrt + the poly7 snapshot) so the differential vs
/// the walker is bit-exact by construction; false when any device-libm
/// transcendental is present (differential is ULP-bounded — measured and
/// recorded, never asserted, never silently relaxed). The symbol table
/// resolves the "poly7" family text with the SAME check apply() uses.
[[nodiscard]] bool cudaArtifactBitExactPolicy(
    const KernelModule& kernel, SymbolTable& symbols) noexcept;

}  // namespace mlk
