# MLK+ Kernel ABI

## CPU kernel contract

The CPU backend realizes a KernelModule in two artifact forms:

1. **Serialized KernelModule** (JSON, `mlk-kernel` format, versioned) —
   executed by the in-process tree-walking executor with schedule parameters
   (tile/vector width/threads) from the module (Rule 54).
2. **C++ source** (`mlkc compile --emit=cpp`; `mlk-poly emit --cpp`) — a
   standalone translation unit exposing the multi-form ABI documented in
   `backends/cpu/include/mlk/backend/cpp_emitter.h` (multi-dim modules
   return an `int` status; the legacy 1-D form is
   `void mlk_kernel(const double* x, const double* y, double* out,
   int64_t n, const double* mlk_scalars, int64_t mlk_n_scalars)`),
   suitable for out-of-process compilation with clang/gcc (spec §12
   first backend path).
3. **x86-64 assembly** (`mlk-poly emit --asm`) — the same contract as
   form 2 emitted as standalone AT&T-syntax x86-64 (System V AMD64,
   PIC, SSE2 scalar IEEE-754; see docs/polyhedral_spec.md §backend).
   The backend driver assembles it out-of-process (`cc -shared`) and
   loads it with dlopen: publication stays file-backed, no in-process
   machine codegen exists (ADR-0003/0006), so the W^X/patching notes
   below are unchanged.

## Input binding order

Placeholders/Variables bind to user scalars/buffers in value-id order of
appearance. This order is part of the ABI and is stable for a given graph
version.

## Publication, W^X, and patching notes

The MVP backend does NOT generate machine code in-process: publication of a
new realization is an atomic pointer swap of an immutable, heap-published
object (release/acquire — Rules 117, 129). Old realizations stay alive until
quiescence via shared ownership (Rule 119). Because no emitter-owned
executable pages are produced, Rule 116 (W^X) is satisfied trivially; the
native-artifact driver (forms 2-3) also creates no in-process codegen — the
external toolchain produces the .so and the system loader maps it, exactly
like a plugin (ADR-0006). If true dynamic code generation is added later,
write-then-execute or dual-mapped pages are mandatory, and Rule 118 patching
rules apply.

## Fallback metadata (Rule 100)

Every compilation emits fallback metadata: Tier-0 re-entry marker,
GraphState snapshots for guards, and dependency lists. A compilation without
fallback metadata is incomplete and rejected.
