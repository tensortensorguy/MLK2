# MLK+ Kernel ABI

## CPU kernel contract

The CPU backend realizes a KernelModule in two artifact forms:

1. **Serialized KernelModule** (JSON, `mlk-kernel` format, versioned) —
   executed by the in-process tree-walking executor with schedule parameters
   (tile/vector width/threads) from the module (Rule 54).
2. **C++ source** (`mlkc compile --emit=cpp`) — a standalone translation
   unit exposing `extern "C" void mlk_kernel(const double* x, const double*
   y, double* out, int64_t n)`, suitable for out-of-process compilation with
   clang/gcc (spec §12 first backend path).

## Input binding order

Placeholders/Variables bind to user scalars/buffers in value-id order of
appearance. This order is part of the ABI and is stable for a given graph
version.

## Publication, W^X, and patching notes

The MVP backend does NOT generate machine code in-process: publication of a
new realization is an atomic pointer swap of an immutable, heap-published
object (release/acquire — Rules 117, 129). Old realizations stay alive until
quiescence via shared ownership (Rule 119). Because no executable pages are
produced, Rule 116 (W^X) is satisfied trivially; if dynamic code generation
is added later, write-then-execute or dual-mapped pages are mandatory, and
Rule 118 patching rules apply.

## Fallback metadata (Rule 100)

Every compilation emits fallback metadata: Tier-0 re-entry marker,
GraphState snapshots for guards, and dependency lists. A compilation without
fallback metadata is incomplete and rejected.
