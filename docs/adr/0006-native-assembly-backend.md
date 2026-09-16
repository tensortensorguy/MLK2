# ADR-0006: Native artifacts via an out-of-process assembly backend

Status: Accepted (amends ADR-0003; ADR-0003's core decision stands)

## Context
The polyhedral layer completed schedule -> codegen -> interpreted
execution with bit-exact verification. The remaining backend gap was
native execution: the C++ emitter path existed, but the project needed
the artifact chain to reach ASSEMBLY (the compiler's own lower half,
not the system compiler's optimizer) while ADR-0003 forbids in-process
machine codegen (W^X Rule 116, patching Rule 118, JIT-spraying
Rule 123).

## Decision
`mlk_backend_cpu` emits STANDALONE ARTIFACTS as text — C++ source and
x86-64 AT&T assembly — from the same KernelModule under one ABI
(multi-dim status-returning form and legacy 1-D form; fixed scalars
tail). The x86-64 emitter lowers the polyhedral schedule ITSELF:
affine loop bounds, ElemIdx addressing, accumulate stores, affine
guards, and SSE2 scalar IEEE-754 operations in the executor's exact
evaluation order (no FMA contraction), so compiled artifacts are
bit-exact against the buffer executor by construction. A backend
driver builds the artifact OUT-OF-PROCESS (cc -shared; C++ artifacts
compile under -fno-exceptions/-fno-rtti to honor the repo contract)
and loads it with dlopen.

## Consequences
ADR-0003's no-in-process-codegen decision is preserved: no emitter
owns executable pages; dlopen maps a toolchain-built file like any
plugin, so Rules 116/118/123 stay satisfied. The parallel/vector
marks are recorded in the artifact header (Rule 148) and the assembly
form executes sequentially (deterministic because parallel rows write
disjoint slabs); OpenMP pragmas ship in the C++ form. Differential
tests build real shared objects and compare every output bit-exactly
(walker vs C++ vs assembly). Honest capability boundary: poly7 Sin
families and speculative guards are rejected by the assembly form;
the mlk-graph text format still cannot round-trip tensor descriptors
(upstream serializer), so tensor kernels enter via the GraphBuilder.
