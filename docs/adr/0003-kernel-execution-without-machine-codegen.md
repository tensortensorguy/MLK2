# ADR-0003: Kernel execution without in-process machine codegen

Status: Accepted

## Context
Spec section 12: "Do not emit machine code directly from the mathematical
graph." Dynamic codegen would pull in W^X page management (Rule 116),
concurrent patching (Rule 118), and JIT-spraying defenses (Rule 123).

## Decision
The MVP CPU backend executes the KernelModule in-process (tree-walk with
schedule parameters) and emits portable C++ source for AOT compilation by a
system compiler. No executable pages are created; publication is an atomic
pointer swap of immutable objects (Rules 117/119).

## Consequences
W^X/patching/spraying rules are satisfied trivially and documented honestly;
the C++ emission path preserves the upgrade path to LLVM/machine backends
without ABI breakage (kernel_abi.md).
