# MLK+ Effect System

Rule 140: the IR has an explicit effect model; passes must not reorder
effects without proof.

## Effect classes

Pure, MemoryRead, MemoryWrite, DomainStateMutation, RandomState, SolverState,
IO, FFI, Allocation, Tracing, ApproximationSwitch, Exception, Guard,
ReferenceBarrier, PhysicalPlacement.

## Rules

- Effects are inferred at construction from the op table
  (`effect_inference.cpp`) and validated by `ir.verify`.
- Native boundary ops (`native_to_math_ref`, `math_to_native_ref`) carry FFI
  effects: the boundary is opaque unless proven otherwise (Rule 112).
- The graph maintains an ordered effect chain; `killNode` splices it, and the
  verifier rejects continuity violations and pure nodes inside the chain
  (Rule 145).
- Only provably pure nodes may be constant-folded, CSE'd, or DCE'd
  (Rules 87, 92).
- Mathematical exceptions are values and control-flow edges, not native
  exceptions (Rule 93).
