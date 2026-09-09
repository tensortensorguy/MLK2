# MLK+ Architecture

**Mathematics first. Representation second. Hardware third.**

MLK+ is a multi-domain mathematical graph compiler, autotuner, superoptimizer,
and runtime system. Tensor is one domain built on top of a general
mathematical graph — not the identity of the IR.

## The pipeline

```
MathGraph (what does it mean?)
    |
Property/Proof IR (what is legally transformable?)
    |
Strategy IR (which algorithm/method?)
    |
Schedule IR (how are loops/vectors/threads organized?)
    |
Physical IR (where do values live and move?)
    |
Kernel IR (lower.to_kernel_ir)
    |
Backend (CPU C++ emission / interpreter)
```

## Realization tiers (spec Part I)

| Tier | Name | Trigger | Contents |
|---|---|---|---|
| 0 | Reference interpreter / oracle | always available | maximum fidelity, full observability |
| 1 | Baseline kernel compiler | first use / heat | conservative canonicalization + lowering |
| 2 | Optimizing JIT / autotuner | sustained heat + profile data | full pass pipeline, measured tuning |
| 3 | AOT / supercompiled (compile=INF) | static proofs + certificates | e-graph saturation, superoptimizer plugins, proofs |

Tier transitions are governed by proofs, profile data, benchmark data,
accuracy contracts, and static guarantees — never by arbitrary timeouts.
Every transition is recorded in telemetry (Rule 138). Tier 0 is the
universal correctness fallback (Rule 115).

## Component map (layout spec §1)

```
compiler/   mlk_core, mlk_ir, mlk_type, mlk_property, mlk_effect, mlk_proof,
            mlk_verifier, mlk_pass_core, mlk_passes, mlk_pipeline, mlk_cost
runtime/    mlk_runtime (GraphState, fallback, execution, telemetry, cache)
autotuner/  mlk_autotune (declarative search, verify-first benching)
superopt/   mlk_superopt (certificated candidate generators)
backends/   mlk_backend_interpreter, mlk_backend_cpu (+ C++ emitter)
tools/      mlkc, mlk-verify, mlk-tune, mlk-bench, mlk-profile, mlk-replay,
            mlk-generate-profiles
tests/      unit, differential, fallback, fuzz, security, replay, golden
```

Dependency flow is strict and one-directional; backends consume lowered IR
plus a target description and never reach into compiler internals.

## Core abstractions

- **Value** — a typed mathematical object: `MathType` (domain, algebra class,
  differentiability, dtype, optional Shape/TensorDescriptor/FunctionSig),
  `FactSet` (tri-state properties), version, stable content hash.
- **Node** — a mathematical transformation (`MathOp`): arithmetic, math
  functions, tensor ops, first-class calculus (Derivative/Integral/Gradient/
  Limit), abstract ops, explicit conversions (Rule 39).
- **MathGraph** — index-based SSA-like graph (Rule 15) with effect chain
  (Rule 140), equivalence recording (Rule 21), and versioned mutations
  (Rule 89).
- **MathGraph / Strategy / Physical separation** — hardware details live in
  the physical layer and reference the math graph; never the reverse
  (Rule 23).
- **Math Domain Profile** — versioned domain semantics + capability flags;
  the core compiler is domain-agnostic (Part 0; Rule 28).

## Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Everything builds with `-std=c++26 -fno-exceptions -fno-rtti` (Rules 6, 8)
and is hermetic by default (Rule 160).
