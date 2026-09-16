# ADR-0007: Certified, budgeted fast-kernel search

Date: 2026-09-17 (round 17)
Status: Accepted
Amends: none. Extends ADR-0005 (polyhedral engine) and ADR-0006
(native assembly backend) with a decision layer above both.

## Context

The uploaded OEIA/Refined/FastKernels axiom set (sections XIV-XVIII)
makes kernel runtime speed a first-class optimization objective while
keeping the repo's core invariants: every claim carries a checkable
certificate (Meta-Axiom 0.1), compile time is a hard kernel constraint
(Rule 17.5), and "fastest possible" may never be claimed without a
global lower bound plus exhaustive feasibility proof (Axiom 14.21).

The polyhedral layer already produces bit-exact transformed kernels and
the backend already builds native artifacts out-of-process; what was
missing is the budgeted SEARCH over kernel variants that selects one
certified winner and reports the full space.

## Decision

1. New component `mlk_fastkernel` (`fastkernel/`), dependency-ordered
   above `mlk_pipeline` and `mlk_backend_cpu` (Rule 70). It is a
   driver-level orchestrator: it introduces no IR, no pass, and no
   execution path of its own — a candidate is exactly a Tier2 compile
   under a declared tile knob plus a launch configuration plus an
   execution path (walker / native assembly / native C++).
2. Certificate bundles per candidate: bit-exact identity (differential
   vs the Tier1 reference), benchmarked runtime upper bound (declared
   method: median of reps after warmups), stage-accounted compile time,
   launch-coverage proof mirroring the executor's chunk arithmetic, and
   the artifact fingerprint for cached candidates.
3. Budget modes per Axiom 15.1 (same-comptime / bounded-extra /
   amortized) with the Axiom 15.6 regression guard implemented as a
   structured rejection, never a silent overrun. Auto-measured B_K^0
   uses max-of-3 declared samples (single-sample noise would decide
   admissibility by luck at the boundary).
4. Extra comptime is admissible only through the pure justification
   function `evaluateExtraComptimeJustification` (Axioms 14.9/15.7:
   absolute / amortized / Pareto forms); otherwise the search falls
   back to the same-comptime kernel.
5. Roofline-style lower bounds come from a DECLARED essential-work
   model plus measured environment peaks; unknown extents keep the
   model tri-state and the certificate is omitted honestly.
6. Claims are honest per Axiom 14.21: "fastest certified kernel within
   the declared searched space" only after exhaustive finite search
   over certified candidates; otherwise "best certified kernel in
   searched space"; "fastest possible" never.
7. Artifact cache validity is a completeness fingerprint (Axiom 14.22):
   kernel source hash, artifact kind, compiler, policy, budget mode,
   environment descriptor AND measured environment values. Reuse
   reports M^reuse = load time (Axiom 15.8).
8. The policy is exact (`bitexact-f64-cpu-v1`): under this policy the
   Axiom-14.17 numerical relaxation menu (reassociation, FMA,
   approximate reductions) is illegal, so a one-ulp divergence rejects
   the candidate.

## Consequences

- `mlk-poly autotune` exposes the search; `--json-out` produces the
  machine-auditable report (Axiom 14.20: the searched space, every
  rejection reason, every certificate).
- The tile-size dimension of the old roadmap ("poly_tile_size autotuner
  integration") is closed by the search; the remaining roadmap items
  are unchanged.
- The executor's threading constants moved to `constants.h`
  (`kKernelExecMaxThreads`, `kParallelChunkElements`) because the
  launch-coverage certificate must read the same values the executor
  acts on.
- `mlk_backend_cpu` gained two driver entry points used by the cache:
  `buildKernelArtifactInDir` (artifact pinned to a stable directory)
  and `loadKernelLibrary` (stage-5-only reuse).
- Known boundaries (roadmap): the searched space covers tile sizes,
  thread overrides and execution paths; scheduling-strategy variants
  (fusion/interchange choices) are chosen by the scheduler's exact
  order search, not the runtime search. Softmax-class kernels cannot
  enter native candidates until the temp-binding ABI exists, and the
  search reports them as `CapabilityUnsupported` rather than hiding
  them.

## Verification

- `tests/unit/unit_fastkernel.cpp` (9 cases): launch coverage
  (partition/disjointness/sequential/resource-violation), exact roofline
  model on a hand-built GEMM (O_min = 2*M*K*N, B_min exact), tri-state
  unknown extents, the three justification forms + refusal, a full
  same-comptime search asserting every certificate field, the
  budget-failure report, refused extra comptime, and cache reuse +
  environment-change invalidation.
- Full suite: 18/18 ctest green, -Werror clean, demo walker + native
  bit-exact (56 outputs, max|diff| = 0).
