# `tensor.contraction_path`

**Category:** tensor · **Tier support:** T1-T3 · **Kill switch:** yes (Rule 60/150)

## Purpose
Matrix-chain DP for contraction order (analysis; recording stage). Builds
ACTUAL chains (a matmul consuming another matmul's result as its left
input), derives the shared-dimension vector p[], runs the textbook DP over
p[], and attaches the result to the IR: each chain node carries its
subtree's optimal split (`contraction_split`), optimal FLOP cost
(`contraction_flops`), and the root additionally records the naive
left-deep cost (`contraction_flops_naive`) so the potential gap is
visible. The pass also emits a `contraction_chains` PerfCounter event.

## Inputs
A verified Math IR graph (with types inferred), PassContext (domain profile, accuracy contract, budget, kill switches, diagnostics, telemetry sink).

## Outputs
Transformed graph (new values; originals recoverable — Rule 21) and pass result metadata.

## Required analyses
See docs/pass_registry.md (contract `required` list per pass).

## Produced analyses
See docs/pass_registry.md (contract `produced` list per pass).

## Legality conditions
Static shapes required.

## Accuracy impact
None.

## Performance impact
FLOP-minimal ordering.

## Consumer status (honest boundary)
No lowering consumes the recorded order yet: realizing a non-left-deep
parenthesization reassociates FP matmul chains, which changes rounding and
therefore needs a Rule 33 accuracy contract (none shipped opens it). The
attrs are strategy data per spec §8.5; realization is tracked with the
reassociation-contract work (ADR-0009).

## Tier support
T1-T3 (see pipeline definitions in `mlk/pipeline/pipeline_runner.h` and spec §9).

## Budget
Bounded by PassBudget (node edits + fixpoint iterations); violations return BudgetExceeded and trigger fallback — never stalls (Rule 131).

## Kill switch
Every pass is independently disableable via the context kill-switch map; skipped passes are recorded in telemetry (Rules 60, 150).

## Tests
See `tests/unit/` and `tests/pass_golden/` for this pass's coverage; differential coverage in `tests/differential/` (Rule 43).

## Known limitations
MVP scope per layout spec §11 (MVP pass set); Tier-3 extensions (full rule tables, polyhedral scheduling, external proof engines) are tracked in the roadmap (docs/architecture.md).
