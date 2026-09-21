# `math.identity_elim`

**Category:** math · **Tier support:** T1-T3 · **Kill switch:** yes (Rule 60/150)

## Purpose
Removes identity ops: x+0, x*1 (FP and INT), int x*0, transpose(transpose(x)). -0.0 semantics gated on the OPERAND's sign (the hazard), not the constant's: `x + (-0.0) -> x` is exact for every x and folds under any profile; `x + (+0.0) -> x` is wrong for x = -0.0 (IEEE gives +0) and folds only when the domain drops negative zero (`preserveNegativeZero = false`).

## Inputs
A verified Math IR graph (with types inferred), PassContext (domain profile, accuracy contract, budget, kill switches, diagnostics).

## Outputs
Transformed graph (new values; originals recoverable — Rule 21) and pass result metadata.

## Required analyses
See docs/pass_registry.md (contract `required` list per pass).

## Produced analyses
See docs/pass_registry.md (contract `produced` list per pass).

## Legality conditions
FP x+0 requires the operand-aware -0.0 policy (round-23 fix: the previous
gate tested signbit(constant), which blocked the sound x+(-0) form and
fired the unsound x+(+0) form). x*0 only for integers. Int constants are
probed through typed helpers (`isIntConst`/`isFpConst`) — the previous
untyped probe read int constants as 0.0 and folded add(x, 5) -> x
(certified miscompile, fixed with regression tests).

## Accuracy impact
Exact under stated policies.

## Performance impact
Removes ops.

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
