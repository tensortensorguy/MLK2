# `accuracy.analyze`

**Category:** accuracy · **Tier support:** T1-T3 · **Kill switch:** yes (Rule 60/150)

## Purpose
Records the effective accuracy policy: approximation is enabled only when profile AND contract both allow (Rule 34).

## Inputs
A verified Math IR graph (with types inferred), PassContext (domain profile, accuracy contract, budget, kill switches, diagnostics).

## Outputs
Transformed graph (new values; originals recoverable — Rule 21) and pass result metadata.

## Required analyses
See docs/pass_registry.md (contract `required` list per pass).

## Produced analyses
See docs/pass_registry.md (contract `produced` list per pass).

## Legality conditions
Contract must exist for any downstream approximation.

## Accuracy impact
Central gate for accuracy.

## Performance impact
None.

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
