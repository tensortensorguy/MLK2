# `math.algebraic_simplify`

**Category:** math · **Tier support:** T1-T3 · **Kill switch:** yes (Rule 60/150)

## Purpose
Bit-exact FP sign/division identities on the Math IR — the complement set
to math.identity_elim (which owns x+0, x*1, int x*0, t(t(x))). Every rule
here is a theorem under IEEE-754 round-to-nearest, not a sample claim:

- `sub(x, +0.0) -> x` — subtracting positive zero preserves every bit.
- `div(x, +1.0) -> x` — division by one is exact.
- `div(x, -1.0) -> neg(x)` — magnitude unchanged, exact sign flip.
- `neg(neg(x)) -> x` — two exact sign flips restore every bit (zeros,
  infinities, NaN payloads included).

## Inputs
A verified Math IR graph (with types inferred), PassContext (domain profile, accuracy contract, budget, kill switches, diagnostics).

## Outputs
Transformed graph (new values; originals recoverable — Rule 21) and pass result metadata.

## Required analyses
See docs/pass_registry.md (contract `required` list per pass).

## Produced analyses
See docs/pass_registry.md (contract `produced` list per pass).

## Legality conditions
Only rewrites whose result is bit-identical for EVERY input (Rule 33).
The reassociation family (flatten/balance) is NOT here: it changes FP
results and no shipped profile opens that gate (ADR-0009). FP-constant
probes are typed (`isFpConst`) — int constants are never matched against
float identities (the miscompile class fixed in round 23).

## Accuracy impact
Bit-exact for every rule; no accuracy contract needed.

## Performance impact
Case-by-case.

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
