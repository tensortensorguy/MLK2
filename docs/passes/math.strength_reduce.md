# `math.strength_reduce`

**Category:** math · **Tier support:** T1-T3 · **Kill switch:** yes (Rule 60/150)

## Purpose
Strength reduction with a bit-exact legality THEOREM:

- `div(x, c) -> mul(x, 1/c)` for compile-time F64 constants `c` that are
  powers of two. For `c = ±2^e` the reciprocal is exactly representable,
  so both forms are the SAME real number and IEEE rounding — a function
  of the real result alone — makes them bit-identical for every input:
  subnormals (both scale the exponent), overflow (same real, same
  saturated result), signed zeros (same sign).

Guard rails: the reciprocal must itself be exactly representable
(`c = 2^-1074` overflows to +inf where `x/c` stays finite — skipped), and
the rewrite applies to F64 results only today (the evaluation precision
must make both `c` and `1/c` exact). `pow(x, 2) -> x*x` is deliberately
NOT here: it is bit-exact only if the evaluator's pow is correctly rounded
for that case — a libm property, not an IEEE one.

## Inputs
A verified Math IR graph (with types inferred), PassContext (domain profile, accuracy contract, budget, kill switches, diagnostics).

## Outputs
Transformed graph (new values; originals recoverable — Rule 21) and pass result metadata.

## Required analyses
See docs/pass_registry.md (contract `required` list per pass).

## Produced analyses
See docs/pass_registry.md (contract `produced` list per pass).

## Legality conditions
No rewrite without legality conditions (Rule 33). Typed FP-constant probes
(`isFpConst`) — int constants are never matched (round-23 miscompile class).

## Accuracy impact
Bit-exact (theorem + differential verification over subnormal/zero/edge
samples in `tests/unit/unit_passes_math.cpp`).

## Performance impact
Cheaper ops.

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
