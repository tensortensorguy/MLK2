# ADR-0002: The MLK_TRY macro family (Rule 26 vs Rule 69)

Status: Accepted

## Context
Rule 69 bans C-style macros for logic; Rule 26 explicitly sanctions "a
custom TRY() macro that compiles down to a single branch" for zero-cost
error propagation with std::expected.

## Decision
Exactly two logic macros exist, both sanctioned by Rule 26:
- MLK_TRYV(expr) — propagate a Result<Ok>'s error.
- MLK_TRY_VAR(name, expr) — declare `name` bound to the success value,
  propagating the error.
Both expand to a single branch with [[unlikely]], use fully-qualified
::mlk::Error (safe at any call site), and are the ONLY logic macros in the
codebase (Rule 69 otherwise).

## Consequences
Uniform, branch-minimal error propagation; grep for #define finds exactly
the sanctioned macros plus header guards.
