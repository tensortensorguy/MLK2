# MLK+ Superoptimizer Specification

Superoptimizers are modular, domain-gated candidate generators (Rules 51, 52;
spec §6/§13). Each candidate carries a certificate: origin, applied rules,
error bound, machine-checkable predicates, fallback plan. No opaque
candidate may be installed.

## Plugins

| Plugin | Searches | Verification |
|---|---|---|
| ScalarPeepholeSuperoptimizer | FMA formation, square forms, strength reduction | algebraic identity + FMA rounding predicate |
| MathFunctionApproximator | polynomial families + Cody-Waite range reduction | measured ULP bound vs libm oracle (Rule 50) |
| AlgebraicEGraphOptimizer | equivalence saturation + multi-extract | algebraic identity (e-class provenance) |

## Measured example

The shipped degree-13 sin family (Cody-Waite pi/2 hi/lo reduction,
correctly-rounded minimax coefficients) measures **max 2 ULPs** against
libm over [-pi, pi] including quadrant edges — the certificate records that
measured bound, and `approx.ulp_verify` re-checks it (Rule 50: candidates
are verified before benchmarking).

## compile=INF (Rule 53)

The offline supercompilation mode uses large search budgets but remains
deterministic for a given seed/profile, produces replayable artifacts,
supports cancellation, and never deadlocks the runtime. Multi-candidate
extraction (`egraph.extractMany`) keeps several equivalent forms alive.

## Kill switches (Rule 60)

Every superoptimizer is independently disableable via the context
kill-switch map; disabling is recorded in telemetry for bisection.
