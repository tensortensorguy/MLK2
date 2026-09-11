# poly.schedule

**Kind**: Transform (Tier 2/3, kill switch: `poly.schedule`)

Order-search + LP-selected affine schedule: every pivot order (permutation of varying dims) is synthesized and scored exactly (parallel rows, innermost unit-stride SIMD fit, carried distance, lexicographic determinism); per order, rows are LP-selected (exact rational simplex, Bland's rule) — parallel-first fusion rows (all live dependence distances identically zero), distance-minimal sequential rows (epigraph objective + progress), identity fallback; totality over varying dims; whole-schedule codegen shape contract (row-constant statements must have the pivot pinned for them — const-then-varying transitions are REALIZED by guarded re-entry, not rejected); const-slot normalization (pivot-varying statements keep a zero constant slot); integer-exact realizability gate (no live pair runs backward in pivot-coordinate order, lexmin-witnessed); integer-exact parallel marking (zero distance on the slice's integer points, not the rational hull); parallel/vector markings

## Contract (Rule 142)

- **Required**: poly.scop, poly.deps
- **Produced**: poly.schedule
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
