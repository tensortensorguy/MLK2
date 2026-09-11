# poly.schedule

**Kind**: Transform (Tier 2/3, kill switch: `poly.schedule`)

identity-prefix schedule + Feautrier-style separator rows (exact rational LP with Bland's rule); parallel/vector markings

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
