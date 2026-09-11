# poly.dependence

**Kind**: Analysis (Tier 2/3, kill switch: `poly.dependence`)

exact RAW/WAR/WAW dependence relations over the product space with original-order purification and accumulate-chain modeling

## Contract (Rule 142)

- **Required**: poly.scop
- **Produced**: poly.deps
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
