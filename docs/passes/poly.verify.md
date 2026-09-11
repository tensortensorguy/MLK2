# poly.verify

**Kind**: Verify (Tier 2/3, kill switch: `poly.verify`)

re-prove schedule legality + structural sanity; restore the baseline kernel on failure (Rules 62/102)

## Contract (Rule 142)

- **Required**: poly.codegen
- **Produced**: poly.kernel.verified
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
