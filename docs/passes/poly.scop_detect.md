# poly.scop_detect

**Kind**: Analysis (Tier 2/3, kill switch: `poly.scop_detect`)

extract the SCoP (statements, full-rank domains, flat access maps) from the affine kernel forest; eligibility gates with actionable rejections

## Contract (Rule 142)

- **Required**: kernel.built / poly.kernel.synth
- **Produced**: poly.scop
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
