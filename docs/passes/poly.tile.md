# poly.tile

**Kind**: Transform (Tier 2/3, kill switch: `poly.tile`)

maximal tileable band prefix detection with knob-driven tile sizes (poly_tile_size); band legality proven on PIVOT-COORDINATE dependence distances (loop-iteration space), so skew-heavy schedules end the band correctly

## Contract (Rule 142)

- **Required**: poly.schedule
- **Produced**: poly.tile
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
