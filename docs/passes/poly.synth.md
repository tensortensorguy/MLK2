# poly.synth

**Kind**: Lowering (Tier 2/3, kill switch: `poly.synth`)

synthesize multi-dim affine nests from baseline kernels (GEMM and Call(ReduceSum) row-reduction (init + accumulate nests; reduction dim carried, outer dim parallel) Call -> init+accumulate nest; 1-D elementwise over rank>=2 outputs -> R-dim broadcast-aware nest)

## Contract (Rule 142)

- **Required**: kernel.built
- **Produced**: poly.kernel.synth
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
