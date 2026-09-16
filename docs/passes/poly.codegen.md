# poly.codegen

**Kind**: Lowering (Tier 2/3, kill switch: `poly.codegen`)

regenerate the kernel loop forest (CLAST-lite): fused loops, hoisted pinned statements, PIECEWISE-SPLIT re-entry (the loop range is cut at every re-entry value; a row-constant statement joins only the singleton segment [v, v] where its guard condition holds structurally — payload unguarded, hot segments branch-free; tiled levels cut the tile range at floor(v/t) with constant-bound point segments inside the singleton tile; the runtime-Guard form remains the copy-budget fallback), tile/point pairs with stack-absolute affine bounds (tile parts are sibling instance ranges — the partial part replays the same statements), re-indexed ElemIdx payloads

## Contract (Rule 142)

- **Required**: poly.tile
- **Produced**: poly.kernel.transformed
- **Invalidated**: downstream poly.* results
- **Budgets**: statement/depth/access/row bounds from `core/constants.h` (Rule 10)

## Failure behavior

Any infeasibility, budget exhaustion, or unsupported shape records an
actionable diagnostic (Rule 67) and leaves the baseline kernel in place
(Rules 62/102/115). See docs/polyhedral_spec.md for the full algorithmic
contract and soundness arguments.
