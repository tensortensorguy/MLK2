# ADR-0005: Hermetic polyhedral engine (no ISL)

**Status**: Accepted
**Context**: The polyhedral pipeline needs Presburger set operations
(elimination, projection, lexmin), exact rational LP, and dependence
analysis. The reference implementations (ISL, PipLib) are external C
dependencies.

## Decision

Build the engine in-house (`mlk_poly`, `polyhedral/`) over checked int64
arithmetic:

1. **Hermetic build** (Rule 160, ADR-0001): no FetchContent, no system
   ISL. The engine is ~4k lines of auditable C++ with the same Result<T>
   discipline as the rest of MLK+ (Rule 6) and no RTTI (Rule 8).
2. **Bounded magnitudes** instead of big integers: inputs are limited to
   2^30 so every documented product stays inside int64 (`-Wpedantic`
   forbids `__int128`; overflow-checked builtins in `checked.h`). Loop
   bounds beyond the limit are rejected at SCoP extraction (Rule 10).
3. **Tri-state feasibility** (Rule 22): FM elimination answers
   Empty / NonEmpty / Unknown; Unknown is treated as NonEmpty by legality
   callers (conservative — loses optimization, never correctness).
4. **Determinism** (Rule 53): Bland's-rule simplex, fixed candidate
   ordering, integer scaling by LCM + gcd normalization.
5. **LP-selected scheduling** (supersedes the original identity-prefix
   shortcut): every row is chosen by an exact LP — parallel-first
   feasibility search (all live dependence distances identically zero;
   fuses statement nests), then distance-minimal sequential rows
   (Farkas-encoded validity + epigraph objective + progress), with the
   identity row as a guaranteed-valid fallback (original-order
   purification orients every dependence forward). A per-row shape
   contract keeps codegen total: loop variables stay pivot dims with box
   bounds, skew coefficients ride the schedule only, and every varying
   dim is spent so each instance replays exactly once. Pivot orders are
   ascending (identity always feasible ⇒ always wins); free-coefficient
   rows + CLAST-style codegen remain the documented roadmap to full Pluto
   ILP; the autotuner gates adoption by measurement (Rule 32).

## Consequences

- Zero dependency risk; the engine's limitations are explicit and
  guarded (bails keep the baseline kernel — Rules 62/102/115).
- Parametric scheduling (symbolic dims) is not supported in v1; kernels
  with dynamic bounds stay on the baseline path until workload
  specialization feeds constant bounds.
- `ctx.accuracy` null guards were added to six pre-existing passes that
  the GEMM demo exposed (Tier2 with no caller-provided contract
  segfaulted); the guards preserve the previous semantics when a
  contract IS provided.
