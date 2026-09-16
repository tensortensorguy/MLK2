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
   dim is spent so each instance replays exactly once. Pivot ORDERS are
   now searched (decision 7).
6. **Execution marks in the IR** (Loop `parallel` / `vectorHint`): the
   scheduler's zero-distance proof travels with the emitted forest,
   hashed/serialized per Rule 24. The walker threads `parallel` loops
   over disjoint index chunks — sound because zero-distance rows make
   slabs location-disjoint — and both are advisory to backends.
7. **Order search with exact scores** (supersedes the ascending-pivot
   enumeration of decision 5): every pivot permutation is a candidate
   transformation, synthesized and scored exactly (parallel rows,
   innermost unit-stride SIMD fit from the affine access maps, total
   carried distance, lexicographic determinism). Exhaustive below 4
   varying dims, greedy prefix extension deeper, both under explicit LP
   budgets (Rule 10) with identity as the always-feasible fallback. A
   whole-schedule shape contract rejects orders whose statements the
   codegen cannot realize.
8. **CLAST-lite guarded re-entry + integer-exact marking** (supersedes
   the suffix shape contract of decision 7): a row-constant statement
   may VARY again at later rows — the emitter re-enters it under an
   affine-equality `Guard` node at its folded value inside the loop and
   its remaining rows continue in the fused deeper loops. This unlocks
   the vectorized fused GEMM (`[i, k, j]`: i parallel/threaded, k
   carries the reduction, j innermost SIMD — the old contract forced
   `[i, j, k]` with no SIMD-able innermost row). Soundness rests on
   three exact conditions, all checked before emission: the row's
   constant slot is zero for every pivot-varying statement (the loop
   position realizes the schedule value), a REALIZABILITY GATE
   integer-exactly proves no live dependence pair runs backward in
   PIVOT-COORDINATE order (lexmin-witnessed; the same
   `integerLeFormFeasible` primitive upgrades parallel marking from the
   rational hull to the integer points — parity-tight slices keep their
   parallel rows), and row-constant statements must have the pivot
   PINNED for them (their instances occupy exactly one coordinate).
   Legacy speculative guards (Rule 5) are untouched; the walker
   evaluates the affine form on the thread-local var stack and admits
   Guard to the thread-safe emission alphabet.
9. **Piecewise-split re-entry** (supersedes the runtime-Guard form of
   decision 8, which remains the copy-budget fallback): a re-entering
   row-constant statement is realized STRUCTURALLY — the loop range is
   cut at every re-entry value and the statement joins only the
   singleton segment `[v, v]`, where the guard condition holds for the
   whole segment, so its payload emits unguarded and every hot segment
   loses the per-iteration branch. Tiled levels cut the tile range at
   `floor(v/t)`; singleton tiles emit constant-bound point segments.
   Segment enumeration in ascending pivot order plus the unchanged
   deeper recursion realize exactly the guarded form's instance order —
   bit-exact by construction (verified three-way: walker, C++ artifact,
   assembly artifact). Emission-mark restoration between replayed
   bodies is per-statement so single-segment payloads keep their marks;
   the piecewise split exposed a latent C++-emitter scoping bug
   (sibling pairs re-declared the same temp names once the guard's
   if-block scope disappeared) — compute pairs now emit in explicit
   per-pair block scopes mirroring the walker's execPair temp lifetime.

## Consequences

- Zero dependency risk; the engine's limitations are explicit and
  guarded (bails keep the baseline kernel — Rules 62/102/115).
- Parametric scheduling (symbolic dims) is not supported in v1; kernels
  with dynamic bounds stay on the baseline path until workload
  specialization feeds constant bounds.
- The order search evaluates up to 24 full schedules at compile time for
  depth-4 SCoPs (bounded by `kPolyMaxSchedulerLps`); measured adoption
  stays with the autotuner (Rule 32).
- `ctx.accuracy` null guards were added to six pre-existing passes that
  the GEMM demo exposed (Tier2 with no caller-provided contract
  segfaulted); the guards preserve the previous semantics when a
  contract IS provided.
