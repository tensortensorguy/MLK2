# MLK+ Polyhedral Optimizer Specification

**Component**: `mlk_poly` (`polyhedral/`) + the `poly.*` pass family
**Status**: implemented, 37 unit tests + differential runtime validation

The polyhedral optimizer brings loop-nest transformation to MLK+ kernel
compilation: SCoP extraction, exact data-dependence analysis, affine
scheduling, tiling, and Kernel IR regeneration — all over a hermetic
Presburger engine with no external dependencies (Rule 160).

## Pipeline placement

```
lower.to_kernel_ir          (baseline 1-D loops + GEMM Call)
    |
poly.synth                  multi-dim affine nests for the synth classes
    |
poly.scop_detect            SCoP extraction (statements, domains, accesses)
    |
poly.dependence             exact RAW / WAR / WAW relations
    |
poly.schedule               LP-selected rows: parallel-first fusion search,
                            distance-minimal sequential rows, identity
                            fallback (see §scheduling)
    |
poly.tile                   tileable band detection + tile sizes
    |
poly.codegen                schedule -> new Kernel IR loop forest
    |
poly.verify                 legality re-proof; baseline restore on failure
    |
backend.emit_binary
```

Tier 1 never runs poly passes (baseline kernel compiler). Tier 2/3 run the
full chain; every pass has a kill switch (Rule 60) and every failure keeps
the baseline kernel (Rules 62/102/115).

## The engine (`polyhedral/include/mlk/poly/`)

| Module | Contents |
|---|---|
| `rational.h` | Exact rationals (int64/int64, normalized; checked int64 ops in `checked.h`; magnitude limit 2^30 keeps cross-products inside int64 — `-Wpedantic` forbids `__int128`) |
| `affine_expr.h` | Integer affine expressions over `{dims, symbols}` spaces |
| `int_set.h` | Disjunctive Presburger sets: intersection/union/complement/subtract, Fourier–Motzkin elimination with tri-state `Feasibility` (Empty / NonEmpty / Unknown — Rule 22 discipline), exact lexMin/lexMax via lexicographic branch-and-bound with witness verification |
| `affine_map.h` | Affine maps: composition, image/preimage via product-space projection |
| `scop.h` | SCoP model (statements, full-rank domains, flat access maps) + extraction from `KernelModule` |
| `dependence.h` | Dependence relations over the product space (src dims \| dst dims) |
| `lp.h` | Exact rational simplex (Bland's rule — deterministic and cycling-free, Rule 53; phase-1 artificials with post-phase-1 basis cleanup) |
| `pluto.h` | Affine scheduling + `verifyScheduleLegality` |
| `tile.h` | Tileable band detection |
| `codegen.h` | Schedule → Kernel IR emission |
| `workspace.h` | Per-compilation `PolyWorkspace` threaded through `PassContext` (Rule 144: explicit state, no hidden globals) |

### Soundness contract

- Fourier–Motzkin gives **rational** emptiness; rational emptiness implies
  integer emptiness (safe direction). Rational non-emptiness is refined by
  witness-verified integer points wherever a point is required (lexmin).
- `Feasibility::Unknown` (budget exhaustion) is treated as **NonEmpty** by
  every legality caller: optimization may be lost, correctness never.
- Equality substitution consumes the equality rationally; integer gaps are
  covered by the same conservative direction.
- Dependence relations are purified with the original program order
  (intra-statement: sink >lex source with distinct instances;
  cross-statement: sink >=lex source with statement-order tie-break).
- Accumulate stores carry an implicit read access — the accumulator chain
  is the reduction-order legality gate (Rules 33/90: reassociation only
  with a contract).

## SCoP model

- One kernel = one SCoP: the whole node forest must be affine
  (Loop/Compute/Store only); mixed kernels stay on the baseline path.
- Statements are Compute+Store pairs (the kernel statement convention,
  docs/kernel_abi.md).
- Domains are full-rank boxes over the scoping depth; statements at
  shallower depth pin deeper dims to 0 (init-then-accumulate semantics).
- Accesses are flat row-major affine index maps (dense buffers); legacy
  ElemA/ElemB/Store forms synthesize innermost-loop coefficients.
- Eligibility: unit steps, constant bounds within the magnitude limit,
  statement/access/depth budgets from `core/constants.h` (Rule 10).

## Scheduling

Two layers: an ORDER SEARCH over pivot permutations, and per-order row
synthesis by LP — nothing is copied from the original loop order.

**Order search.** Every pivot ORDER (a permutation of the varying dims) is
a candidate transformation. Depth <= `kPolyMaxOrderEnumerateDepth` (4)
enumerates ALL orders; deeper SCoPs extend a prefix greedily one pivot at
a time. Each candidate order is synthesized (below) and scored exactly:

1. parallel rows (more is better),
2. innermost unit-stride fit — the LAST row parallel AND every active
   statement's accesses stride 0/1 along its pivot, computed from the
   affine access maps (SIMD-able innermost loop),
3. total carried distance — the exact rational sum of per-dependence
   minimum distances at their resolving rows,
4. lexicographic pivot order (determinism; identity wins ties).

The first best order wins (`pluto_interchange_by_stride_fit`: a kernel
whose accesses are unit-stride along dim 0 schedules dim 0 INNERMOST — a
loop interchange the ascending-first scheduler never considered). Budgets
(Rule 10): `kPolyMaxSchedulerLps` globally, `kPolyMaxOrderLps` per order;
a tripped budget rejects the candidate order, never the pass. A candidate
whose synthesis errors is skipped identically; the identity order is
always synthesized, so the search is total.

**Per-order row synthesis.** Each live dependence d contributes its SLICE
(relation refined by all earlier rows' `dist == 0` equalities); Farkas'
lemma turns "dist_r(v) >= 0 for all v in slice" into linear constraints
over the coefficients. Three deterministic stages, first feasible wins:

1. **Parallel search** — a feasibility LP with BOTH `dist >= 0` and
   `-dist >= 0` blocks per live dependence: a row under which every live
   dependence distance is identically zero. Such rows FUSE statement
   nests and carry nothing (producer/consumer elementwise chains collapse
   into one loop; see `pluto_fusion_emergence`).
2. **Sequential search** — minimize `sum_d M_d` (distance epigraphs)
   subject to validity and `sum_d M_d >= 1` (progress): distance-minimal
   carrying rows, the fusion/skewing driver. Stencil time rows fall out
   here, with the space row parallel (`pluto_sor_time_space_parallel`).
3. **Identity fallback** — `theta = e_pivot`. Original-order purification
   orients every dependence forward for ANY pivot sequence (dims with
   zero distance refine; the first nonzero-distance pivot resolves
   positively), so this row is always valid; it guarantees termination
   even when both LP searches reject.

Shape contract (checked per realized row, codegen compatibility): every
statement is either a foldable constant (nonzero coefficients only on
pinned dims) or varies on the row's PIVOT dim with a positive coefficient.
Spent-dim (outer-loop) coefficients stay free in the LP — skew terms —
and never reach loop bounds: with the loop variable kept as the pivot dim
and box bounds, the nest enumerates instances in exactly the schedule's
lexicographic order (equal prefixes force equal spent coords by
induction — valid for arbitrary pivot SEQUENCES, which is what makes the
order search codegen-safe). Rows are emitted until every varying dim is
spent (TOTALITY: each instance maps to a distinct schedule vector, so
codegen replays each payload exactly once). Dependences still live at
that point are schedule-tied and resolve by statement order
(`origOrder`); `verifyScheduleLegality` proves exactly that tie-break
rule. Candidate LPs that error (budget/overflow, e.g. an LCM-realized row
past the coefficient bound) reject the CANDIDATE (Rule 10), never the
order or the schedule.

Whole-schedule shape contract (order-search rejection): every statement
is either a foldable constant (all nonzero coefficients on pinned dims —
its instances occupy exactly one pivot coordinate) or varies on the
row's pivot with a positive coefficient. Row-constant statements MAY
vary again at later rows: the emitter re-enters them under an
affine-equality `Guard` at their folded value inside the loop (CLAST-
style guarded re-entry — see §codegen). Two conditions keep the
realized nest order equal to the schedule order:

1. **Const-slot normalization** — every pivot-varying statement's
   constant slot is zero in the LP, so the emitted loop position
   realizes the schedule value exactly; over box domains containing the
   origin this also forces all row coefficients non-negative (skew
   terms stay forward everywhere).
2. **Realizability gate** — for every live dependence, an integer-exact
   check (lexmin witness over `{slice ∧ v_p(dst) - v_p(src) <= -1}`)
   proves no instance pair runs BACKWARD in pivot-coordinate order; a
   witness or a budget trip rejects the row (the stage falls through).
   The identity fallback is gate-free — its unit coefficients and zero
   slots make the distance form equal the coordinate form.

Parallel marking: a row is parallel iff every live dependence distance
is IDENTICALLY ZERO on the slice's INTEGER points (the instances that
execute). The rational hull is a pre-filter only: `dist` is integer-
affine over integer points, so "nonzero somewhere" is the union
`dist >= 1 ∨ dist <= -1`, decided by `integerLeFormFeasible` (rational
pre-filter + witness-verified lexmin; a budget trip marks sequential —
Rule 22). A dependence that is integer-exactly zero but fractionally
nonzero on the hull (transposed flat-index collisions, parity-tight
relations) therefore KEEPS its parallel row — the rational-hull
conservatism that cost one parallel row in earlier rounds is closed.
The innermost parallel row is the vectorizable one.
`verifyScheduleLegality` re-proves the final schedule lexicographically;
`poly.verify` runs it before the backend.

## Codegen

Emission walks schedule levels over the statement group (CLAST-lite):

- **Separator levels** (all-constant rows) emit statements in
  (time, origOrder) order.
- **Loop levels** emit one shared loop over the row's PIVOT dim
  (fusion; skew coefficients on spent dims are order-only and invisible
  in bounds). Statements whose row is pin-foldable (all nonzero
  coefficients on dims pinned for that statement — the init statement's
  `k == 0`) hoist as constant items: values below the loop range emit
  before, above after, at the range start only when the remaining rows
  are constants dominated by the varying statements' minimums
  (`canEmitBefore`, conservative box check). Pin-folding also applies to
  payload re-indexing: coefficients on pinned dims fold into the ElemIdx
  / store offset.
- **Guarded re-entry** — a row-constant statement whose folded value
  falls INSIDE the loop range (or whose hoisting is unsafe) re-enters
  the loop: a `Guard` node with the affine-equality form
  `1*var - value == 0` wraps its payload, and its remaining schedule
  rows continue in the fused deeper loops (`for k { for i { for j {
  if (k == 0) init; acc } } }` — the vectorized fused GEMM the suffix
  shape contract used to reject). Guards accumulate per statement and
  materialize as nested Guard nodes at every payload emission (tile
  part duplication replays them). Soundness: the pivot is PINNED for
  every row-constant statement (its instances occupy exactly one
  coordinate), and the realizability gate (§scheduling) keeps every
  live pair's pivot-coordinate order forward, so the guard fires at the
  statement's schedule position and the exhaust-level emission replays
  statement pairs in origOrder (the tie-break the schedule proves).
- **Tiled band levels** emit a TILE loop (constant bounds
  `floor(lo/t)`, `floor(hi/t)`) plus a POINT loop with affine bounds
  `t*ti … t*ti + t - 1`; a trailing partial tile (when `(hi+1) % t != 0`)
  carries a constant clipped end. Tile parts are SIBLING instance
  ranges: the emission marks and guard state restore between parts so
  the partial part replays the same statements for its own point range
  (bounded by `kPolyMaxCodegenCopies`). The point loop's affine bounds
  are STACK-ABSOLUTE coefficient vectors with the tile coefficient at
  the tile-index slot.
- Payloads replay the statement Compute chain and Store (accumulate
  preserved) with ElemIdx operands re-indexed from scop-dim to
  stack-position coefficients; store targets map the same way.

Kernel IR extensions (all backward compatible, hashed and serialized per
Rule 24): `KernelOperand::Kind::ElemIdx` (buffer + per-stack-position
coefficients + offset), affine loop bounds (`beginCoeffs/endCoeffs` over
the enclosing var stack, STACK-ABSOLUTE — coeffs[p] multiplies vars[p]),
buffer-dim loop bounds (`endBuf/endDim`), `Store::accumulate`, the Loop
execution marks `parallel`/`vectorHint` (set by codegen from the
scheduler's row marks; the walker threads `parallel` loops, `vectorHint`
is advisory — see Executor), and the Guard affine-equality form
(`guardCoeffs/guardOffset`: children run only where the form over the
enclosing var stack evaluates to zero; empty coeffs keep the legacy
speculative-guard semantics, Rule 5).

Bail conditions (conservative, baseline preserved): statements neither
pin-foldable-with-pinned-pivot nor pivot-varying (scheduler shape
contract violation), unaligned tile lower bounds, copy-budget
exhaustion, unbounded levels, accesses referencing unpinned unspent
dims, guard dims unbound at emission.

## Executor

`executeKernelOnBuffers` routes any module containing multi-dim features
through a recursive tree walker (loop-var stack, affine bounds, ElemIdx
resolution, accumulate stores); the legacy 1-D fast path keeps its
chunked threading. Multi-dim loops marked `parallel` (zero-distance rows
proven by the scheduler) run disjoint index chunks on threads — same
executor-owned thread decision as the 1-D path (`kKernelExecMaxThreads`,
the `threads` schedule param, the `kParallelChunkElements` amortization
threshold), withheld for subtrees containing `Call` nodes (symbol
interning is not hot-path-safe under threading). Zero-distance rows mean
slabs read/write disjoint locations at that level, so any interleaving
produces the sequential result bit-exact (Rule 43) — no locks
(Rule 137). Each worker records its terminal state in a slot; the parent
joins and returns the first failure deterministically.
Affine-equality `Guard` nodes evaluate their form on the thread-local
var stack (overflow-checked builtins) and skip their children when it is
nonzero; Guard belongs to the thread-safe emission alphabet (the
predicate reads only the var stack).
`Call(ReduceSum)` executes as a row-wise trailing-dim sum in ascending-k
order — exactly the order the transformed accumulate chain preserves.

## Verification story

- Unit: 46 tests in `tests/unit/unit_poly.cpp` covering the engine
  (rationals, sets, FM, lexmin, maps), extraction, dependences, LP,
  scheduling legality + determinism, FUSION EMERGENCE (producer/consumer
  single parallel row + bit-exact fused execution), wavefront dependences,
  SOR time/space parallel marking, tiling, codegen structure, the full
  pass chain, differential execution, LOOP INTERCHANGE emergence
  (stride-fit order search + bit-exact permuted execution), THREADED
  parallel-loop determinism (parallel/vector marks + bit-exact under
  chunked threads), the ReduceSum synthesis class, INTEGER-EXACT
  parallel marking (a parity-tight slice keeps its parallel row where
  the rational hull marks it sequential), GUARD predicate execution
  (the affine-equality form filters children), and the TILED GUARDED
  GEMM (full tiles + partial tails across two levels, bit-exact vs the
  untiled schedule — the tile-part replay and stack-absolute point
  bounds).
- Differential (Rules 43/85/90): the transformed GEMM executes over dense
  buffers and matches a straightforward reference (bit-exact for the
  no-reassociation class; accumulation order per output cell preserved).
- `poly.verify` re-proves legality inside every Tier2/3 compilation and
  restores the baseline on failure.

## Roadmap

- Full Pluto ILP locality objective over all rows simultaneously (the
  order search composes exact per-row scores; an ILP with memory-reuse
  terms could weight fusion across nests beyond the carried-distance
  proxy).
- Parametric SCoPs (symbolic dims with runtime guards) — currently
  requires constant bounds after workload specialization.
- Softmax/multi-input elementwise synthesis classes; >2-input elementwise
  (needs the baseline buffer-materialization limit lifted).
- Multiple SCoP regions per kernel.
- Piecewise (split) codegen for guarded bands: the guard predicate is
  correct but runs per iteration; splitting the separator iteration out
  of the loop range removes the branch from hot loops.
- Per-statement loop-bound generalization: fused statements currently
  share the intersection of their pivot bounds (the synth classes agree;
  disagreeing domains bail via the shape contract).
- `poly_tile_size` autotuner integration over the order dimension (the
  search is compile-time exact; tile sizes remain the measured knob).
