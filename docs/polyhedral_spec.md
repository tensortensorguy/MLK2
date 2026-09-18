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
  CHAIN-FINAL READS are the documented exception: a cross-statement RAW
  whose source write is an ACCUMULATE store (AccumMode Add/Max) feeds a
  reduction chain, and a plain reader consumes the chain's FINAL value —
  it may depend on every chain writer, so the lexicographic narrowing
  (which models per-iteration flow for init/accumulate pairs) is
  UNSOUND there and the FULL tie relation is kept (a conservative
  superset of the true dataflow). Any row varying a chain dim then
  carries distances of both signs and is rejected — reading a completed
  reduction chain requires band separation (softmax synthesis exposed
  this: the old narrowing let the scheduler fuse the exp band into the
  rowmax band, emitting kernels that read the RUNNING max; the
  differential caught it and the rule is now structural).
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

## Synthesis

`poly.synth` upgrades baseline tensor work into the multi-dim nests the
SCoP extractor lifts (the "MathGraph -> Kernel IR -> SCoP" chained path).
Recognized classes: Call(MatMul) and Call(ReduceSum) (init + accumulate
nests), 1-D elementwise over rank >= 2 outputs (broadcast-aware R-dim
nest), and — new — **Call(Softmax)**, the stable-form statement chain:

    Y[M,K] = softmax_rows(A[M,K])
    S0: RM[m] = -inf                    (depth 1)
    S1: RM[m] max= x[m,k]               (AccumMode::Max; k band)
    S2: E[m,k] = exp(x[m,k] - RM[m])    (k band)
    S3: S[m] = 0                        (depth 1)
    S4: S[m] += E[m,k]                  (AccumMode::Add; k band)
    S5: Y[m,k] = E[m,k] / S[m]          (k band)

with three MATERIALIZED TEMP BUFFERS (`KernelBuffer::isTemp`: rowmax [M],
exp [M,K], sum [M]) — the first synthesis class that needs the baseline
buffer-materialization limit lifted. Key exactness contracts:

- The reference kernel (`execSoftmaxCall`) and the synthesized chain
  replay the SAME operations in the SAME order per row (ascending-k max,
  ascending-k exp, ascending-k sum, divide) — bit-exact by construction
  (Rules 43/90).
- The value passthrough is `Sub(x, 0)`, which is the EXACT identity for
  every IEEE-754 double including -0.0 (Add would flip -0.0 to +0.0);
  with `Add(-inf, 0)` as the -inf carrier, no synthesized value differs
  from the reference value.
- `AccumMode::Max` stores are the order-insensitive row-max primitive:
  `out = (value > out) ? value : out`. NaN never replaces the running
  value ((NaN > cur) is false); ±0 ties keep the current slot.
- The four sibling k-bands execute sequentially as emitted (the walker
  runs children in order). FUSING the bands is the scheduler's
  band-shift roadmap item: the chain-final-read dependences above
  (S1->S2, S4->S5) carry distances of both signs on k, so no schedule
  row may vary k while both bands are live — the current scheduler
  reports "no schedule found" and the synthesized (correct) nests are
  kept (Rules 62/102/115). The optimizer loses fusion, never
  correctness; `poly.schedule` reports the fallback as an Info
  diagnostic instead of failing the pipeline.

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
codegen replays each payload at least once — the piecewise split
(§codegen) duplicates a payload only across DISJOINT loop segments,
one per containing range). Dependences still live at
that point are schedule-tied and resolve by statement order
(`origOrder`); `verifyScheduleLegality` proves exactly that tie-break
rule. Candidate LPs that error (budget/overflow, e.g. an LCM-realized row
past the coefficient bound) reject the CANDIDATE (Rule 10), never the
order or the schedule.

Whole-schedule shape contract (order-search rejection): every statement
is either a foldable constant (all nonzero coefficients on pinned dims —
its instances occupy exactly one pivot coordinate) or varies on the
row's pivot with a positive coefficient. Row-constant statements MAY
vary again at later rows: the emitter re-enters them at their folded
value via the PIECEWISE SPLIT (see §codegen). Two conditions keep the
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
- **Piecewise-split re-entry (CLAST range splitting)** — a row-constant
  statement whose folded value falls INSIDE the loop range (or whose
  hoisting is unsafe) re-enters at its coordinate: the loop range is
  CUT at every re-entry value and the statement joins only the
  singleton segment `[v, v]`, where the guard condition
  `1*var - value == 0` holds for the WHOLE segment — its payload emits
  UNGUARDED there, and its remaining schedule rows continue in the
  fused deeper loops of that segment (`for i { for k in [0,1) { for j {
  init; acc } } for k in [1,K) { for j { acc } } }` — the vectorized
  fused GEMM, branch-free in the hot k range). Tiled levels cut the
  TILE range at `floor(v/t)`; the singleton tile emits constant-bound
  point segments. Segment enumeration in ascending pivot order plus the
  unchanged deeper recursion realize EXACTLY the guarded form's
  instance order, so the split is bit-exact by construction and the
  per-iteration branch disappears from every hot segment. Emission
  marks restore between replayed bodies PER-STATEMENT (only the
  statements a body replays) so single-segment payloads keep their
  emitted marks. The runtime `Guard` form remains as the FALLBACK when
  the split's replay budget would exceed `kPolyMaxCodegenCopies`
  (correct, retains the branch). Soundness: the pivot is PINNED for
  every row-constant statement (its instances occupy exactly one
  coordinate), and the realizability gate (§scheduling) keeps every
  live pair's pivot-coordinate order forward.
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
dims, guard dims unbound at emission (guards arise only on the budget
fallback path).

## Executor

`executeKernelOnBuffers` routes any module containing multi-dim features
through a recursive tree walker (loop-var stack, affine bounds, ElemIdx
resolution, accumulate stores); the legacy 1-D fast path keeps its
chunked threading. `isTemp` buffers are MATERIALIZED by the executor:
one zero-initialized allocation per temp (bufferId-indexed in the
bindings' `temps` array, element count bounded by
`kKernelTempElementsLimit`), alive for the duration of the call — every
element must be written before it is read (the synthesized init
statements guarantee that). Multi-dim loops marked `parallel` (zero-distance rows
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
`Call(Softmax)` executes as the four-pass stable form (rowwise: max,
exp into the output row, sum, divide — all ascending-k), which is the
reference order the synthesized chain replays.

## Verification story

- Unit: 59 tests in `tests/unit/unit_poly.cpp` covering the engine
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
  (the affine-equality form filters children), the TILED GUARDED
  GEMM (full tiles + partial tails across two levels, bit-exact vs the
  untiled schedule — the tile-part replay and stack-absolute point
  bounds), and the SOFTMAX synthesis class (pipeline + K==1/shape sweep,
  Max-accumulate store semantics incl. the NaN/±0 select rules, and the
  native THREE-WAY temp ABI — walker / C++ artifact / assembly artifact
  bit-exact on adversarial NaN and ±0 rows).
- Differential (Rules 43/85/90): the transformed GEMM executes over dense
  buffers and matches a straightforward reference (bit-exact for the
  no-reassociation class; accumulation order per output cell preserved).
- `poly.verify` re-proves legality inside every Tier2/3 compilation and
  restores the baseline on failure.

## Backend (native artifacts)

The polyhedral layer closes the compiler loop with two standalone AOT
artifact forms, produced by `mlk_backend_cpu` and driven out-of-process
(ADR-0003 as amended by ADR-0006 — no in-process machine codegen exists
anywhere; there are no emitter-owned W^X pages, Rule 118 patching is
vacuous, and the artifact is mapped file-backed by the system loader
like any plugin):

- **C++ source** (`emitCppSource`): portable scalar C++ mirroring the
  buffer-executor semantics verbatim, including OpenMP pragmas for the
  scheduler-proven parallel/vector marks under `#ifdef _OPENMP`
  (inert without the flag — Rule 148: gated and recorded).
- **x86-64 assembly** (`emitAsmSource`): AT&T-syntax, System V AMD64,
  position-independent (RIP-relative rodata, PLT calls), SSE2 scalar
  IEEE-754 doubles. The emitted code performs the SAME operations in
  the SAME order as the walker — `addsd/subsd/mulsd/divsd/sqrtsd` plus
  libm PLT calls for transcendentals, no FMA contraction anywhere — so
  the differential test against the executor is bit-exact by
  construction (Rules 33/43/90). Parallel/vector marks are RECORDED in
  the artifact header (Rule 148); the artifact executes sequentially,
  which stays deterministic because parallel rows write disjoint slabs.

Both emitters share one contract: the walker's routing decision
(`isMultiDimModule`) selects the ABI form, and the fixed scalars tail
(`mlk_scalars, mlk_n_scalars`) is ALWAYS part of the signature so the
driver dispatch and the artifact can never disagree about arity.
Loop bounds (constant / affine over the enclosing var stack /
buffer-dim bounds), ElemIdx affine addressing, accumulate stores,
affine-equality guards, temp chains (cap = the executor's 64 slots),
and the legacy 1-D form are all supported; multi-dim artifacts check
store-address sign at runtime and report a nonzero ABI code where the
walker raises InvalidGraph (the driver maps it back).

**Native temp ABI (multi-dim form).** Temp scratch is part of the
BINDABLE TABLE: every `isTemp` buffer contributes a `(ptr, dims)` pair
in table order, exactly like an output. The DRIVER materializes the
scratch — one zero-initialized allocation per temp with the walker's
element-count model and the SAME `kKernelTempElementsLimit` guard
(hoisted to constants.h so walker and driver read one constant) — and
the storage outlives the call. Max-accumulate stores emit the walker's
EXACT select `(value > cur) ? value : cur`: the C++ artifact emits the
ternary verbatim; the assembly artifact emits
`comisd` + `jbe`-guarded store (unordered/NaN and <= all set CF|ZF, so
`jbe` keeps the running slot — `vmaxsd` would break the NaN rule).
The softmax class therefore runs THREE-WAY bit-exact (buffer walker /
C++ artifact / x86-64 assembly artifact), including the adversarial
NaN and +/-0 rows that pin the select semantics
(`softmax_native_three_way_bitexact`,
`max_accumulate_store_native_bitexact`). The legacy 1-D form keeps
rejecting temps/Max structurally (the 1-D executor routes every such
module through the multi-dim walker; a legacy-form temp or Max never
reaches the emitters). The verified
"poly7" Sin family (math_families.h certificate) emits LOCAL helper
routines in the assembly artifact — Cody-Waite quadrant reduction and
degree-13 minimax Horner residuals mirrored OPERATION FOR OPERATION
from the C reference, constants taken from the same header (Rule 77),
sign flips via the register-mediated Neg pattern (xorpd with a memory
operand would demand 16-byte literal-pool alignment the quad pool does
not guarantee), floor/fmod through the PLT. Unsupported nodes fail
emission honestly: Call (lower by poly.synth first, Rule 121),
speculative guards (Rule 5), AllocBuffer/CopyBuffer, non-unit steps,
and legacy dynamic bounds inside multi-dim modules.

Every ARGUMENT lives in a home stack slot spilled by the prologue:
argument registers are caller-saved, and a PLT call inside a compute
(sin/exp/tanh, the poly7 helpers) clobbers %rdi..%r9 — reading an
argument register after such a call silently uses a clobbered pointer
(a REAL bug the poly7 test exposed: every earlier asm artifact was
PLT-free, so the latent clobber never fired). Argument reads are now
call-safe by construction.

The **backend driver** (`backend_driver.h`) runs emit -> write ->
`cc -shared` (C++ artifacts additionally get `-O2 -fPIC
-fno-exceptions -fno-rtti` — the artifact is held to the same
no-exception contract as the host) -> `dlopen` -> invoke, with a
per-stage error result (compiler logs are captured and reported —
Rule 67). The C++/assembly emitters are themselves pure text
generation: deterministic (re-emission is byte-identical, Rule 53),
hashable inputs (Rule 24), and the whole path is verified by the
`asm_backend_*` / `backend_*` tests, which build REAL shared objects
and compare every output element bit-exactly against the buffer
executor (three-way: walker vs C++ artifact vs assembly artifact).
Tools: `mlk-poly demo --backend=asm|cpp` executes the native artifact
inside the demo report; `mlk-poly emit <graph.mlk> [--asm|--cpp]
[--out=<path>] [--workdir=<dir>] [--compile]` produces the standalone
artifact text and optionally the built `.so`.

## Fast-kernel search (OEIA/Refined/FastKernels)

The `mlk_fastkernel` component (`fastkernel/`) makes kernel runtime
speed a first-class, CERTIFIED optimization objective under a hard
compile-time budget, implementing the uploaded OEIA/Refined/FastKernels
axiom set (sections XIV-XVIII). The search is a driver-level
orchestrator: it owns no IR of itself and composes the pipeline (per-
variant Tier2 compiles), the out-of-process artifact driver, and the
buffer executor.

**Certificate requirement (Meta-Axiom 0.1).** No claim without a
certificate; every candidate outcome carries a machine-checkable bundle
(`VariantCertificate`), and the whole report serializes to JSON
(`FastKernelReport::toJson`) so the audit does not require the search
process.

**Policy (Axioms 0.2/1.4/14.17).** The search policy is
`bitexact-f64-cpu-v1`: exact f64 kernel identity. Reassociation, FMA
contraction, approximate reductions — the entire Axiom-14.17 numerical
relaxation menu — are ILLEGAL under this policy, so a candidate that
diverges by one ulp is REJECTED (`IdentityMismatch`), never tolerated.

**Kernel variants (Axioms 14.1/14.3).** The declared variant space is
tile sizes (0 = untiled, via the `poly.tile` kill switch) x execution
paths (buffer executor / native assembly artifact / native C++ artifact)
x walker thread overrides. Repeated declarations deduplicate in
declaration order. Launch configuration is part of the kernel object
(Axiom 14.13): thread overrides live in `scheduleParams["threads"]`.

**Compile-time accounting (Axioms 14.5/15.2).** Per candidate: the
stage-accounted Tier2 compile, artifact emission + out-of-process build
(coalesced into `build_sec` for first use), and load. Cache reuse
reports `M^reuse` = load time alone with `comptimeFromCache` set
(Axiom 15.8).

**Budget modes (Axiom 15.1) and the regression guard (Axiom 15.6).**
`SameComptime` (B_allowed = B_K^0), `BoundedExtra` (B_K^0 + dB_K <=
dB_max; extra comptime admissible only through the justification gate),
`Amortized` (objective U^run + M^kernel/N). B_K^0 = 0 means AUTO: max of
3 declared samples of the default-pipeline compile time (a single
sample's ~0.3% noise would decide admissibility by measurement luck at
the boundary; the max is the conservative estimate, and the provenance
string says so). A candidate with M^kernel > B_allowed is INADMISSIBLE —
recorded as `ComptimeBudgetExceeded`, never a silent overrun (Rule 17.5:
compile-time is a kernel constraint).

**Extra-comptime justification (Axioms 14.9/15.7).** The pure decision
function `evaluateExtraComptimeJustification` certifies one of:
absolute runtime improvement (U_new <= U_base - tau), amortized payoff
(alpha*dM <= N*dR), or Pareto improvement (strict runtime gain, compile
time the only excused dimension). U_base is the best certified
same-comptime candidate; when none exists it falls back to the
specification's own measured runtime. Without a certificate the extra
comptime is refused (`NoJustification`) and the search prefers the
same-comptime kernel (Rule 17.2).

**Identity and runtime certificates (Axioms 14.2/14.4).** The oracle is
the Tier1 reference execution; identity is a bit-exact differential
(max|diff| = 0 element-for-element; NaN never matches). U^run is the
declared benchmark method (median of `benchReps` timed runs after
`warmupReps` warmups; min/max recorded alongside).

**Launch-coverage certificate (Axioms 14.13/14.14).**
`certifyLaunchCoverage` mirrors the buffer executor's `decideThreads` +
chunk decomposition (the same named constants from constants.h) and
proves the worker slabs cover the loop domain exactly and pairwise
disjointly — no dropped, duplicated, or out-of-bounds work. A threads
override above `kKernelExecMaxThreads` is a resource violation
(Axiom 14.16) reported in the certificate.

**Roofline lower bound (Axioms 14.11/14.12).** The essential-work model
is DECLARED: B_min = non-temp buffers moved once (temps assumed
cache-resident), O_min = 2*M*K*N (MatMul), M*K (ReduceSum), 5*M*K
(softmax max/sub/exp/add/div row model), 1 per fused Compute expression,
1 per accumulate store. Unknown buffer extents mark the model UNKNOWN —
the roofline certificate is then omitted (tri-state, Rule 22), never
guessed. Environment peaks BW_h / T_h are measured once per search
(calibrated stream triad / four independent multiply-add chains with
volatile sinks; sub-clock-resolution reps are scaled until measurable —
an uncalibrated probe reads as an absurd peak, which is a broken
certificate). The winner's certified gap U^run/L - 1 is reported, never
hidden.

**Honest claims (Axiom 14.21).** Every candidate evaluated and
certified (or rejected for budget/identity/justification/resource) =>
"fastest certified kernel within the declared searched space (exhaustive
finite search)". Any build/capability failure => "best certified kernel
in searched space". No feasible kernel => "budget failure" (Axiom 15.6).
"Fastest possible" is never claimed.

**Cache validity (Axioms 14.22/15.8).** Artifacts cache under
`<cacheDir>/fk-<fingerprint>/` where the fingerprint binds the kernel
source hash, artifact kind, compiler, policy, budget mode, and the
environment (descriptor AND measured peak values — two environments
sharing a label but differing in peaks are different environments). A
matching entry is dlopened directly (`loadKernelLibrary`); any change
invalidates by fingerprint.

**CLI.** `mlk-poly autotune [--m=--k=--n=] [--tiles=..] [--exec=..]
[--budget-mode=same|extra|amortized] [--comptime-budget-ms=X]
[--extra-budget-ms=X] [--tau-ms=X] [--alpha=A] [--executions=N]
[--reps=R] [--warmup=W] [--cache-dir=D] [--json-out=F]` prints the
certificate table and (optionally) the full JSON report. Tests:
`tests/unit/unit_fastkernel.cpp` (9 cases: coverage certificate, exact
roofline model, unknown-extent tri-state, justification forms, full
same-comptime search, budget-failure report, refused extra comptime,
cache reuse + invalidation).

## Roadmap

- **Band-shift scheduling** (per-statement constant offsets on varying
  rows): the machinery that lets the scheduler express SEPARATE bands
  over the same loop dim — softmax's rowmax/exp/sum/div phases over k
  (the chain-final-read dependences demand it), and generally any
  kernel whose phases must complete per region before the next starts.
  Requires per-statement loop bounds in codegen (the min/max-of-affine
  forms CLAST emits) alongside the piecewise split.
- Full Pluto ILP locality objective over all rows simultaneously (the
  order search composes exact per-row scores; an ILP with memory-reuse
  terms could weight fusion across nests beyond the carried-distance
  proxy).
- Parametric SCoPs (symbolic dims with runtime guards) — currently
  requires constant bounds after workload specialization.
- Multi-input elementwise synthesis classes beyond softmax; >2-input
  elementwise (temp materialization infrastructure now exists, and the
  native temp ABI makes synthesized temps first-class artifact
  citizens).
- Multiple SCoP regions per kernel.
- Per-statement loop-bound generalization: fused statements currently
  share the intersection of their pivot bounds (the synth classes agree;
  disagreeing domains bail via the shape contract).
- Split for tiled levels currently keeps the guard-free point segments
  only inside singleton tiles; a point-level split that removes the
  affine point-loop bound machinery for partial tiles could shrink the
  emitted forest further.

## GPU backend (design contract, implemented round 19)

The pipeline extends to GPU targets with NO structural change: the
polyhedral layer already owns the schedule, the parallel marks are
proven, and the artifact driver already treats "emit text → external
toolchain → load → invoke through the buffer ABI → differential vs
the walker" as the only sanctioned publication model (ADR-0003/0006 —
never in-process machine codegen). The GPU phase reuses every stage
and swaps the emission + invocation tail:

- **Artifact form.** CUDA C++ (`.cu`) first, PTX later if a use case
  demands it — nvcc compiles out-of-process exactly like `cc` today;
  the artifact is a `.so` exposing the SAME `mlk_kernel` ABI, where
  the host-side function manages device memory explicitly
  (cudaMalloc/cudaMemcpy H2D, launch, memcpy D2H) behind the same
  Result-mapped error codes. Multi-dim temp ABI: device-side scratch
  materialized with the walker's element-count model and the same
  `kKernelTempElementsLimit`.
- **Thread mapping from proven marks.** Scheduler rows marked
  parallel map to grid/block dimensions (tile loops → blocks, point
  loops → threads); serial (carried-dependence) dims stay a serial
  loop inside the kernel thread. The single-parallel-level rule from
  the CPU emitter carries over: one launch per outermost parallel
  band, never per point iteration.
- **Bit-exactness boundary (the hard part, declared up front).** A
  kernel thread computes one output cell with its k-chain in the same
  ascending order as the walker — no cross-thread reduction, no
  reassociation, `-fmad=false` (separate mul/add, matching the walker
  and the asm emitter's contract). Transcendentals: `exp`/`erf` are
  libm-ordered on the CPU path; the GPU path must either reproduce
  the same bit patterns (device libm does NOT guarantee it) or the
  differential contract for those ops is explicitly re-declared as
  ULP-bounded with the bound measured and recorded — never silently
  relaxed (Rule 90 honesty; softmax is the first test case).
- **Verification.** The existing three-way differential harness
  (walker vs artifact vs Tier1 baseline) runs unchanged; identity is
  still memcmp-grade where the op set allows, ULP-bounded-with-
  certificate where device libm forces it.
- **Honest initial boundary.** Single GPU, one stream, no unified
  memory, no multi-device partitioning; buffers ≤ device memory with
  the limit checked and reported, not guessed. Everything outside the
  boundary is a structured rejection, never a fallback.

Entry point when this phase starts: `ArtifactKind::Cuda` +
`GpuBackendDriverConfig` (compiler = `nvcc`, arch flag declared per
config), reusing `buildKernelArtifactInDir`, the artifact cache
fingerprint (compiler kind becomes part of it), and `mlk-poly bench
--paths=cuda` for the same Rule 49 protocol.

### GPU backend — implementation status (round 19)

The contract above is IMPLEMENTED as `cuda_emitter.cpp` (emitCudaSource)
+ the GPU driver entry points (buildGpuKernelArtifact / InDir). What
exists, mechanism by mechanism:

- **Artifact shape.** One `__global__` device kernel per forest ROOT
  (mlk_dev_<idx>, capped at kCudaMaxDeviceKernels = 64, honest rejection
  beyond), each taking the FULL bindable table (ptr + dims per buffer,
  temps included) + the scalars pair + an `int* mlk_status` slot. The
  host wrapper `extern "C" int mlk_kernel(...)` keeps the ABI EXACTLY —
  the driver dispatch is form-based (multiDim/legacy), not
  backend-based, so run() needs no dispatch changes. Temp table entries
  stay in the host signature for arity uniformity but are IGNORED by
  the artifact (device scratch is materialized internally from the
  module's static temp dims; the driver passes dummies for gpu_ kernels
  — documented boundary, not a silent drop).
- **Padded-grid collapse from proven marks.** The outermost chain of
  parallel-marked unit-step loops whose interior bodies are exactly one
  loop child is peeled into a flat 1-D grid, one thread per instance
  tuple, decomposed row-major (outermost first — the walker's stack
  order). Point loops with AFFINE bounds over the prefix (tile → point
  pairs) ARE peelable: per-level PADDED trip counts are the exact
  interval max of the level's trip over the prefix box (interval
  arithmetic on affine forms, text-generated so host and device totals
  share one form), and each thread checks its var against its OWN
  begin/end forms, returning on padding holes. Soundness: the parallel
  marks prove every collapsed level's instances slab-disjoint, so any
  enumeration is deterministic; coverage is exact because the padded
  trip is ≥ every prefix tuple's real trip and holes exit before any
  payload runs. Interior bodies with sibling sub-nests stop the chain
  (peeling would duplicate the siblings across threads); everything
  deeper — serial, carried, guards, split segments — walks inside the
  thread in the walker's exact order. A root without a parallel chain
  launches single-thread and the header RECORDS it (Rule 148: never a
  silent loss of parallelism).
- **Host wrapper semantics.** cudaMalloc per bindable buffer + per dims
  array + scalars + status; H2D for inputs AND outputs (accumulate/Max
  stores read the output slot); device temps from static dims,
  zero-initialized (cudaMemset, the walker's model, the same
  kKernelTempElementsLimit enforced at emission); one launch per root
  IN ROOT ORDER with cudaDeviceSynchronize between (the walker's
  sequential root discipline); status readback maps device violations
  to the ABI int; D2H for the writable set; a cleanup ladder frees
  exactly what was allocated on every path. ABI error codes: 0 ok, 1
  negative store flat (checked at every store site — the walker's
  InvalidGraph), 2 cuda runtime alloc/copy, 3 launch/sync, 4 no CUDA
  device.
- **Exactness policy (declared per module, consumed by the gate).**
  `cudaArtifactBitExactPolicy()` classifies the module with the SAME
  scan the emitter's apply() performs: {Add, Sub, Mul, Div, Neg, Sqrt,
  Rsqrt, poly7-Sin} are device-exact (IEEE correctly rounded mul/add/
  div/sqrt; floor/fmod exact; the poly7 device snapshot is pure
  mul/add) — the differential vs the walker is bit-exact BY
  CONSTRUCTION under the mandatory `--fmad=false` build flag. Any
  device-libm op (Exp, Log, libm-Sin, Cos, Tan, Tanh, Erf, Gelu, Pow's
  non-fast form) flips the module to ULP-BOUNDED: the differential is
  MEASURED (max ordered-bit ULP distance) and recorded in the report —
  never asserted, never silently relaxed. NaN/inf constants emit as
  bit-exact intrinsic forms (__longlong_as_double), never libm macros.
- **Driver.** `GpuBackendDriverConfig{compiler="nvcc", arch="sm_70",
  workdirBase}` — the arch is DECLARED (validated sm_<digits>, an
  invalid arch is InvalidArgument; the artifact states the hardware it
  targets) and the build spawns `nvcc -shared -Xcompiler -fPIC
  --fmad=false -arch=<arch> kernel.cu` out-of-process with the same
  log-capturing spawn as the CPU path. buildKernelArtifact(kind=Cuda)
  is an explicit InvalidArgument so the arch flag can never be
  accidentally defaulted. `cudaToolchainAvailable()` is the recorded
  PATH probe (device PRESENCE is intentionally not probed — it is
  discovered at run time through the artifact's own ABI code 4).
- **CLI + bench.** `mlk-poly emit <graph.mlk> --cuda [--arch=sm_XX]
  [--compile]` emits/loads the .cu artifact; `mlk-poly bench` gains the
  `cuda` path — honest `cuda-toolchain-unavailable` skip rows when the
  probe fails, bit-exact gating for device-exact modules, and the
  measured max-ULP recorded in the row note for ULP-bounded modules.
- **Verification without a device.** The .cu text is fully
  machine-checkable in-network: structure asserts pin the grid
  collapse, the ABI wrapper, the --fmad=false contract, the negative-
  flat check, and the policy line (cuda_emission_gemm_structure);
  byte-identical re-emission (cuda_emission_deterministic); legacy 1-D
  thread-per-element mapping (cuda_emission_legacy_elementwise); the
  policy classification boundary (cuda_emission_policy_classification);
  the recorded-probe honest skip + arch validation + explicit-selection
  rejection (cuda_driver_honest_skip — which doubles as the LIVE
  four-way bit-exact test on a CUDA-equipped machine); Call-node
  rejection (cuda_emission_rejects_call_nodes). The differential gates
  activate automatically wherever nvcc + a device exist; on CUDA-less
  machines the skip conditions themselves are the recorded checks.

### GPU backend — implementation status (round 20): multi-axis launch geometry + fast-kernel GPU candidates

Two open roadmap items closed:

- **Multi-axis launch geometry (never clamps).** The flat thread id in
  every device kernel is now the FULL hardware linearization over the
  six CUDA axes (x fastest: bz, by, bx block grid; tz, ty, tx block
  threads) — with a 1-D launch it degenerates to the historical
  `blockIdx.x * blockDim.x + threadIdx.x`. The launch geometry itself
  is chosen by a GENERATED host-side helper (`mlk_assign_geometry`,
  plain C, emitted once per module before the wrapper): a deterministic
  greedy over the RUNTIME padded trips — the innermost levels pack onto
  the block axes (bx, then by, then bz; per-axis caps 1024/1024/64,
  cumulative product ≤ 1024 threads), the remaining levels pack onto
  the grid axes innermost-first (gx, then gy, then gz; caps 2^31-1 /
  65535 / 65535 — gy/gz are the overflow valves for grids beyond the
  x-axis limit), and any shape that fits no axis keeps the historical
  1-D flat fallback (grid = ceil(total/1024), block = min(total,1024)).
  The helper NEVER clamps a dim: multi-axis launches satisfy
  threads == total EXACTLY, so the hardware thread space bijectively
  covers the padded instance space and every instance executes exactly
  once (the fallback's ceil excess is absorbed by the existing
  `mlk_flat >= mlk_total` guard). Soundness is assignment-invariant:
  the device's per-level div/mod chain reads the flat index directly
  (row-major over the level trips), so the geometry only shapes HOW
  flat values are realized in hardware, never WHAT they mean — for any
  dims the positional linearization is a bijection onto [0, threads).
  Cap checks are division-form (no overflow); serial roots launch
  1×1 through the same helper (n = 0). The generated helper is plain
  C and is behaviorally pinned by
  `cuda_emission_geometry_helper_behavioral`: the test extracts THIS
  exact text from the emitted artifact, compiles it with cc, and
  checks hand-computed assignments (tile-band packing, the gy overflow
  valve, the 1-D fallback shape), the hardware-caps envelope on every
  returned dim, the threads==total / threads>=total soundness
  condition, and exhaustive hardware-index → flat injectivity (rank
  == flat for every hardware tuple; each instance realized once).
- **Fast-kernel GPU candidates.** `ExecPath::NativeCuda` joins the
  DECLARED search space (default execPaths now walker/asm/cpp/cuda):
  Cuda candidates build through the GPU driver entry points (the arch
  flag is DECLARED per `FastKernelSearchConfig.gpu`, never defaulted),
  flow through the same budget gate → artifact-cache probe → bit-exact
  identity differential → benchmarked U^run pipeline as the CPU native
  paths, and — where the toolchain or device is missing — end in a
  structured `CapabilityUnsupported`/`BuildFailed` rejection naming the
  failed probe (the declared space is reported in full, Axiom 14.20;
  the winner claim downgrades to "best certified kernel in searched
  space" honestly). The artifact-cache fingerprint binds the DECLARED
  GPU compiler + arch for Cuda candidates (two arch declarations are
  different artifacts and never share a cache entry). CLI:
  `mlk-poly autotune --exec=walker,asm,cpp,cuda [--arch=sm_XX]`.
- **Honest boundaries (roadmap).** No device has touched these
  artifacts yet in this environment (no nvcc): the emission is
  structure-verified, the geometry helper is behaviorally verified on
  the CPU via the extracted generated text, and the compile/run path
  stays probe-gated. Block-geometry HEURISTICS are untuned (packing is
  chosen for simplicity and determinism; a spread heuristic, shared-
  memory tiling, stream orchestration, and PTX emission stay open —
  each needs hardware evidence before it is claimed as an
  optimization, Rule 90).
