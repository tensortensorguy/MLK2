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
poly.schedule               identity prefix + separator rows (Feautrier-
                            style hard resolution; see §scheduling)
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

Row synthesis has two phases:

1. **Identity prefix** — `theta_r = v_r` for `r < depth`. This reproduces
   the original program order, which is legal by construction for
   extraction-produced SCoPs, and yields the classic `[i, j, k]` GEMM
   schedule with i/j parallel-marked (no dependence carried) and k
   carrying the reduction.
2. **Separator rows** — dependences the identity never strictly separates
   (cross-statement same-position pairs, e.g. init vs accumulator) are
   resolved Feautrier-style: one primary dependence per row receives a
   hard `dist >= 1` constraint (Farkas-encoded), the others keep
   `dist >= 0` with epigraph objectives minimizing their max distances
   (the fusion driver). Intra-statement primaries are tried first so
   dimension rows precede statement-order separators.

Parallel marking: a row is parallel iff every live dependence distance is
identically zero on its refined slice (a row that *carries* a dependence —
strictly positive distance — is sequential). The innermost parallel row is
the vectorizable one. `verifyScheduleLegality` re-proves the final
schedule lexicographically; `poly.verify` runs it before the backend.

Known quality limitation: hard per-row resolution does not implement the
full Pluto fusion objective (skewed multi-row separations); schedules are
legal and deterministic but not always distance-optimal. The autotuner
gates adoption by measurement (Rule 32); full Pluto ILP formulation is the
roadmap item.

## Codegen

Emission walks schedule levels over the statement group:

- **Separator levels** (all-constant rows) emit statements in
  (time, origOrder) order.
- **Loop levels** (row == ±e_d) emit one shared loop per dimension
  (fusion). Statements whose domain pins the dim (singletons — the init
  statement's `k == 0`) hoist before the loop when their remaining rows
  are constants dominated by the varying statements' minimums
  (`canEmitBefore`, conservative box check).
- **Tiled band levels** emit a TILE loop (constant bounds
  `floor(lo/t)`, `floor(hi/t)`) plus a POINT loop with affine bounds
  `t*ti … t*ti + t - 1`; a trailing partial tile (when `(hi+1) % t != 0`)
  carries a constant clipped end. Inner-tree duplication per split is
  bounded by `kPolyMaxCodegenCopies`.
- Payloads replay the statement Compute chain and Store (accumulate
  preserved) with ElemIdx operands re-indexed from scop-dim to
  stack-position coefficients; store targets map the same way.

Kernel IR extensions (all backward compatible, hashed and serialized per
Rule 24): `KernelOperand::Kind::ElemIdx` (buffer + per-stack-position
coefficients + offset), affine loop bounds (`beginCoeffs/endCoeffs` over
the enclosing var stack), buffer-dim loop bounds (`endBuf/endDim`), and
`Store::accumulate`.

Bail conditions (conservative, baseline preserved): non-unimodular rows,
fission across dims at one level, unaligned tile lower bounds, copy-budget
exhaustion, unbounded levels.

## Executor

`executeKernelOnBuffers` routes any module containing multi-dim features
through a recursive tree walker (loop-var stack, affine bounds, ElemIdx
resolution, accumulate stores); the legacy 1-D fast path keeps its
chunked threading. Multi-dim execution is single-threaded in v1 (Rule 12
roadmap).

## Verification story

- Unit: 37 tests in `tests/unit/unit_poly.cpp` covering the engine
  (rationals, sets, FM, lexmin, maps), extraction, dependences, LP,
  scheduling legality + determinism, tiling, codegen structure, the
  full pass chain, and differential execution.
- Differential (Rules 43/85/90): the transformed GEMM executes over dense
  buffers and matches a straightforward reference (bit-exact for the
  no-reassociation class; accumulation order per output cell preserved).
- `poly.verify` re-proves legality inside every Tier2/3 compilation and
  restores the baseline on failure.

## Roadmap

- Full Pluto ILP objective (skewed multi-row fusion schedules).
- Parametric SCoPs (symbolic dims with runtime guards) — currently
  requires constant bounds after workload specialization.
- ReduceSum/softmax synthesis classes; >2-input elementwise (needs the
  baseline buffer-materialization limit lifted).
- Multi-threaded multi-dim execution (Rule 12).
- Multiple SCoP regions per kernel.
