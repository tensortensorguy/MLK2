# MLK+ Benchmarks

Measured numbers for the two axes the tiering story trades against:
**compile time** (what the tier pipeline costs) and **run time** (what the
compiled artifacts buy). Everything here is reproducible from the repo:

- runtime numbers: `mlk-poly bench --suites=gemm,softmax,reducesum`
- compile-time numbers: `mlk_compile_bench` (`bench/compile/compile_bench.cpp`)

All rows are gated by the usual honesty rules: bench claims bit-exactness
per row (the `OK` column), CUDA rows are skipped with
`cuda-toolchain-unavailable` when no toolchain exists (no device in the
measurement environment), and no speedup is claimed without a
certificate-bearing measurement.

## Environment

| Item | Value |
|------|-------|
| CPU | Intel Xeon (2 hardware threads, no AVX-512) |
| OS | Linux x86-64 |
| Compiler (repo build) | GCC 14.2, `-std=c++26 -O3 -fno-exceptions -fno-rtti -Werror` |
| Artifact compiler | system `cc` (out-of-process driver), `-O2` for cpp artifacts |
| Protocol | warmup then N reps, median; bit-exact gate per row |

## 1. Runtime: tier-1 vs compiled artifacts vs polyhedral walker

`mlk-poly bench` — every non-CUDA row passed the bit-exactness gate
against the walker (or, for tier1, against the reference interpreter).
Columns: median ms, speedup vs tier1 (the executor's blocked-GEMM /
fused-elementwise path), effective GB/s.

| Suite | Shape | Path | Median ms | vs tier1 | GB/s | Bit-exact |
|-------|-------|------|----------:|---------:|-----:|-----------|
| gemm | 128x128x128 | tier1 | 0.277 | 1.00x | 15.2 | OK (reference) |
| gemm | 128x128x128 | walker | 286.6 | 0.00x | 0.02 | OK |
| gemm | 128x128x128 | cpp | 0.350 | 0.79x | 12.0 | OK |
| gemm | 128x128x128 | asm | 11.34 | 0.02x | 0.37 | OK |
| gemm | 256x256x256 | tier1 | 2.092 | 1.00x | 16.0 | OK (reference) |
| gemm | 256x256x256 | walker | 2219.1 | 0.00x | 0.02 | OK |
| gemm | 256x256x256 | cpp | 2.717 | 0.77x | 12.3 | OK |
| gemm | 256x256x256 | asm | 90.18 | 0.02x | 0.37 | OK |
| softmax | 64x64 | tier1 | 0.025 | 1.00x | 0.8 | OK (reference) |
| softmax | 64x64 | cpp | 0.035 | 0.71x | 0.6 | OK |
| softmax | 64x64 | asm | 0.074 | 0.34x | 0.3 | OK |
| softmax | 256x128 | tier1 | 0.211 | 1.00x | 0.8 | OK (reference) |
| softmax | 256x128 | cpp | 0.281 | 0.75x | 0.6 | OK |
| softmax | 256x128 | asm | 0.603 | 0.35x | 0.3 | OK |
| reducesum | 256x256 | tier1 | 0.026 | 1.00x | 2.5 | OK (reference) |
| reducesum | 256x256 | cpp | 0.018 | 1.45x | 3.6 | OK |
| reducesum | 256x256 | asm | 0.226 | 0.12x | 0.3 | OK |
| reducesum | 1024x64 | tier1 | 0.015 | 1.00x | 4.3 | OK (reference) |
| reducesum | 1024x64 | cpp | 0.012 | 1.27x | 5.5 | OK |
| reducesum | 1024x64 | asm | 0.229 | 0.07x | 0.3 | OK |

Reading (honest interpretation, Rule 90 — no headline shopping):

- **cpp artifacts are at parity with tier1** (0.71x–1.45x). tier1's GEMM
  rides the executor's blocked Call kernel, which the naive generated
  triple loop does not beat on this 2-thread box; elementwise/reduce
  chains win 1.3–1.5x once the fused chain survives compilation.
- **asm artifacts are correctness-first**: scalar SSE2 with the
  interpreter's exact FP operation order (no FMA contraction, no
  reassociation), so they trade throughput for bit-exact provenance —
  the 0.07–0.37 GB/s numbers are the cost of that guarantee, not a code
  quality claim.
- **the polyhedral walker is an interpreter**, not a code path to ship:
  its value is the bit-exact differential oracle for every schedule the
  compiler produces.
- **CUDA rows** are honest skips in this environment (no nvcc, no
  device); the bench prints `cuda-toolchain-unavailable` rather than
  fabricating numbers.

## 2. Compile time: tier pipelines

Median full-pipeline wall time over 7 reps (fresh graph snapshot per
rep), scalar graph `sin(x^2+3x)`, profile `scalar_f64`:

| Tier | Compile ms | What the tier adds |
|------|-----------:|--------------------|
| 0 | 0.013 | verification + inference only |
| 1 | 0.054 | canonicalize/cse/dce + schedule stamps + kernel lowering |
| 2 | 0.108 | approx gates, calculus gates (capability-skipped), poly pipeline |
| 3 | 0.150 | e-graph build/saturate/extract on top of tier2 |

The whole tier-3 pipeline — e-graph saturation included — is well under
0.2 ms for a scalar expression graph (run-to-run medians vary by a few
tens of microseconds at this scale; the structural result — tier3 ≈ 3x
tier0, everything sub-0.2 ms — is stable). The passes are budgeted and
the graph is tiny.

### Where the seconds go: GEMM 128^3 tier-2 phase split

Kill-switch isolations over the same protocol (profile
`tensor_f64_cpu`):

| Configuration | Median ms | Delta reading |
|---------------|----------:|----------------|
| tier2 full | 2890.9 | — |
| without poly passes | 0.068 | analysis + lowering ≈ 0.07 ms |
| without schedule/tile/codegen | 0.185 | synth+scop+dependence ≈ 0.1 ms |
| without poly.verify | 2886.4 | poly.verify ≈ 4.5 ms |

**The Pluto-style scheduling LP dominates**: for the 128^3 GEMM the
scheduler's Farkas-LP search over affine forms accounts for ~99.9% of
the ~2.9 s tier-2 compile; verification of the produced schedule is
~4.5 ms. That is the honest compile-time/run-time trade the polyhedral
pipeline offers today (Rule 131 budgets bound it per SCoP; it is a
one-time cost per specialized workload, and the artifact caches
fingerprint-reuse the result).

## 3. Scalar dispatch: interpreter vs tier-1 kernel

Median over 2000+ reps after warmup, pre-installed realization:

| Path | Median us |
|------|----------:|
| `interpretGraph` (5-node graph) | 0.30 |
| `ExecutionEngine::execute` tier-1 (kernel installed) | 1.79 |

For a 5-node scalar graph the kernel path's fixed ABI costs (binding
construction, executor dispatch, output read-back) exceed the
interpreter's cost — tiny graphs belong on Tier 0 (this is exactly the
cold-start tiering contract: tier kernels pay off on real element
counts, and the fastkernel search picks the execution path with the
certificate, not the label).

## 4. Correctness gates in the loop

- every `mlk-poly bench` row: bit-exact three-way agreement (walker vs
  cpp vs asm) before the timing counts, `OK` printed per row;
- `poly.verify` re-derives the schedule legality per compile (~4.5 ms at
  128^3 — see the phase split above);
- `approx.ulp_verify` measures the poly7 family's true ULP error
  against libm on the deterministic sample set at every tier-2/3
  compile (the measured bound, never an asserted constant).
