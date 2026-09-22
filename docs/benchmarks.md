# MLK+ Benchmarks

Measured numbers for the two axes the tiering story trades against:
**compile time** (what the tier pipeline costs) and **run time** (what the
compiled artifacts buy). Everything here is reproducible from the repo:

- runtime numbers: `mlk-poly bench --suites=gemm,softmax,reducesum`
- compile-time numbers: `mlk_compile_bench` (`bench/compile/compile_bench.cpp`)

Record: 2026-09-23 — the first full sweep after the P0 correctness
remediation (e-graph identity/transplant, lexMin/lexMax, measured ULP
verify) and the pass-catalog truth round. All rows are gated by the usual
honesty rules: bench claims bit-exactness per row (the `OK` column), CUDA
rows are skipped with `cuda-toolchain-unavailable` when no toolchain exists
(no device in the measurement environment), and no speedup is claimed
without a certificate-bearing measurement.

## Environment

| Item | Value |
|------|-------|
| CPU | Intel Xeon (2 hardware threads, no AVX-512) |
| OS | Linux x86-64 |
| Compiler (repo build) | GCC 14.2, `-std=c++26 -O3 -fno-exceptions -fno-rtti -Werror` |
| Artifact compiler | system `cc` (out-of-process driver), `-O2` for cpp artifacts |
| Protocol | warmup then N reps, median; bit-exact gate per row |

## 1. Runtime: tier-1 vs compiled artifacts vs polyhedral walker

`mlk-poly bench` (30 reps, 5 warmups) — every timed row passed the
bit-exactness gate against the tier1 reference (28/28). Columns: median
ms, speedup vs tier1 (the executor's blocked-GEMM / fused-elementwise
path), and GFLOP/s over the declared essential-work model (2·M·K·N
MatMul, 5·M·K softmax, M·K ReduceSum — a model, never a hardware peak
claim).

| Suite | Shape | Path | Median ms | vs tier1 | GFLOP/s | Bit-exact |
|-------|-------|------|----------:|---------:|--------:|-----------|
| gemm | 64x48x56 | tier1 | 0.036 | 1.00x | 9.47 | OK (reference) |
| gemm | 64x48x56 | walker | 33.21 | 0.00x | 0.010 | OK |
| gemm | 64x48x56 | cpp | 0.027 | 1.35x | 12.80 | OK |
| gemm | 64x48x56 | asm | 0.938 | 0.04x | 0.37 | OK |
| gemm | 128x128x128 | tier1 | 0.270 | 1.00x | 15.53 | OK (reference) |
| gemm | 128x128x128 | walker | 289.04 | 0.00x | 0.015 | OK |
| gemm | 128x128x128 | cpp | 0.348 | 0.78x | 12.05 | OK |
| gemm | 128x128x128 | asm | 11.37 | 0.02x | 0.37 | OK |
| gemm | 256x256x256 | tier1 | 2.157 | 1.00x | 15.56 | OK (reference) |
| gemm | 256x256x256 | walker | 2206.25 | 0.00x | 0.015 | OK |
| gemm | 256x256x256 | cpp | 2.789 | 0.77x | 12.03 | OK |
| gemm | 256x256x256 | asm | 90.30 | 0.02x | 0.37 | OK |
| softmax | 64x64 | tier1 | 0.025 | 1.00x | 0.82 | OK (reference) |
| softmax | 64x64 | walker | 0.640 | 0.04x | 0.032 | OK |
| softmax | 64x64 | cpp | 0.035 | 0.72x | 0.59 | OK |
| softmax | 64x64 | asm | 0.074 | 0.34x | 0.28 | OK |
| softmax | 256x128 | tier1 | 0.221 | 1.00x | 0.74 | OK (reference) |
| softmax | 256x128 | walker | 5.088 | 0.04x | 0.032 | OK |
| softmax | 256x128 | cpp | 0.286 | 0.77x | 0.57 | OK |
| softmax | 256x128 | asm | 0.597 | 0.37x | 0.27 | OK |
| reducesum | 256x256 | tier1 | 0.026 | 1.00x | 2.51 | OK (reference) |
| reducesum | 256x256 | walker | 2.698 | 0.01x | 0.024 | OK |
| reducesum | 256x256 | cpp | 0.018 | 1.44x | 3.61 | OK |
| reducesum | 256x256 | asm | 0.231 | 0.11x | 0.28 | OK |
| reducesum | 1024x64 | tier1 | 0.015 | 1.00x | 4.32 | OK (reference) |
| reducesum | 1024x64 | walker | 2.796 | 0.01x | 0.023 | OK |
| reducesum | 1024x64 | cpp | 0.012 | 1.27x | 5.48 | OK |
| reducesum | 1024x64 | asm | 0.231 | 0.07x | 0.28 | OK |

(CUDA rows, one per case, are honest skips in this environment — no nvcc,
no device; the bench prints `cuda-toolchain-unavailable` rather than
fabricating numbers. On a CUDA host the same verb fills them under each
module's declared exactness policy.)

Reading (honest interpretation, Rule 90 — no headline shopping):

- **cpp artifacts are at parity with tier1** (0.72x–1.44x). tier1's GEMM
  rides the executor's blocked Call kernel, which the naive generated
  triple loop does not beat on this 2-thread box; elementwise/reduce
  chains win 1.3–1.4x once the fused chain survives compilation.
- **asm artifacts are correctness-first**: scalar SSE2 with the
  interpreter's exact FP operation order (no FMA contraction, no
  reassociation), so they trade throughput for bit-exact provenance —
  the 0.28–0.37 GFLOP/s numbers are the cost of that guarantee, not a
  code quality claim. Packed `mulpd/addpd` remains legal only across an
  independent output dim (lanes keep each cell's k-chain order), needs a
  j-innermost interchange from the codegen, and stays the best-payoff
  asm roadmap item.
- **the polyhedral walker is an interpreter**, not a code path to ship:
  its value is the bit-exact differential oracle for every schedule the
  compiler produces.

## 2. Compile time: tier pipelines

Median full-pipeline wall time over 7 reps (fresh graph snapshot per
rep), scalar graph `sin(x^2+3x)`, profile `scalar_f64`:

| Tier | Compile ms | What the tier adds |
|------|-----------:|--------------------|
| 0 | 0.003 | verification + inference only |
| 1 | 0.020 | canonicalize/cse/dce + schedule stamps + kernel lowering |
| 2 | 0.062 | approx gates, calculus gates (capability-skipped), poly pipeline |
| 3 | 0.077 | e-graph build/saturate/extract on top of tier2 |

The whole tier-3 pipeline — e-graph saturation included — is well under
0.1 ms for a scalar expression graph (run-to-run medians vary by a few
tens of microseconds at this scale; the structural result — tier3 >
tier0, everything sub-0.1 ms — is stable). The passes are budgeted and
the graph is tiny.

### Where the seconds go: GEMM 128^3 tier-2 phase split

Kill-switch isolations over the same protocol (profile
`tensor_f64_cpu`):

| Configuration | Median ms | Delta reading |
|---------------|----------:|----------------|
| tier2 full | 2892.8 | — |
| without poly passes | 0.023 | analysis + lowering ≈ 0.02 ms |
| without schedule/tile/codegen | 0.136 | synth+scop+dependence ≈ 0.1 ms |
| without poly.verify | 2893.0 | poly.verify ≈ 0.2 ms |

**The Pluto-style scheduling LP dominates**: for the 128^3 GEMM the
scheduler's Farkas-LP search over affine forms accounts for ~99.99% of
the ~2.9 s tier-2 compile; verification of the produced schedule is
sub-millisecond. That is the honest compile-time/run-time trade the
polyhedral pipeline offers today (Rule 131 budgets bound it per SCoP; it
is a one-time cost per specialized workload, and the artifact caches
fingerprint-reuse the result).

## 3. Scalar dispatch: interpreter vs tier-1 kernel

Median over 2000+ reps after warmup, pre-installed realization:

| Path | Median us |
|------|----------:|
| `interpretGraph` (5-node graph) | 0.29 |
| `ExecutionEngine::execute` tier-1 (kernel installed) | 1.65 |

For a 5-node scalar graph the kernel path's fixed ABI costs (binding
construction, executor dispatch, output read-back) exceed the
interpreter's cost — tiny graphs belong on Tier 0 (this is exactly the
cold-start tiering contract: tier kernels pay off on real element
counts, and the fastkernel search picks the execution path with the
certificate, not the label).

## 4. Correctness gates in the loop

- every `mlk-poly bench` row: bit-exact agreement (tier1 reference vs
  walker vs cpp vs asm) before the timing counts, `OK` printed per row;
- `poly.verify` re-derives the schedule legality per compile
  (sub-millisecond at 128^3 — see the phase split above);
- `approx.ulp_verify` measures the poly7 family's true ULP error
  against libm on the deterministic sample set at every tier-2/3
  compile (the measured bound, never an asserted constant).
