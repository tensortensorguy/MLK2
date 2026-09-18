# MLK+ — Mathematical Language Kernel Plus

**A multi-domain mathematical graph compiler, autotuner, superoptimizer, and
runtime.** Tensor is one domain built on top of a general mathematical graph
— not the identity of the IR.

> Mathematics first. Representation second. Hardware third.

Implemented in **C++26** (`-std=c++26 -fno-exceptions -fno-rtti`) per the
MLK+ Compiler Laws & Architecture Specification — all 160 rules mapped in
[docs/compliance_matrix.md](docs/compliance_matrix.md).

## Quick start

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build

# compile + run the worked example f(x) = sin(x^2 + 3x)
./build/bin/mlkc compile examples/scalar/sin_graph.mlk --tier=1 --emit=ir
./build/bin/mlkc run     examples/scalar/sin_graph.mlk --tier=0 --x2.0
./build/bin/mlk-verify   examples/scalar/sin_graph.mlk
./build/bin/mlk-tune     tune     examples/scalar/sin_graph.mlk
```

## What is inside

| Area | Contents |
|---|---|
| `compiler/` | Math IR (index-based, interned, effect-carrying), type/property/effect inference, proof/accuracy contracts, pass framework with contracts + kill switches, 54 registered passes (analysis, canonicalization, e-graph, calculus, tensor, approximation, schedule, physical, polyhedral, lowering, backend), pipeline runner with per-tier pipelines |
| `runtime/` | Tier 0 reference interpreter, GraphState/fallback/deoptimization, async tier-up with atomic kernel publication, telemetry, realization cache |
| `autotuner/` | Declarative search spaces, verify-before-benchmark harness, roofline pruning, noise filtering, complete-key caching |
| `superopt/` | Certificated candidate generators: scalar peephole, math-function approximation (measured-ULP degree-13 sin: max 2 ULPs vs libm), algebraic e-graph optimizer |
| `polyhedral/` | Hermetic polyhedral engine (Presburger sets, Fourier-Motzkin, exact lexmin, rational simplex) + the poly.* pass family: SCoP extraction, exact dependence analysis, affine scheduling, tiling, Kernel IR code generation, schedule verification (docs/polyhedral_spec.md) |
| `backends/` | Tier 0 interpreter backend, CPU backend (KernelModule execution + native AOT artifacts: C++ source, x86-64 assembly, and CUDA emission; out-of-process cc/nvcc build + dlopen driver — docs/polyhedral_spec.md §backend, §GPU-backend) |
| `fastkernel/` | Certified, budgeted fast-kernel search (OEIA/Refined/FastKernels): per-variant identity/runtime/comptime/launch certificates, budget modes with regression guard, roofline lower bound, fingerprint-validated artifact cache, honest winner claims — docs/polyhedral_spec.md §fast-kernel-search, `mlk-poly autotune` |
| `tools/` | `mlkc`, `mlk-poly`, `mlk-verify`, `mlk-tune`, `mlk-bench`, `mlk-profile`, `mlk-replay`, `mlk-generate-profiles` |
| `tests/` | unit (10 suites), differential (Rule 43), fallback (Rule 102), fuzz (Rule 152), security (Rule 124), replay (Rule 158) |
| `docs/` | architecture, IR/property/effect/profile/autotuner/superoptimizer/bytecode/ABI/cache specs, 47 pass docs, ADRs, compliance matrix |

## Design laws (selection)

- **Rule 15**: graph references are 32-bit indices; cache-friendly and
  serializable.
- **Rule 21**: equivalence is recorded, never silently destroyed — the
  e-graph keeps equivalent forms alive; originals stay recoverable.
- **Rule 22**: properties are first-class and tri-state; Unknown never
  silently becomes True.
- **Rule 33**: no algebraic rewrite without legality — floating-point
  addition is not associative without a contract; matrix multiplication is
  never assumed commutative.
- **Rule 34**: no approximation without an explicit, verified error
  contract.
- **Rule 6/8**: no exceptions, no RTTI — `std::expected` + the sanctioned
  `MLK_TRY` macros (ADR-0002).

See [docs/architecture.md](docs/architecture.md) and the four-tier
realization model (Tier 0 oracle → Tier 1 baseline → Tier 2 autotuned →
Tier 3 compile=INF supercompiled).

## License

Apache-2.0
