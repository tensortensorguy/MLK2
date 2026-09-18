# MLK+ Compliance Matrix

Rule → enforcement mapping (spec Part XVI). Reviewed each release.
Enforcement keys: **build** = compiler flags, **lint** = scripts/lint.sh,
**test** = automated suite, **arch** = structural/type-system guarantee,
**doc** = documented deviation/waiver, **todo** = tracked roadmap item
(waiver per Part XV: rule ID, reason, owner, risk, mitigation, telemetry,
expiry).

| Rule | Requirement (summary) | Enforcement |
|---|---|---|
| 1 | Portable Math IR/bytecode for all domains | test (unit_ir, differential), arch |
| 2 | Math Domain Profile per domain | test (domain_profile loader), profiles/ |
| 3 | Semantic oracle authoritative | test (differential: Tier0 ground truth) |
| 4 | Runtime capability declarations | arch (CapabilityFlags), profiles/ |
| 5 | GraphState on every guard | test (graph_verifier, unit_runtime) |
| 6 | No exceptions on hot paths | build (-fno-exceptions), lint (throw grep) |
| 7 | PMR monotonic allocation, bulk-free | arch (MonotonicArena), todo: wired into every pass arena |
| 8 | No RTTI | build (-fno-rtti), lint (dynamic_cast/typeid grep) |
| 9 | No shared_ptr/function in hot IR | arch (raw ids), lint |
| 10 | Passes idempotent + monotonic, budgeted fixpoints | test (passes_are_idempotent, egraph budgets) |
| 11 | Evaluation threads never block on JIT | test (unit_runtime), arch (async compile) |
| 12 | Thread-local allocation for evaluation | todo (single-threaded MVP; pool API ready) |
| 13 | Compiler works on frozen snapshots | arch (immutable snapshot contract), doc |
| 14 | Epoch-based reclamation | arch (shared_ptr quiescence on swap), doc |
| 15 | Index-based graph (uint32 ids) | arch (NodeId/ValueId), test |
| 16 | Interned symbols; no std::string in IR | arch (SymbolId), lint |
| 17 | Open-addressing hash maps in hot paths | arch (OpenHashMap/Set), lint |
| 18 | SparseSet/BitVector for dataflow | arch, test (unit_core) |
| 19 | SmallVector for 1-4 element data | arch, test (unit_core) |
| 20 | SoA for bulk pass processing | todo (graphs are SoA-ready via value/node arrays) |
| 21 | Equivalence classes, non-destructive rewriting | arch (recordEquivalent), test (unit_egraph, unit_ir) |
| 22 | Tri-state properties; Unknown ≠ True | arch (TriState), test (property inference) |
| 23 | Math state separate from physical state | arch (physical layer refs math) |
| 24 | Stable hashes + versioned serialization | test (hash_stability, json roundtrips) |
| 25 | [[likely]]/[[unlikely]]/[[assume]] | arch (used in macros/graph indexing) |
| 26 | Zero-cost error propagation (expected/TRY) | arch (Result everywhere), ADR-0002 |
| 27 | No magic numbers; named constexpr | arch (core/constants.h), review |
| 28 | No target/domain hacks in generic passes | arch (profiles + op tables), test (capability gates) |
| 29 | Heuristics empirically validated, overridable | doc (tile/vector defaults + knobs), test (autotune) |
| 30 | No silent fallbacks without telemetry | test (unit_runtime, fallback), arch |
| 31 | No stable-hardware assumptions | arch (HardwareInfo::detect) |
| 32 | No optimization without measurable win | test (autotune measures), bench harness |
| 33 | No algebraic rewrite without legality | test (neg-zero, matmul commutativity), arch |
| 34 | No approximation without error contract | test (policy gate, ulp_verify), arch |
| 35 | No FFI optimization without ABI proof | arch (FFI = effect barrier; no FFI opts) |
| 36 | No vectorization without dependence proof | doc (SSA freshness proof), test |
| 37 | Persistent state versioned | test (security: cache version reject) |
| 38 | Bitmasked orthogonal state (Flags<E>) | arch (Flags), test |
| 39 | No implicit conversions in IR | test (type_infer rejects mismatch) |
| 40 | Physicalization preserves semantics | arch (physical refs math; Rule 146 checks) |
| 41 | 5 regression tests per bugfix | process (this build fixed fold/pow bugs with suite coverage) |
| 42 | ≥10 golden tests per pass | test (unit case tables per pass + golden runner), todo: file-based goldens expanded |
| 43 | Differential testing in CI | test (differential suite, all tiers) |
| 44 | Weekly fallback fuzz | test (fuzz suite; CI cron todo) |
| 45 | Replay logs for CI failures | arch (replay artifacts), test (replay) |
| 46 | Perf regressions need waiver | process (bench gates; perf CI todo) |
| 47 | Verifier after every pass (debug) | arch (PipelineRunner verifyBetweenPasses) |
| 48 | Test names encode bug/feature | process (descriptive suite/test names) |
| 49 | Statistically valid benchmarks | test (benchmarker protocol) |
| 50 | Superopt candidates verified before bench | test (superopt ULP verify) |
| 51 | Superoptimizers modular + domain-gated | arch (plugin interface), test |
| 52 | Every candidate carries certificate | test (certificate fields), arch |
| 53 | compile=INF deterministic + cancellable | arch (deterministic e-graph, cancellation) |
| 54 | Declarative search spaces | test (search_space roundtrip/reject) |
| 55 | Cost models include lower bounds | arch (roofline), test |
| 56 | Benchmark real workloads | arch (seeded realistic inputs) |
| 57 | Complete cache keys | test (security suite) |
| 58 | Correctness precedes benchmarking | test (autotune verify-first) |
| 59 | Noise handling; ties prefer simplicity | arch (noise filter + seed-priority) |
| 60 | Kill switches everywhere | arch (ctx.killed per pass/plugin), lint |
| 61 | Assumptions have watchdog + trip | arch (SpeculationInfo invalidation dep) |
| 62 | Specialization has fallback + budget | test (fallback suites) |
| 63 | Profile data carries confidence | arch (SpeculationInfo.confidence) |
| 64 | Aggressive passes use cost models | arch (cost.roofline feeds pruning) |
| 65 | Guards carry complete metadata | test (verifier guard checks) |
| 66 | Pre-commit < 2s | scripts/run_precommit.sh (29 ms measured) |
| 67 | Actionable diagnostics | test (diag fields: expected/actual/rule/fix) |
| 68 | [[nodiscard]] on Result returns | build (-Werror=unused-result) |
| 69 | No #define macros for logic | arch (only Rule-26-sanctioned TRY macros), ADR-0002 |
| 70 | Fast incremental builds | build (per-library targets, Ninja) |
| 71 | Scripted refactoring | scripts/gen_pass_registry.py, gen_docs.py |
| 72 | Self-contained reproducible tests | test (hermetic harness, no network) |
| 73 | No "small bug" rationalization | process (fold/pow P0s found + fixed this build) |
| 74 | No workarounds for compiler bugs | process (root-cause fixes only) |
| 75 | No implicit knowledge transfer | doc (docs/, ADRs, pass docs) |
| 76 | No premature simplification | process (two-use-case rule noted in ADRs) |
| 77 | No copy-paste; declarative tables | arch (op/property/effect/cost tables) |
| 78 | Exhaustive switches; no default returns | build (-Werror=switch), lint |
| 79 | No lazy algorithms (linear scan etc.) | arch (O(1) lookups via open addressing) |
| 80 | No untested code paths | test (15 suites, 60+ cases) |
| 81 | Performance-aware implementation | build (-O3), arch (SBO, open addressing) |
| 82 | No deletion-by-avoidance | process (hard parts implemented or tracked with roadmap) |
| 83 | No fragile implementations | test (fuzz, security, malformed inputs) |
| 84 | Slop detection checklist | lint + review checklist below |
| 85 | Oracle authoritative | test (differential) |
| 86 | Observable effects not reordered | arch (effect chain), verifier |
| 87 | Only pure expressions folded | test (constant_fold gates) |
| 88 | Dynamic features = correctness requirements | arch (symbolic ops preserved at Tier 0) |
| 89 | Specializations versioned + invalidatable | arch (graph.version) |
| 90 | Numeric semantics preserved exactly | test (neg-zero, NaN creation gates) |
| 91 | Approximation is a semantic change | test (policy gate records decision) |
| 92 | Value elimination keeps observables | test (dce effect/output protection) |
| 93 | Math exceptions are values | arch (interpreter sentinel + events) |
| 94 | States reconstructible on demand | test (GraphState roundtrip) |
| 95 | Async/streaming/iterative safety | todo (async compile tested; streaming roadmap) |
| 96 | Debugging/tracing remain correct | arch (Trace kernel op; hooks stable) |
| 97 | Value identity/reference mgmt explicit | doc (kernel_abi.md), todo where applicable |
| 98 | No assumptions about hashes/addresses | test (hash stability), arch (no ASLR deps) |
| 99 | Static typing ≠ runtime proof unless certified | arch (guards or proofs; Rule 99 checks in verifier) |
| 100 | Fallback metadata = required output | test (emit_binary fallback marker) |
| 101 | GraphState complete + machine-checkable | test (rejects incomplete) |
| 102 | Guard failure → exact lower-tier state | test (fallback suite) |
| 103 | Fallback loops detected + throttled | test (site throttling) |
| 104 | Speculative side effects reversible/deferred | arch (pure-only speculation in MVP) |
| 105 | Managed references tracked | doc (no moving GC in MVP; kernel_abi.md) |
| 106 | Correct read/write barriers | doc (N/A without moving GC — documented) |
| 107 | Long kernels support cancellation | arch (Safepoint polls) |
| 108 | Runtime frames walkable | todo (metadata model defined; walk tooling roadmap) |
| 109 | Stack overflow checks | todo (recursion depth bounds implemented in symbolic diff/egraph) |
| 110 | Allocation fast paths fail safely | doc (no-exceptions semantics documented) |
| 111 | Runtime call transitions preserve ABI | arch (C ABI for kernel entry) |
| 112 | FFI opaque unless proven | arch (FFI effects; no cross-boundary opts) |
| 113 | Weak handles/finalizers see valid state | doc (N/A in MVP — documented) |
| 114 | Shape/layout mutation invalidates code | arch (version-based invalidation) |
| 115 | Tier 0 universal fallback | test (differential; execute fallback path) |
| 116 | W^X executable memory | doc (no executable pages; ADR-0003) |
| 117 | Atomic code publication | arch (release/acquire swap) |
| 118 | Concurrent patching safety | doc (no patching; ADR-0003) |
| 119 | Old code freed after quiescence | arch (shared ownership) |
| 120 | Artifacts record dependencies | arch (CacheKey dependencies) |
| 121 | Generated code constrained | arch (Call = approved entrypoints only) |
| 122 | Platform exploit mitigations | todo (build flags: PIE/RELRO via default toolchain; hardening roadmap) |
| 123 | JIT spraying defenses | doc (no executable pages; ADR-0003) |
| 124 | Profiles/bytecode/artifacts untrusted | test (fuzz, security, loader) |
| 125 | Code cache pressure managed | arch (bounded ring telemetry; cache budgets todo) |
| 126 | AOT artifacts carry manifest | test (cache entry manifest fields) |
| 127 | AOT artifacts verified before load | test (security suite) |
| 128 | Shared state race-free; TSAN-clean | test (thread suite; TSAN in run_ci_full) |
| 129 | Kernel pointer swaps safe/reversible | arch (atomic swap, old retained) |
| 130 | Safepoint latency bounded | arch (single atomic load) |
| 131 | Explicit compile/tune budgets | arch (PassBudget; violations → fallback) |
| 132 | Compilations cancellable | arch (CancellationToken through pipeline) |
| 133 | Hotness counters robust | arch (saturating counters, telemetry ring) |
| 134 | Recompilation throttled | arch (fallback throttle; recompile backoff todo) |
| 135 | OSR/live replacement semantically exact | todo (GraphState machinery ready; OSR roadmap) |
| 136 | Invalidation ordered + visible | arch (release/acquire protocol) |
| 137 | No global locks on hot paths | arch (atomic dispatch; mutex only cold telemetry) |
| 138 | Tier transitions observable | test (telemetry tier events) |
| 139 | Compiler bugs never crash programs | arch (fallback + telemetry; tests) |
| 140 | Explicit effect model | arch (EffectSet/chain), test |
| 141 | Speculative nodes carry metadata | test (verifier checks) |
| 142 | Passes declare contracts | arch (PassContract; tier checks in runner) |
| 143 | Deterministic, replayable compilation | test (sorted registry, deterministic passes) |
| 144 | No hidden global mutable state | arch (context-passed; sanctioned symbol table) |
| 145 | Verifier checks fallback/proof/memory metadata | test (verifier suite) |
| 146 | Backend lowering preserves semantics | arch (effect order + contract carried) |
| 147 | Register/buffer allocation managed-ref safe | doc (no managed refs in MVP; kernel_abi.md) |
| 148 | Target features gated + recorded | arch (HardwareInfo gates; emitter records) |
| 149 | Every node/superopt/trampoline specified | doc (math_ir_spec.md + pass docs) |
| 150 | Static proofs machine-checkable; kill switches | arch (certificates + predicates; kill map) |
| 151 | Differential oracle testing continuous | test (differential suite in CI) |
| 152 | Fuzzing covers IR/artifacts | test (fuzz suite) |
| 153 | Sanitizer matrix | build (run_ci_full.sh: ASan/UBSan/TSan) |
| 154 | Memory/fallback stress modes | todo (stress harness roadmap; fallback forced in tests) |
| 155 | Code install/patching concurrency-tested | doc (no patching; ADR-0003), test (async swap) |
| 156 | Perf gates measure multi-dimensional | bench harness (min/median/stddev/ops); full gate CI todo |
| 157 | Telemetry structured/stable/private | test (security: no user data) |
| 158 | Replay artifacts sufficient | test (replay suite) |
| 159 | ABI/FFI/security tests in CI | test (security suite; FFI tests roadmap) |
| 160 | Governance/ADRs/compliance/hermetic | doc (this matrix + ADRs), build (hermetic default) |
| 161 | Polyhedral schedules are legal (dependences preserved) or the baseline kernel is kept | test (unit_poly legality + differential execution), poly.verify in-pipeline |
| 162 | Polyhedral engine decisions are exact (no rounded legality) | arch (exact rationals, checked int64, tri-state Feasibility), test (unit_poly) |
| 163 | Polyhedral guarded re-entry is schedule-legal (const-slot normalization + integer-exact pivot-coordinate realizability gate) | arch (polyhedral_spec §scheduling/codegen), test (unit_poly guarded GEMM structure + bit-exact tiled/untiled differential) |
| 164 | Parallel/vector marks follow INTEGER instance points, not the rational hull; guard execution is thread-safe (var-stack predicate only) | arch (integerLeFormFeasible tri-state, Rule 22), test (unit_poly parity-tight marking + guard predicate execution) |
| 165 | Native artifacts (C++ / x86-64 asm) mirror the buffer executor exactly; assembled artifacts are verified bit-exact against the walker (no silent semantic drift in the backend) | arch (polyhedral_spec §backend, kernel_abi form 3), test (unit_poly asm_backend_* three-way bit-exact suites) |
| 166 | No in-process machine codegen: artifacts are emitted as text and built/loaded out-of-process (file-backed mapping); parallel marks recorded per Rule 148; unsupported nodes fail emission honestly | arch (ADR-0003/0006, backend_driver stage errors), test (backend_rejects_lowered_first_nodes + recorded-marks assertions) |
| 167 | Reads of a reduction chain's location depend on the WHOLE chain (chain-final reads are not per-iteration flow); unschedulable kernels fall back gracefully, never break the pipeline | arch (polyhedral_spec §soundness/§synthesis, ADR-0005 decision 10), test (unit_poly softmax_synth_pipeline_bitexact, max_accumulate_store_semantics) |
| 169 | Native temp ABI: isTemp buffers are bindable (ptr, dims) table entries materialized zero-init by the driver with the walker's element-count model and limit; Max-accumulate stores emit the exact walker select ((v > cur) ? v : cur — NaN never replaces, ±0 ties keep the slot) in BOTH emitters; the softmax class runs three-way bit-exact incl. adversarial NaN/±0 rows | arch (polyhedral_spec §backend native temp ABI, ADR-0006), test (unit_poly softmax_native_three_way_bitexact, max_accumulate_store_native_bitexact) |
| 170 | Pass-registry resolution is by TEXT (Rule 16): raw SymbolIds are table-scoped and never compared across SymbolTables; the process-wide registry outlives any embedding's table, so id matching could (and did) dispatch the wrong pass | arch (pass_registry.h resolution contract), test (unit_poly native-artifact suites exercise byName under 60 tables of accumulated cross-test registrations; softmax_native_three_way_bitexact fails under raw-id matching) |
| 168 | No kernel-speed claim without a certificate: budgeted fast-kernel search with per-candidate identity/runtime/compile-time/launch certificates, declared search space + structured rejections, honest claims (never "fastest possible" without exhaustive certification), compile-time budget guard (same/extra/amortized modes, extra comptime only with certified payoff), roofline lower bound from a declared essential-work model, and fingerprint-validated artifact cache | arch (polyhedral_spec §fast-kernel-search, ADR-0007), test (unit_fastkernel: 9 cases incl. budget-failure report, refused extra comptime, cache invalidation) |
| 171 | CUDA artifacts mirror the buffer executor semantics with a DECLARED exactness policy: device-exact ops (IEEE mul/add/div/sqrt + poly7) gate bit-exact under mandatory --fmad=false; device-libm transcendental modules are ULP-bounded with the max distance measured and recorded, never silently relaxed; parallel marks proven by the scheduler are the ONLY source of grid parallelism (Rule 148); unsupported shapes fail emission honestly | arch (polyhedral_spec #GPU-backend, ADR-0008, cuda_emitter.h), test (unit_poly cuda_emission_* suites: structure, determinism, policy boundary, recorded-probe honest skip, Call rejection) |
| 172 | GPU artifact publication follows the same out-of-process model as CPU artifacts (emit text -> external nvcc -> dlopen): the arch flag is declared and validated (never silently defaulted), buildKernelArtifact(kind=Cuda) rejects explicitly in favor of the GPU entry points, and the toolchain probe is the recorded check that lets tests/bench skip honestly on CUDA-less machines | arch (backend_driver.h GpuBackendDriverConfig contract, ADR-0008), test (cuda_driver_honest_skip: arch validation, explicit-selection rejection, probe-gated skip/live four-way) |
| 173 | CUDA launch geometry is multi-axis and NEVER clamps: every device kernel computes its flat thread id from the full hardware linearization (x fastest over the six axes); a generated host-side greedy over the runtime padded trips assigns consecutive level runs to block axes (cumulative <= 1024, caps 1024/1024/64) then grid axes (gx 2^31-1, gy/gz 65535 overflow valves) with a 1-D flat fallback; multi-axis launches satisfy threads == total exactly (bijection onto the padded instance space) and the geometry is semantically inert because the device div/mod chain reads flat directly | arch (polyhedral_spec #GPU-backend round 20, ADR-0008 decision 7, cuda_emitter.h), test (cuda_emission_geometry_helper_behavioral: extracted generated text compiled with cc — hand-computed assignments, caps envelope, threads==total soundness, exhaustive flat injectivity; cuda_emission_multiaxis_text + cuda_emission_multiaxis_tiled_gemm_text: helper/dim3/trips structure + determinism) |
| 174 | Fast-kernel search declares its space in full: the Cuda execution path is a first-class variant (arch DECLARED per search config and bound into the artifact-cache fingerprint), and toolchain/device absence yields structured CapabilityUnsupported/BuildFailed rejections naming the failed probe — the winner claim downgrades honestly instead of hiding the uncertified candidates | arch (fast_kernel.h ExecPath::NativeCuda + FastKernelSearchConfig.gpu, ADR-0008 decision 8), test (unit_fastkernel cuda_candidate_declared_space: structured rejection + honest no-winner claim without nvcc, admissible bit-exact path with toolchain; live autotune --exec=cuda reports the rejection rows) |

## Slop checklist (Rule 84) — verified for this tree

- [x] No unnamed numeric constants in logic (core/constants.h)
- [x] No duplicated code blocks (declarative tables; shared emitters)
- [x] No silent fallbacks or unsafe default returns (telemetry + Result)
- [x] No prohibited containers in hot paths (open addressing, SBO)
- [x] All invariants documented and validated (verifier)
- [x] No premature abstractions without two consumers
- [x] No untracked workarounds or HACK comments
- [x] No target-specific logic outside backends (profiles + tables)
- [x] No domain-specific logic outside profiles (capability gates)
- [x] No algebraic rewrite without legality conditions
- [x] No approximation without accuracy contract
- [x] All new code paths have test coverage (15 suites)
- [x] Every new guard has a GraphState attachment
- [x] Every speculative node carries complete metadata
- [x] Every fallback point reachable with metadata
- [x] No raw buffer pointers across safepoints untracked
- [x] No getenv/mutex-locking in dispatch loops
- [x] No atomic RMW in per-instruction hot paths
- [x] W^X maintained (no emitter-owned executable pages; native artifacts built out-of-process — ADR-0003/0006)
- [x] Code/kernel publication atomic with release semantics
