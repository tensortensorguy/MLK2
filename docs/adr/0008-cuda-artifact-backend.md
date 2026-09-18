# ADR-0008: CUDA artifact backend (GPU phase)

Date: 2026-09-18 (round 19)
Status: Accepted
Amends: ADR-0006 (native assembly backend) — the GPU path reuses its
publication model and adds a third artifact form plus dedicated GPU
driver entry points.

## Context

The user directive "then when ready, we start on GPU" follows the
benchmark round: the pipeline (synth → SCoP → dependences → Pluto
schedule → tiling → CLAST codegen) now ends in verified native
artifacts, and the differential harness (walker vs artifact vs Tier1)
is the established oracle. The GPU phase per docs/polyhedral_spec.md
§GPU-backend must reach device execution with NO structural change to
the pipeline: the polyhedral layer owns the schedule, the parallel
marks are proven, and the only sanctioned publication model is "emit
text → external toolchain → load → invoke through the buffer ABI →
differential vs the walker" (ADR-0003/0006 — never in-process machine
codegen, Rules 116/118 vacuously satisfied).

The hard constraints, declared up front in the spec:

- Bit-exactness: per-cell k-chains in the walker's ascending order, no
  cross-thread reduction, no reassociation, `--fmad=false` mandatory.
  Device libm transcendentals are NOT guaranteed bit-identical to host
  libm — the differential for such modules is re-declared as
  ULP-bounded with the bound measured and recorded, never silently
  relaxed (Rule 90).
- Thread mapping from PROVEN marks only (Rule 148): parallel rows map
  to grid threads; serial carried dims stay serial inside the thread.
- Honesty without a device: the development environment has no nvcc
  and no GPU, so everything machine-checkable must be pinned in-network
  and every compile/run path gated by a recorded probe.

## Decision

1. **Artifact form**: standalone CUDA C++ (`.cu` text) — one
   `__global__` device kernel per forest ROOT, all sharing the FULL
   bindable table signature + scalars pair + an `int* mlk_status` slot;
   a host wrapper `extern "C" int mlk_kernel(...)` keeps the EXACT
   multi-form ABI and manages device memory internally
   (cudaMalloc/H2D/launch/D2H/cudaFree behind the same Result-mapped
   error codes: 1 negative flat, 2 runtime, 3 launch, 4 no device).
   The caller's view of a loaded kernel is unchanged from the CPU
   artifacts — the driver dispatch is form-based, not backend-based.
2. **Padded-grid collapse**: the outermost chain of parallel-marked
   unit-step loops (interior bodies = exactly one loop child) peels
   into a flat 1-D grid, row-major decomposition. Point loops with
   affine bounds over the prefix (tile → point pairs) ARE peelable via
   PADDED trip counts (exact interval max of each level's trip over the
   prefix box) with per-thread actual-range checks skipping the padding
   holes — sound because the parallel marks prove slab-disjointness at
   every collapsed level, and exact because padded trips upper-bound
   every prefix tuple's real trip. Everything deeper (serial/carried/
   guards/split segments) walks inside the thread in the walker's
   order. Roots without a parallel chain launch single-thread and the
   header records it.
3. **Exactness policy as data**: cudaArtifactBitExactPolicy() classifies
   each module with the same scan the emitter performs — device-exact
   for {Add, Sub, Mul, Div, Neg, Sqrt, Rsqrt, poly7-Sin} (bit-exact
   differential by construction), ULP-bounded for device-libm
   transcendentals (differential measured as max ordered-bit ULP
   distance and recorded). Drivers/tests/bench consume the flag to pick
   the honest gate; NaN/inf constants emit as bit-exact
   __longlong_as_double forms.
4. **Driver**: dedicated GPU entry points (buildGpuKernelArtifact /
   buildGpuKernelArtifactInDir) with a DECLARED arch (validated
   sm_<digits>; never defaulted silently) and the mandatory
   `--fmad=false` flag; buildKernelArtifact(kind=Cuda) rejects
   explicitly so the arch can never be accidentally defaulted.
   cudaToolchainAvailable() is the recorded PATH probe; device presence
   is discovered at run time through the artifact's ABI code 4.
5. **Temp ABI**: temp table entries remain in the host signature for
   dispatch-arity uniformity; the artifact materializes its own device
   scratch from the module's static temp dims (walker model,
   kKernelTempElementsLimit enforced at emission) and the driver passes
   dummies for gpu_ kernels. Documented boundary, not a silent drop.
6. **Bench**: the `cuda` path joins the Rule 49 protocol — honest
   skip rows on CUDA-less machines, bit-exact gating for device-exact
   modules, measured max-ULP recorded for ULP-bounded modules.
7. **Multi-axis launch geometry (round 20, never clamps)**: every
   device kernel computes its flat thread id from the FULL hardware
   linearization (x fastest over the six axes); the launch geometry is
   chosen at RUN TIME by a GENERATED host-side greedy over the padded
   trips (`mlk_assign_geometry`, plain C, emitted once per module) —
   innermost levels pack the block axes (cumulative product ≤ 1024,
   per-axis caps 1024/1024/64), the rest pack the grid axes innermost-
   first (gx 2^31-1, then gy/gz 65535 as overflow valves), and any
   shape that fits no axis keeps the 1-D flat fallback. The helper
   never clamps: multi-axis launches satisfy threads == total exactly,
   so the hardware thread space bijectively covers the padded instance
   space; soundness is assignment-invariant because the device's
   div/mod chain reads flat directly (row-major over level trips). The
   generated helper is behaviorally verified by extracting its text
   and compiling it with cc (cuda_emission_geometry_helper_behavioral).
8. **Fast-kernel GPU candidates (round 20)**: ExecPath::NativeCuda
   joins the DECLARED search space; Cuda candidates build through the
   GPU entry points with the arch DECLARED per search config (the
   artifact-cache fingerprint binds the declared GPU compiler + arch),
   and toolchain/device absence yields structured rejections naming
   the failed probe — the declared space is reported in full and the
   winner claim downgrades honestly (Axiom 14.20/14.21).
9. **Shared-memory slab reuse (round 21, OPT-IN)**: read-only buffers
   with dims-only value hulls are copied into dynamic shared memory
   per root and their reads redirected (value-identical; the hull is
   a value-keyed superset so per-segment split symbols fold without
   position collisions). The slab twin converts the padded-collapse
   early returns into a live flag — every thread reaches the
   cooperative load and the barrier (divergent __syncthreads is UB) —
   and the wrapper picks plain/slab per launch over runtime dims
   within a declared budget. The value logic is pinned BIT-EXACT
   through the C++ mirror artifact (same plan, heap scratch); the
   device prologue is structure-verified. No speedup is claimed
   without hardware (Rule 90); default-off keeps artifacts
   byte-identical.

## Consequences

- The compiler loop is closed for a third target with the same
  verification spine: MathGraph → polyhedral schedule → CLAST →
  `.cu` TEXT → external nvcc → dlopen → buffer-ABI execution →
  differential vs the walker (four-way where a device exists).
- On machines without CUDA (including this environment) the emission
  contract, determinism, policy boundary, and driver rejection paths
  are fully test-verified; the live compile/run path is probe-gated
  and activates automatically where nvcc + a device exist.
- Open (roadmap): block-geometry heuristics are untuned; chunked
  slabs for over-budget hulls (needs the segment-interleave
  restructure), a spread heuristic, stream orchestration, and PTX
  emission remain — each needs hardware evidence before being claimed
  as an optimization.
