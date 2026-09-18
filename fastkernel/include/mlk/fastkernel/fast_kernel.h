// MLK+ fast-kernel search — OEIA/Refined/FastKernels (spec upload,
// sections XIV-XVIII; docs/polyhedral_spec.md §fast-kernel-search).
//
// Kernel runtime speed becomes a first-class, CERTIFIED optimization
// objective under a hard compile-time budget:
//
//   - Every candidate kernel variant k carries a certificate bundle
//     (Meta-Axiom 0.1: no claim without certificate): identity
//     (k ≡_π s, bit-exact differential — Axiom 14.2), a benchmarked
//     runtime upper bound U^run (Axiom 14.4), a stage-accounted
//     compile-time cost M^kernel (Axioms 14.5/15.2), and a launch
//     coverage certificate (Axioms 14.13/14.14).
//   - The searched variant space is DECLARED and reported in full
//     (Axiom 14.20): every candidate ends admissible or rejected with a
//     structured reason (budget / identity / justification / resource /
//     capability).
//   - Budget modes (Axiom 15.1): same-comptime (B_allowed = B_K^0),
//     bounded-extra (B_K^0 + ΔB_K ≤ ΔB_max, admissible only with a
//     justification certificate — Axiom 14.9), amortized (objective
//     C^run + M^kernel / N). Budget regression is a rejection, never a
//     silent overrun (Axiom 15.6, Rule 17.5).
//   - A roofline-style runtime lower bound L^roof = max(B_min/BW_h,
//     O_min/T_h) is derived from the essential-work model and the
//     measured environment peaks (Axioms 14.11/14.12); the certified
//     gap U^run / L^run − 1 is reported, never hidden.
//   - The winner claim is honest (Axiom 14.21): "fastest certified
//     kernel within the declared searched space" only when every
//     declared candidate was evaluated; otherwise "best certified
//     kernel in searched space"; never "fastest possible".
//   - Compiled artifacts are cached under a completeness fingerprint
//     (source hash, backend flags, policy, environment — Axiom 14.22);
//     reuse reports M^reuse, misses rebuild (Axiom 15.8).
//
// Determinism (Rule 53): enumeration order, tie-breaks (strict <,
// declared order wins), and every generated artifact are deterministic;
// measured times are recorded as benchmark observations, not derived
// values. No exceptions anywhere (Rule 6); Result<T> end to end.
#pragma once

#include <cstdint>
#include <string>

#include "mlk/backend/backend_driver.h"
#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/support/json.h"
#include "mlk/support/kernel_ir.h"
#include "mlk/type/domain_profile.h"

namespace mlk::fastkernel {

/// Cache-entry format version (Axiom 14.22 / Rule 37: versioned keys).
inline constexpr uint32_t kFastKernelCacheVersion = 1;

/// The policy π of this search (Axioms 0.2/1.4): exact f64 kernel
/// identity — no reassociation, no FMA contraction, no approximate
/// reduction is permitted, so every candidate must reproduce the
/// reference element-for-element bit-exactly (Axiom 14.17: numerical
/// relaxations are ILLEGAL under this policy; a candidate that differs
/// is rejected, not tolerated within a tolerance).
inline constexpr const char* kFastKernelPolicy = "bitexact-f64-cpu-v1";

// --- Environment h (Axiom 0.3: cost is environment-relative) -------------

/// The environment model h: measured peak bandwidth and computational
/// throughput (benchmark certificates — the raw measurement method is
/// declared in measureEnvironment), plus the hardware thread budget.
struct EnvironmentModel {
    double bandwidthBytesPerSec{0.0};  // BW_h (stream-triad measurement)
    double throughputOpsPerSec{0.0};   // T_h (independent mul-add chains)
    uint32_t hardwareThreads{1};       // std::thread::hardware_concurrency
    /// Descriptor recorded into every certificate and the artifact-cache
    /// fingerprint (Axiom 14.22: an environment change invalidates the
    /// cache entry).
    std::string descriptor{};
};

/// Measures the environment peaks once per search (Axiom 14.12 requires
/// certified environment bounds; the measurement IS the certificate —
/// stream triad for BW_h, four independent multiply-add chains for T_h,
/// best-of-reps, volatile sinks against dead-code elimination).
[[nodiscard]] Result<EnvironmentModel> measureEnvironment();

// --- Essential-work model (Axioms 14.11/14.12) ---------------------------

/// The specification's essential work: the minimum memory movement and
/// the minimum operation count ANY implementation of this kernel must
/// perform. These feed the roofline lower bound. The counting rules are
/// DECLARED (and reported):
///   - B_min: each non-temp buffer moved once (f64, 8 B/element) —
///     temps are assumed cache-resident (declared assumption);
///   - O_min: Call(MatMul) = 2*M*K*N, Call(ReduceSum) = M*K,
///     Call(Softmax) = 5*M*K (max/sub/exp/add/div row model), every
///     fused Compute expression = 1, every accumulate store = 1.
/// Unknown buffer extents (dynamic bounds without dims) mark the model
/// UNKNOWN — the roofline certificate is then omitted honestly
/// (tri-state, Rule 22), never guessed.
struct EssentialWork {
    int64_t minBytes{0};  // B_min
    int64_t minOps{0};    // O_min
    bool bytesKnown{false};
    bool opsKnown{false};
};

/// Computes the essential-work model of a kernel module.
[[nodiscard]] EssentialWork essentialWorkModel(const KernelModule& kernel);

/// Roofline lower bound L^roof = max(B_min / BW_h, O_min / T_h)
/// (Axiom 14.12). Returns 0.0 when the model or the environment bounds
/// are unknown (no certificate, no claim).
[[nodiscard]] double rooflineLowerBound(const EssentialWork& work,
                                        const EnvironmentModel& env);

// --- Launch configuration (Axioms 14.13/14.14) ---------------------------

/// One worker's work slab [begin, endInclusive] of a parallel loop.
struct LaunchSlab {
    int64_t begin{0};
    int64_t endInclusive{-1};
};

/// The launch-coverage certificate for one kernel execution: the
/// parallel-marked loops' worker slabs must cover the iteration domain
/// EXACTLY (union == domain) and pairwise disjointly (Axiom 14.14 —
/// no dropped, duplicated, or out-of-bounds work). The slab arithmetic
/// mirrors the buffer executor's decideThreads + chunk decomposition
/// (same named constants from constants.h), so a passing certificate
/// proves the decomposition the executor will actually run.
struct LaunchCoverage {
    bool covered{false};  // union == domain && pairwise disjoint
    uint32_t threads{1};  // workers the executor will spawn
    SmallVector<LaunchSlab, 8> slabs{};  // first certified parallel loop
};

/// Certifies the launch configuration of `kernel` for a root-domain of
/// `domainElements` elements. A `threads` schedule param > the executor
/// cap is a resource-limit violation (Axiom 14.16) and reports
/// covered = false.
[[nodiscard]] LaunchCoverage certifyLaunchCoverage(
    const KernelModule& kernel, SymbolTable& symbols,
    int64_t domainElements);

// --- Kernel variants (Axioms 14.1/14.3) ----------------------------------

/// Execution path of a variant (part of the launch configuration,
/// Axiom 14.13): the buffer executor (walker) or a native artifact
/// built out-of-process from the emitted assembly / C++ source / CUDA
/// source (the GPU artifact builds through the dedicated GPU driver
/// entry points — the arch flag is DECLARED, never defaulted; spec
/// #GPU-backend). A Cuda candidate without an nvcc toolchain is a
/// structured CapabilityUnsupported rejection naming the probe (the
/// declared space is reported in full — Axiom 14.20), never a silent
/// omission.
enum class ExecPath : uint8_t {
    Walker = 0,
    NativeAsm = 1,
    NativeCpp = 2,
    NativeCuda = 3,
};

[[nodiscard]] const char* execPathName(ExecPath p) noexcept;

/// A kernel variant configuration: the certified transformation axes of
/// the searched space (Axiom 14.3 — tiling, launch config, backend).
///   tileSize: poly.tile band edge (0 = untiled — poly.tile disabled via
///             its kill switch); threads: walker worker override
///             (0 = executor default); exec: execution path.
struct VariantConfig {
    int64_t tileSize{0};
    ExecPath exec{ExecPath::Walker};
    uint32_t threads{0};
};

// --- Certificates (Meta-Axiom 0.1) ---------------------------------------

/// Per-candidate certificate bundle. Every field is a checkable claim;
/// the report carries them all (Axiom 14.20).
struct VariantCertificate {
    // Identity (Axiom 14.2): k ≡_π s — bit-exact differential against
    // the Tier1 reference execution over the same inputs.
    bool identityCertified{false};
    double identityMaxAbsDiff{0.0};
    // Runtime upper bound (Axiom 14.4): benchmark certificate, declared
    // method = median of benchReps timed runs after warmupReps warmups.
    bool runtimeCertified{false};
    double runtimeUpperBoundSec{0.0};  // U^run
    double runtimeMinSec{0.0};
    double runtimeMaxSec{0.0};
    uint32_t benchReps{0};
    uint32_t warmupReps{0};
    // Compile-time stage accounting (Axioms 14.5/15.2): parse+lower+pass
    // pipeline, artifact emission, out-of-process build, load.
    double compileSec{0.0};
    double emitSec{0.0};
    double buildSec{0.0};
    double loadSec{0.0};
    double comptimeTotalSec{0.0};  // M^kernel = sum of the stages
    bool comptimeFromCache{false}; // M^reuse (Axiom 15.8)
    // Launch (Axioms 14.13/14.14).
    LaunchCoverage launch{};
    // Artifact fingerprint (Axiom 14.22); empty for walker variants.
    std::string fingerprint{};
    /// Native artifact library path (cache address); empty for walker.
    std::string libraryPath{};
};

// --- Budget contract (Axioms 14.6-14.9, 15.1, 15.6, 15.7) ----------------

/// Compile-time budget mode (Axiom 15.1).
enum class BudgetMode : uint8_t {
    SameComptime = 0,  // B_allowed = B_K^0 (Rule 17.1)
    BoundedExtra = 1,  // B_allowed = B_K^0 + ΔB_K ≤ ΔB_max (Rule 17.2)
    Amortized = 2,     // objective C^run + M^kernel / N (Axiom 15.1)
};

[[nodiscard]] const char* budgetModeName(BudgetMode m) noexcept;

/// The budget contract. baselineComptimeSec == 0 means AUTO: B_K^0 is
/// the measured compile time of the default Tier2 pipeline (the kernel
/// a non-searching compile would produce) — recorded with its
/// provenance in the report.
struct BudgetContract {
    BudgetMode mode{BudgetMode::SameComptime};
    double baselineComptimeSec{0.0};      // B_K^0 (0 = auto-measured)
    double extraComptimeMaxSec{0.0};      // ΔB_max
    double minRuntimeImprovementSec{0.0}; // τ (Axiom 14.9; 0 = any gain)
    double alpha{1.0};                    // comptime weight (amortization)
    int64_t amortizedExecutions{1};       // N
};

// --- Extra-comptime justification (Axioms 14.9/15.7) ---------------------

/// The justification decision for spending extra compile time: one of
///   - absolute runtime improvement: U_new <= U_base - tau;
///   - amortized payoff: alpha * dM <= N * (U_base - U_new);
///   - Pareto improvement: U_base - U_new > 0 (every other budgeted
///     dimension within bound — compile time is the excused dimension).
struct JustificationDecision {
    bool justified{false};
    double marginSec{0.0};  // certified payoff margin
    const char* form{""};
};

/// Pure decision function (unit-testable; the search applies it to
/// certified quantities only).
[[nodiscard]] JustificationDecision evaluateExtraComptimeJustification(
    double uNew, double uBase, double deltaM, double tau, double alpha,
    int64_t executions) noexcept;

// --- Search space (Axiom 14.20) ------------------------------------------

enum class RejectReason : uint8_t {
    None = 0,
    IdentityMismatch,        // failed the bit-exact differential (14.2)
    ComptimeBudgetExceeded,  // M^kernel > B_allowed (15.6, Rule 17.5)
    NoJustification,         // extra comptime without payoff (14.9)
    BuildFailed,             // emit/build/load failure (recorded)
    ResourceLimit,           // e.g. threads > executor cap (14.16)
    CapabilityUnsupported,   // e.g. temps in native form (honest boundary)
};

[[nodiscard]] const char* rejectReasonName(RejectReason r) noexcept;

/// One candidate's full outcome: configuration, certificate (or
/// rejection), and its optimization objective.
struct VariantOutcome {
    VariantConfig config{};
    bool admissible{false};
    RejectReason reject{RejectReason::None};
    std::string rejectDetail{};
    VariantCertificate cert{};
    /// The compiled variant (the certified kernel body — Axiom 14.1);
    /// empty when compilation or the build was rejected/failed.
    KernelModule kernel{};
    /// Optimization objective: U^run (same/extra comptime) or
    /// U^run + M^kernel/N (amortized).
    double objectiveSec{0.0};
    bool usedExtraComptime{false};  // M^kernel > B_K^0
    /// Payoff margin of the justification certificate (seconds of
    /// runtime gain, or its amortized equivalent).
    double justificationMarginSec{0.0};
};

/// Search configuration: the DECLARED variant space (Axiom 14.3) plus
/// the benchmark and budget policy.
struct FastKernelSearchConfig {
    /// Tile sizes to synthesize; 0 = the untiled kernel (poly.tile kill
    /// switch). Deduplicated, declaration order preserved.
    SmallVector<int64_t, 8> tileSizes{0, 16, 32};
    /// Execution paths to certify. The Cuda path builds through the
    /// GPU driver (nvcc out-of-process); on a machine without the
    /// toolchain its candidates end in a structured rejection that
    /// names the failed probe (recorded, never silent — Rule 148).
    SmallVector<ExecPath, 4> execPaths{ExecPath::Walker,
                                       ExecPath::NativeAsm,
                                       ExecPath::NativeCpp,
                                       ExecPath::NativeCuda};
    /// Walker thread-count overrides; 0 = executor default. Values above
    /// the executor cap are part of the declared space and get REJECTED
    /// with ResourceLimit (Axioms 14.16/14.20 — rejections are reported,
    /// not hidden). Native paths ignore thread overrides (sequential).
    SmallVector<uint32_t, 8> threadCounts{0};
    uint32_t warmupReps{2};
    uint32_t benchReps{5};
    BudgetContract budget{};
    BackendDriverConfig driver{};
    /// GPU driver knobs for the Cuda execution path (compiler + the
    /// DECLARED arch flag — the arch is fingerprint material, so two
    /// arch declarations never share a cache entry).
    GpuBackendDriverConfig gpu{};
    /// Artifact cache directory; empty disables artifact caching
    /// (Axiom 14.22). Cached entries are keyed by a completeness
    /// fingerprint (kernel hash, backend, compiler, policy, environment).
    std::string cacheDir{};
    /// Measure the environment (true) or trust config.env.
    bool measureEnvironment{true};
    EnvironmentModel env{};
};

// --- Report (Axioms 14.20/14.21, section XVI contract) -------------------

/// The full search report: declared space, every outcome with its
/// certificate bundle or structured rejection, the roofline lower bound
/// and certified gap, the budget accounting, and the honest claim.
struct FastKernelReport {
    EnvironmentModel env{};
    EssentialWork work{};
    double rooflineLowerSec{0.0};       // L^roof (0 = no certificate)
    double winnerGapOverLower{0.0};     // U^run / L^run − 1 (> 0)
    SmallVector<VariantOutcome, 16> candidates{};
    int64_t winnerIndex{-1};
    bool budgetFailure{false};          // Axiom 15.6 report path
    std::string claim{};                // honest claim string (14.21)
    std::string budgetProvenance{};     // auto-measured vs policy-set B_K^0
    double baselineComptimeSec{0.0};    // effective B_K^0
    double baselineRuntimeSec{0.0};     // the specification's own U^run
    uint32_t cacheHits{0};
    uint32_t cacheMisses{0};

    [[nodiscard]] json::Value toJson() const;
};

/// Runs the budgeted fast-kernel search over the declared variant space.
///
/// Pipeline (per candidate): Tier2 compile with the variant's tile knob
/// (stage-accounted) → budget gate (15.6) → native build via the
/// out-of-process driver (or the artifact cache) → bit-exact identity
/// differential against the Tier1 reference (14.2) → benchmarked U^run
/// (14.4) → launch coverage certificate (14.14) → justification gate for
/// extra-comptime candidates (14.9). The winner minimizes the objective
/// among admissible candidates (14.10); the report states the claim
/// honestly (14.21).
[[nodiscard]] Result<FastKernelReport> runFastKernelSearch(
    MathGraph& graph, SymbolTable& symbols,
    const MathDomainProfile& profile, const FastKernelSearchConfig& config);

}  // namespace mlk::fastkernel
