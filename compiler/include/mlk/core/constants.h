// MLK+ named constants.
//
// Rule 27: magic numbers are forbidden. Every threshold, budget, limit, tile
// heuristic, search budget, benchmark repetition count, and cost-model
// constant used in any optimization or tuning pass is defined here (or in a
// nearby dedicated constants block) as a named, documented constexpr.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace mlk::constants {

// --- IR sizes --------------------------------------------------------------
/// Rule 15: graph references are 32-bit indices.
inline constexpr uint32_t kInvalidId = 0xFFFFFFFFu;
/// Rule 19: typical inline capacity for operand/user lists.
inline constexpr std::size_t kOperandInline = 4;
inline constexpr std::size_t kUsersInline = 4;
inline constexpr std::size_t kShapeInline = 4;
inline constexpr std::size_t kFactsInline = 4;

/// Graph growth: initial capacities for the value/node storage vectors.
inline constexpr std::size_t kInitialValueCapacity = 64;
inline constexpr std::size_t kInitialNodeCapacity = 64;

/// Maximum rank of a shape supported by the IR core.
inline constexpr std::size_t kMaxRank = 8;
/// Maximum representative-chain depth in the equivalence map (defensive,
/// acyclic by construction).
inline constexpr uint32_t kMaxEquivalenceDepth = 64;

// --- E-graph (Rules 10 and 54: guarded fixpoint with strict budget) --------
inline constexpr std::size_t kEgraphDefaultMaxNodes = 8192;
inline constexpr uint32_t kEgraphDefaultMaxIterations = 24;
inline constexpr std::size_t kEgraphMaxClassesPerNode = 64;

// --- Pass budgets (Rule 10, Rule 131) ---------------------------------------
inline constexpr uint32_t kMaxPassFixpointIterations = 16;
inline constexpr int64_t kDefaultPassEditBudget = 1 << 20;  // 1M node edits
inline constexpr uint32_t kTier1MaxNodes = 1u << 22;        // 4M nodes
inline constexpr uint32_t kTier2MaxNodes = 1u << 23;        // 8M nodes

// --- Workload buckets (analysis.workload.bucket) ----------------------------
/// Bucket edges for dynamic dimensions: [1..32][33..128][129..512][513..2048]+
inline constexpr std::array<int64_t, 4> kWorkloadBucketEdges{32, 128, 512, 2048};

// --- Benchmark protocol (Rule 49: statistically valid measurement) ----------
inline constexpr uint32_t kBenchDefaultWarmup = 5;
inline constexpr uint32_t kBenchDefaultReps = 30;
inline constexpr double kBenchNoiseRelativeStddev = 0.05;  // 5% noise floor
inline constexpr double kBenchMinRepresentativeRuntimeMs = 0.02;
inline constexpr uint32_t kBenchRecheckTopK = 3;

// --- Autotuner budgets (Rule 54: declarative, budgeted search) --------------
inline constexpr uint32_t kTuneDefaultMaxCandidates = 64;
inline constexpr uint32_t kTuneDefaultLocalSearchNeighbors = 8;

// --- Numerics / accuracy defaults (Rule 34) ---------------------------------
inline constexpr double kDefaultMaxUlps = 4.0;
inline constexpr double kDefaultMaxAbsError = 1e-6;
inline constexpr double kDefaultMaxRelError = 1e-6;

// --- Approximation (Rule 34: no approximation without contract) -------------
inline constexpr std::size_t kPolyMaxDegree = 15;
/// Number of sample points for ULP verification of math-function candidates.
inline constexpr std::size_t kUlpVerifySamples = 512;
inline constexpr std::size_t kUlpVerifyEdgeSamples = 64;

// --- Hardware defaults (Rule 31: no assumption of stable hardware; these
//     are conservative lower bounds, probed/overridable at runtime) ----------
inline constexpr std::size_t kDefaultCacheLineBytes = 64;
inline constexpr std::size_t kDefaultSimdWidthF32 = 8;   // AVX2-era floor
inline constexpr double kDefaultPeakFlopsPerCorePerCycleF64 = 4.0;
inline constexpr double kDefaultPeakFlopsPerCorePerCycleF32 = 8.0;
inline constexpr double kDefaultMemoryBandwidthGiBs = 12.0;
inline constexpr std::size_t kDefaultAlignmentBytes = 64;

// --- Tensor tiling heuristics (Rule 29: empirically validated defaults;
//     each is overridable via the declarative search space) ------------------
inline constexpr int64_t kTileMinSize = 16;
inline constexpr int64_t kTileMaxSize = 512;
inline constexpr int64_t kDefaultTileM = 64;
inline constexpr int64_t kDefaultTileN = 64;
inline constexpr int64_t kDefaultTileK = 32;
inline constexpr std::array<int64_t, 4> kTileChoices{32, 64, 128, 256};
inline constexpr std::array<int64_t, 3> kUnrollChoices{1, 2, 4};
inline constexpr std::array<int64_t, 3> kVectorWidthChoices{4, 8, 16};

// --- Kernel/KernelIR --------------------------------------------------------
inline constexpr int64_t kKernelLoopDynamicBound = -1;
inline constexpr std::size_t kKernelMaxFusionDepth = 32;
/// Upper bound on ONE temp buffer's element count (DoS guard for
/// executor-allocated scratch; 16M doubles = 128 MiB). The buffer walker
/// and the native-artifact driver BOTH materialize isTemp scratch and
/// must read the SAME limit (the artifact is the compiled form of the
/// same contract — docs/polyhedral_spec.md #backend).
inline constexpr int64_t kKernelTempElementsLimit = 1 << 24;

// --- Polyhedral engine (mlk_poly; see docs/polyhedral_spec.md) ---------------
/// Maximum loop dimensions per SCoP statement (Rule 10: bounded passes).
inline constexpr uint32_t kPolyMaxDims = 8;
/// Maximum statements per SCoP.
inline constexpr uint32_t kPolyMaxStatements = 8;
/// Maximum memory accesses per statement.
inline constexpr uint32_t kPolyMaxAccessesPerStatement = 8;
/// Schedule rows synthesized by the Pluto-style scheduler per SCoP.
inline constexpr uint32_t kPolyMaxScheduleRows = 12;
/// |coefficient| ceiling for schedule matrices (keeps LP exact in int64).
inline constexpr int64_t kPolyMaxScheduleCoeff = 1024;
/// Default rectangular tile edge applied by poly.tile (Rule 29 heuristic,
/// overridable via the poly_tile_size knob; tuned by the autotuner).
inline constexpr int64_t kPolyDefaultTileSize = 32;
/// Default SIMD width recorded by poly.schedule for innermost vector dims.
inline constexpr int64_t kPolyDefaultVectorWidth = 8;
/// Maximum inner-tree duplication across tiled levels in poly.codegen.
inline constexpr int64_t kPolyMaxCodegenCopies = 16;
/// Pivot-order search: exhaustive permutation enumeration below this depth
/// (4! = 24 orders), greedy prefix extension at deeper SCoPs.
inline constexpr uint32_t kPolyMaxOrderEnumerateDepth = 4;
/// Global LP-solve budget for one poly.schedule invocation across all
/// enumerated orders (Rule 10: bounded passes; tripping degrades the
/// search to the identity order instead of failing the pass).
inline constexpr uint32_t kPolyMaxSchedulerLps = 4096;
/// Per-order LP budget inside the order search (one candidate order's
/// synthesis; tripping rejects that order, never the schedule).
inline constexpr uint32_t kPolyMaxOrderLps = 512;

// --- Runtime / telemetry (Rule 130, Rule 157) -------------------------------
inline constexpr uint32_t kSafepointPollIntervalIterations = 4096;
inline constexpr std::size_t kTelemetryRingCapacity = 1024;
/// Rule 103: repeated fallback at the same site is throttled past this count.
inline constexpr uint32_t kFallbackThrottleThreshold = 8;
/// Hard cap on worker threads per parallel-marked loop (the buffer
/// executor and the fast-kernel launch-coverage certificate must agree
/// on this — docs/polyhedral_spec.md §fast-kernel-search).
inline constexpr std::size_t kKernelExecMaxThreads = 4;
/// Minimum loop trip count before a parallel-marked loop is threaded at
/// all (spawn amortization; shared by the executor and the launch
/// certificate, Axiom 14.14).
inline constexpr int64_t kParallelChunkElements = 16384;

// --- Realization cache (Rule 57: complete keys; Rule 37: versioning) --------
inline constexpr uint32_t kRealizationCacheFormatVersion = 1;
inline constexpr uint32_t kGraphFileFormatVersion = 1;
inline constexpr uint32_t kBytecodeFormatVersion = 1;

// --- Symbol table ------------------------------------------------------------
inline constexpr std::size_t kSymbolInitialBuckets = 1024;

// --- Misc ---------------------------------------------------------------------
/// Rule 27 compliance sentinel: the largest bare integer allowed in pass
/// logic without a named constant (used by scripts/lint.sh guidance).
inline constexpr int kMaxBareLiteral = 2;

}  // namespace mlk::constants
