// Fast-kernel search implementation (see fast_kernel.h for the contract;
// axiom references inline). All timing uses steady_clock; all failures
// are Result errors or structured rejections — nothing throws, nothing
// silently overruns a budget.
#include "mlk/fastkernel/fast_kernel.h"

#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "mlk/core/constants.h"
#include "mlk/core/hash_map.h"
#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/runtime/telemetry.h"

namespace mlk::fastkernel {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double nowSeconds() noexcept {
    return std::chrono::duration<double>(Clock::now().time_since_epoch())
        .count();
}

/// FNV-1a 64 (fingerprint material only — not a security primitive).
[[nodiscard]] uint64_t fnv1a64(const void* data, std::size_t len,
                               uint64_t seed) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = seed;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

[[nodiscard]] uint64_t fnv1a64String(const std::string& s,
                                     uint64_t seed) noexcept {
    return fnv1a64(s.data(), s.size(), seed);
}

[[nodiscard]] std::string fingerprintHex(uint64_t h) {
    static const char* kDigits = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = kDigits[h & 0xFULL];
        h >>= 4;
    }
    return out;
}

/// Product of buffer dims with a magnitude guard (exact int64; the
/// engine-wide magnitude ceiling keeps every intermediate in range).
[[nodiscard]] bool dimsProduct(const SmallVector<int64_t, 4>& dims,
                               int64_t& out) noexcept {
    int64_t acc = 1;
    for (const int64_t d : dims) {
        if (d < 0) return false;
        if (d != 0 && acc > (1LL << 40) / d) return false;  // overflow guard
        acc *= d;
    }
    out = acc;
    return true;
}

[[nodiscard]] double medianOf(SmallVector<double, 16> v) noexcept {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// --- Environment measurement (benchmark certificate, Axiom 14.12) -------

inline constexpr std::size_t kBandwidthProbeElements = 1u << 19;  // 4 MiB f64
inline constexpr std::size_t kThroughputProbeElements = 1u << 15;
inline constexpr uint32_t kEnvironmentProbeReps = 5;
/// Probe duration floor: a rep shorter than the clock's useful
/// resolution makes best-of-reps collapse to ~0 and the derived peak
/// absurd; the probes scale their inner repetitions to clear this.
inline constexpr double kProbeMinSeconds = 0.002;
inline constexpr uint32_t kProbeMaxInnerReps = 4096;
/// Body of the BW probe: d[i] = a[i] + b[i] moves 3 * 8 bytes/element —
/// the declared traffic model of the stream-triad certificate.
inline constexpr int64_t kStreamTriadBytesPerElement = 24;

struct ProbeResult {
    double bandwidthBytesPerSec{0.0};
    double throughputOpsPerSec{0.0};
};

/// Times one rep of `body` executed `innerReps` times. The FIRST
/// calibration rep doubles innerReps until the duration clears the
/// clock floor (kProbeMinSeconds) — a sub-resolution rep measures as ~0
/// and poisons best-of selection with an absurd peak.
template <typename Body>
[[nodiscard]] std::pair<double, uint32_t> timedRep(const Body& body,
                                                   uint32_t innerReps) {
    for (;;) {
        const double t0 = nowSeconds();
        for (uint32_t r = 0; r < innerReps; ++r) body();
        const double t1 = nowSeconds();
        const double dt = t1 - t0;
        if (dt >= kProbeMinSeconds || innerReps >= kProbeMaxInnerReps) {
            return {dt, innerReps};
        }
        innerReps *= 2;  // calibration: below the clock floor, scale up
    }
}

[[nodiscard]] ProbeResult runEnvironmentProbes() noexcept {
    ProbeResult out;

    // Bandwidth (stream triad): the declared traffic model is
    // kStreamTriadBytesPerElement bytes per element per inner pass
    // (2 reads + 1 write of f64). Each rep is timed over its calibrated
    // inner-pass count and the per-pass time is what competes in
    // best-of.
    {
        SmallVector<double, 8> a(kBandwidthProbeElements, 1.0);
        SmallVector<double, 8> b(kBandwidthProbeElements, 2.0);
        SmallVector<double, 8> d(kBandwidthProbeElements, 0.0);
        // Volatile sink: the measured body must actually run at every
        // optimization level, or the certificate certifies nothing.
        volatile double sink = 0.0;
        double bestPerPass = 0.0;
        for (uint32_t r = 0; r < kEnvironmentProbeReps; ++r) {
            const auto [dt, inner] = timedRep(
                [&] {
                    for (std::size_t i = 0; i < kBandwidthProbeElements;
                         ++i) {
                        d[i] = a[i] + b[i];
                    }
                    sink = d[0] + d[kBandwidthProbeElements / 2];
                },
                1);
            const double perPass = dt / static_cast<double>(inner);
            if (bestPerPass == 0.0 || perPass < bestPerPass) {
                bestPerPass = perPass;
            }
        }
        if (bestPerPass > 0.0) {
            out.bandwidthBytesPerSec =
                static_cast<double>(kBandwidthProbeElements) *
                static_cast<double>(kStreamTriadBytesPerElement) /
                bestPerPass;
        }
    }

    // Throughput (four INDEPENDENT multiply-add chains, 2 flops each per
    // iteration): chain independence lets the peak reflect issue
    // bandwidth rather than one dependency chain's latency. Same
    // calibrated per-inner-pass timing.
    {
        double x0 = 1.0, x1 = 2.0, x2 = 3.0, x3 = 4.0;
        volatile double sink = 0.0;
        double bestPerPass = 0.0;
        for (uint32_t r = 0; r < kEnvironmentProbeReps; ++r) {
            const auto [dt, inner] = timedRep(
                [&] {
                    for (std::size_t i = 0; i < kThroughputProbeElements;
                         ++i) {
                        x0 = x0 * 0.9999999 + 1e-9;
                        x1 = x1 * 0.9999998 + 2e-9;
                        x2 = x2 * 0.9999997 + 3e-9;
                        x3 = x3 * 0.9999996 + 4e-9;
                    }
                    sink = x0 + x1 + x2 + x3;
                },
                1);
            const double perPass = dt / static_cast<double>(inner);
            if (bestPerPass == 0.0 || perPass < bestPerPass) {
                bestPerPass = perPass;
            }
        }
        if (bestPerPass > 0.0) {
            out.throughputOpsPerSec =
                static_cast<double>(kThroughputProbeElements) * 8.0 /
                bestPerPass;
        }
    }
    return out;
}


// --- Essential-work op model (Axioms 14.11/14.12; declared rules) ------

inline constexpr int64_t kFlopsPerMulAddPair = 2;
/// Softmax row model: max-compare + subtract + exp + sum-add + divide.
inline constexpr int64_t kSoftmaxOpsPerElement = 5;

struct WalkCounts {
    int64_t ops{0};
    bool known{true};
};

[[nodiscard]] int64_t bufferElements(const KernelBuffer& b,
                                     bool& known) noexcept {
    int64_t prod = 0;
    if (!b.dims.empty()) {
        if (!dimsProduct(b.dims, prod)) {
            known = false;
            return 0;
        }
        return prod;
    }
    if (b.elements >= 0 && b.elements < constants::kKernelLoopDynamicBound) {
        return b.elements;
    }
    known = false;  // dynamic extent without dims: the model stays honest
    return 0;
}

/// Walks the node forest once: Call nodes contribute their declared
/// class model; Compute nodes contribute one op per fused expression
/// plus one per accumulate store (the += or max-select itself).
void walkOps(const KernelModule& kernel, const KernelNode& node,
             WalkCounts& counts) noexcept {
    switch (node.op) {
        case KernelOp::Compute: {
            if (counts.known) {
                const int64_t chain =
                    static_cast<int64_t>(node.exprs.size());
                const int64_t store =
                    node.accum != AccumMode::None ? 1 : 0;
                if (chain > (1LL << 40) - counts.ops) {
                    counts.known = false;
                } else {
                    counts.ops += chain + store;
                }
            }
            break;
        }
        case KernelOp::Call: {
            if (!counts.known) break;
            const bool aOk = node.bufferA < kernel.buffers.size();
            if (!aOk) {
                counts.known = false;
                break;
            }
            bool knownA = true;
            const int64_t aElems =
                bufferElements(kernel.buffers[node.bufferA], knownA);
            if (!knownA) {
                counts.known = false;
                break;
            }
            if (node.math == MathOp::MatMul) {
                // A = [M, K], B = [K, N]: the closed-form class model
                // needs B's last dim and A's first; anything else leaves
                // the count UNKNOWN (tri-state, Rule 22).
                if (node.bufferB >= kernel.buffers.size()) {
                    counts.known = false;
                    break;
                }
                const KernelBuffer& bufA = kernel.buffers[node.bufferA];
                const KernelBuffer& bufB = kernel.buffers[node.bufferB];
                if (bufA.dims.size() < 2 || bufB.dims.size() < 2) {
                    counts.known = false;
                    break;
                }
                const int64_t mRows = bufA.dims[0];
                const int64_t kDim = bufA.dims[1];
                const int64_t nCols = bufB.dims[bufB.dims.size() - 1];
                if (mRows <= 0 || kDim <= 0 || nCols <= 0) {
                    counts.known = false;
                    break;
                }
                const int64_t mk = mRows * kDim;  // magnitudes guarded
                if (mk > (1LL << 40) - counts.ops) {
                    counts.known = false;
                    break;
                }
                counts.ops += kFlopsPerMulAddPair * mk * nCols;
            } else if (node.math == MathOp::ReduceSum) {
                if (aElems > (1LL << 40) - counts.ops) {
                    counts.known = false;
                } else {
                    counts.ops += aElems;  // one add per input element
                }
            } else if (node.math == MathOp::Softmax) {
                if (aElems > ((1LL << 40) - counts.ops) /
                                 kSoftmaxOpsPerElement) {
                    counts.known = false;
                } else {
                    counts.ops += kSoftmaxOpsPerElement * aElems;
                }
            } else {
                counts.known = false;  // unmodeled call class: honest gap
            }
            break;
        }
        default:
            break;
    }
    for (const uint32_t child : node.children) {
        if (child >= kernel.nodes.size()) {
            counts.known = false;
            return;
        }
        walkOps(kernel, kernel.nodes[child], counts);
    }
}

}  // namespace

// --- Public API -----------------------------------------------------------

Result<EnvironmentModel> measureEnvironment() {
    EnvironmentModel env;
    env.hardwareThreads = std::thread::hardware_concurrency();
    if (env.hardwareThreads == 0) env.hardwareThreads = 1;
    const ProbeResult probes = runEnvironmentProbes();
    env.bandwidthBytesPerSec = probes.bandwidthBytesPerSec;
    env.throughputOpsPerSec = probes.throughputOpsPerSec;
    if (env.bandwidthBytesPerSec <= 0.0 || env.throughputOpsPerSec <= 0.0) {
        return err(ErrorCode::Internal,
                   "fastkernel: environment probes produced no time "
                   "(clock resolution?)");
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "cpu-threads=%u;bw=%.3gB/s;t=%.3gop/s",
                  env.hardwareThreads, env.bandwidthBytesPerSec,
                  env.throughputOpsPerSec);
    env.descriptor = buf;
    return env;
}

EssentialWork essentialWorkModel(const KernelModule& kernel) {
    EssentialWork work;
    // Essential bytes: every non-temp input/output buffer moved once.
    int64_t bytes = 0;
    bool bytesKnown = true;
    for (const KernelBuffer& b : kernel.buffers) {
        if (b.isTemp) continue;  // declared cache-resident assumption
        if (!b.isInput && !b.isOutput) continue;
        bool known = true;
        const int64_t elems = bufferElements(b, known);
        if (!known) {
            bytesKnown = false;
            break;
        }
        if (elems > (1LL << 40) - bytes / 8) {
            bytesKnown = false;
            break;
        }
        bytes += elems * 8;
    }
    work.minBytes = bytes;
    work.bytesKnown = bytesKnown;
    // Essential ops.
    WalkCounts counts;
    for (const KernelNode& node : kernel.nodes) {
        walkOps(kernel, node, counts);
        if (!counts.known) break;
    }
    work.minOps = counts.ops;
    work.opsKnown = counts.known;
    return work;
}

double rooflineLowerBound(const EssentialWork& work,
                          const EnvironmentModel& env) {
    if (!work.bytesKnown || !work.opsKnown) return 0.0;
    if (env.bandwidthBytesPerSec <= 0.0 || env.throughputOpsPerSec <= 0.0) {
        return 0.0;
    }
    const double bwTerm =
        static_cast<double>(work.minBytes) / env.bandwidthBytesPerSec;
    const double opTerm =
        static_cast<double>(work.minOps) / env.throughputOpsPerSec;
    return std::max(bwTerm, opTerm);
}

LaunchCoverage certifyLaunchCoverage(const KernelModule& kernel,
                                     SymbolTable& symbols,
                                     const int64_t domainElements) {
    LaunchCoverage cov;
    cov.covered = true;
    cov.threads = 1;
    if (domainElements <= 0) {
        cov.covered = false;
        return cov;
    }
    // Mirror decideThreads exactly (kernel_buffers.cpp): hw cap, then the
    // "threads" schedule param, then the spawn-amortization gate.
    std::size_t threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    if (threads > constants::kKernelExecMaxThreads) {
        threads = constants::kKernelExecMaxThreads;
    }
    const SymbolId threadsKey = symbols.intern("threads");
    if (const int64_t* v = kernel.scheduleParams.find(threadsKey)) {
        if (*v >= 1) threads = static_cast<std::size_t>(*v);
    }
    if (domainElements < constants::kParallelChunkElements) threads = 1;
    if (threads > constants::kKernelExecMaxThreads) {
        // Resource-limit violation (Axiom 14.16): the executor would cap
        // this — the certificate reports the violation, never hides it.
        cov.covered = false;
        cov.threads = static_cast<uint32_t>(threads);
        return cov;
    }
    cov.threads = static_cast<uint32_t>(threads);
    // Find the FIRST parallel-marked loop with a statically resolvable
    // trip count (constant end, or endBuf/endDim buffer dims) and
    // certify its decomposition; sequential-only kernels are trivially
    // covered by the single worker.
    const auto certifySlabs = [&](const int64_t total) {
        const int64_t chunk =
            (total + static_cast<int64_t>(threads) - 1) /
            static_cast<int64_t>(threads);
        int64_t covered = 0;
        int64_t prevEnd = -1;
        for (std::size_t t = 0; t < threads; ++t) {
            const int64_t b = static_cast<int64_t>(t) * chunk;
            const int64_t e = std::min(b + chunk, total) - 1;
            if (b > e) {
                // Empty tail slabs (total < threads * chunk) do no work;
                // they neither drop nor duplicate instances.
                LaunchSlab slab;
                slab.begin = b;
                slab.endInclusive = e;
                cov.slabs.push_back(slab);
                continue;
            }
            if (b != prevEnd + 1) {
                cov.covered = false;  // hole or overlap: disjoint union fails
            }
            prevEnd = e;
            covered += e - b + 1;
            LaunchSlab slab;
            slab.begin = b;
            slab.endInclusive = e;
            cov.slabs.push_back(slab);
        }
        if (covered != total || prevEnd != total - 1) cov.covered = false;
    };
    // Iterative traversal (no recursion bound issues; the module is a
    // tree — cycle freedom is the lower_to_kernel_ir contract).
    SmallVector<uint32_t, 32> stack{};
    for (uint32_t root = kernel.nodes.size(); root > 0; --root) {
        stack.push_back(root - 1);
    }
    bool certifiedOne = false;
    while (!stack.empty()) {
        const uint32_t id = stack.back();
        stack.pop_back();
        if (id >= kernel.nodes.size()) {
            cov.covered = false;
            break;
        }
        const KernelNode& n = kernel.nodes[id];
        if (n.isLoop() && n.parallel && !certifiedOne) {
            int64_t total = -1;
            if (n.end != constants::kKernelLoopDynamicBound) {
                total = n.end - n.begin;
            } else if (n.endBuf < kernel.buffers.size() &&
                       n.endDim >= 0 &&
                       n.endDim < static_cast<int32_t>(
                                      kernel.buffers[n.endBuf].dims.size())) {
                total = kernel.buffers[n.endBuf].dims[static_cast<
                    std::size_t>(n.endDim)] -
                        n.begin;
            }
            if (total > 0) {
                certifySlabs(total);
                certifiedOne = true;
            }
        }
        for (const uint32_t child : n.children) stack.push_back(child);
    }
    if (!certifiedOne && cov.slabs.empty()) {
        // No resolvable parallel loop: the single worker covers all work
        // sequentially — coverage holds trivially.
        cov.covered = cov.threads == 1;
        LaunchSlab whole;
        whole.begin = 0;
        whole.endInclusive = domainElements - 1;
        cov.slabs.push_back(whole);
    }
    return cov;
}

JustificationDecision evaluateExtraComptimeJustification(
    const double uNew, const double uBase, const double deltaM,
    const double tau, const double alpha,
    const int64_t executions) noexcept {
    JustificationDecision d;
    // Absolute runtime improvement (Axiom 14.9).
    if (uBase - uNew >= tau && uNew < uBase) {
        d.justified = true;
        d.marginSec = uBase - uNew;
        d.form = "runtime-improvement";
        return d;
    }
    // Amortized payoff (Axiom 14.9 / 15.7): alpha * dM <= N * dR.
    const double dR = uBase - uNew;
    if (executions > 0 && dR > 0.0 &&
        alpha * deltaM <=
            static_cast<double>(executions) * dR) {
        d.justified = true;
        d.marginSec =
            dR - alpha * deltaM / static_cast<double>(executions);
        d.form = "amortized-payoff";
        return d;
    }
    // Pareto improvement: runtime strictly better, every other budgeted
    // dimension within its bound (compile time is the excused one).
    if (dR > 0.0) {
        d.justified = true;
        d.marginSec = dR;
        d.form = "pareto-improvement";
        return d;
    }
    d.justified = false;
    d.marginSec = 0.0;
    d.form = "";
    return d;
}

const char* execPathName(const ExecPath p) noexcept {
    switch (p) {
        case ExecPath::Walker: return "walker";
        case ExecPath::NativeAsm: return "native-asm";
        case ExecPath::NativeCpp: return "native-cpp";
        case ExecPath::NativeCuda: return "native-cuda";
    }
    return "unknown";
}

const char* budgetModeName(const BudgetMode m) noexcept {
    switch (m) {
        case BudgetMode::SameComptime: return "same-comptime";
        case BudgetMode::BoundedExtra: return "bounded-extra-comptime";
        case BudgetMode::Amortized: return "amortized-comptime";
    }
    return "unknown";
}

const char* rejectReasonName(const RejectReason r) noexcept {
    switch (r) {
        case RejectReason::None: return "none";
        case RejectReason::IdentityMismatch: return "identity-mismatch";
        case RejectReason::ComptimeBudgetExceeded:
            return "comptime-budget-exceeded";
        case RejectReason::NoJustification: return "no-justification";
        case RejectReason::BuildFailed: return "build-failed";
        case RejectReason::ResourceLimit: return "resource-limit";
        case RejectReason::CapabilityUnsupported:
            return "capability-unsupported";
    }
    return "unknown";
}

// --- Search implementation ------------------------------------------------

namespace {

/// Deterministic input filler (Rule 53): the same pattern the mlk-poly
/// demo uses, applied flat over the input pool.
[[nodiscard]] double fillInput(const std::size_t flat,
                               const std::size_t salt) noexcept {
    return static_cast<double>(((flat * 7 + salt * 5) % 13)) * 0.25;
}

/// The per-variant Tier2 compile: stage-accounted (Axiom 15.2), with the
/// variant's tile knob or the poly.tile kill switch for the untiled form.
[[nodiscard]] Result<std::pair<KernelModule, double>> compileVariant(
    MathGraph& graph, SymbolTable& symbols,
    const MathDomainProfile& profile, const int64_t tileSize) {
    DiagnosticEngine diag;
    TelemetrySink telemetry;
    AccuracyContract contract;
    mlk::OpenHashMap<mlk::SymbolId, int64_t> knobs;
    mlk::OpenHashMap<mlk::SymbolId, bool> kills;
    const mlk::SymbolId tileKey = symbols.intern("poly_tile_size");
    const mlk::SymbolId tilePass = symbols.intern("poly.tile");
    if (tileSize > 0) {
        *knobs.findOrInsert(tileKey, nullptr) = tileSize;
    } else {
        // Untiled variant: the poly.tile kill switch (Rule 60).
        *kills.findOrInsert(tilePass, nullptr) = true;
    }
    KernelModule kernel;
    mlk::PipelineRunner runner(symbols, &telemetry);
    mlk::PassContext ctx;
    ctx.domainProfile = &profile;
    ctx.accuracy = &contract;
    ctx.diag = &diag;
    ctx.symbols = &symbols;
    ctx.tier = mlk::Tier::Tier2;
    ctx.kernelOut = &kernel;
    ctx.knobs = &knobs;
    ctx.killSwitches = &kills;
    const double t0 = nowSeconds();
    auto r = runner.run(mlk::Tier::Tier2, ctx, graph, &kernel);
    const double t1 = nowSeconds();
    if (!r.has_value()) {
        return err(ErrorCode::Internal,
                   "fastkernel: Tier2 compile failed: " +
                       r.error().message);
    }
    return std::make_pair(std::move(kernel), t1 - t0);
}

/// Builds the flat buffer bindings for a module: inputs in buffer-table
/// order (filled deterministically), outputs freshly allocated.
struct VariantBindings {
    KernelBufferBindings io{};
    SmallVector<SmallVector<double, 8>, 8> ownedInputs{};
    SmallVector<SmallVector<double, 8>, 8> ownedOutputs{};
    int64_t outputElements{0};
};

[[nodiscard]] Result<VariantBindings> makeBindings(
    const KernelModule& kernel, const std::size_t inputSalt) {
    VariantBindings vb;
    int64_t maxElements = 0;
    for (const KernelBuffer& b : kernel.buffers) {
        bool known = true;
        const int64_t elems = bufferElements(b, known);
        if (!known) {
            return err(ErrorCode::UnsupportedCapability,
                       "fastkernel: buffer '" +
                           std::to_string(b.elements) +
                           "' has no resolvable extent; the search "
                           "requires fully-shaped tensor kernels");
        }
        if (b.isInput) {
            SmallVector<double, 8> buf(static_cast<std::size_t>(elems));
            for (int64_t i = 0; i < elems; ++i) {
                buf[static_cast<std::size_t>(i)] =
                    fillInput(static_cast<std::size_t>(i), inputSalt);
            }
            vb.io.inputs.push_back(buf.data());
            vb.ownedInputs.push_back(std::move(buf));
        } else if (b.isOutput) {
            SmallVector<double, 8> buf(static_cast<std::size_t>(elems),
                                       0.0);
            vb.io.outputs.push_back(buf.data());
            vb.ownedOutputs.push_back(std::move(buf));
            maxElements = std::max(maxElements, elems);
        }
        maxElements = std::max(maxElements, elems);
    }
    if (vb.io.inputs.empty() || vb.io.outputs.empty()) {
        return err(ErrorCode::UnsupportedCapability,
                   "fastkernel: kernel must bind at least one input and "
                   "one output buffer");
    }
    vb.io.elements = maxElements;
    vb.outputElements = maxElements;
    return vb;
}

/// Compares two output pools bit-exactly (policy π: exact identity).
[[nodiscard]] double maxAbsDiff(const SmallVector<SmallVector<double, 8>, 8>&
                                    a,
                                const SmallVector<SmallVector<double, 8>, 8>&
                                    b) noexcept {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            return std::numeric_limits<double>::infinity();
        }
        for (std::size_t e = 0; e < a[i].size(); ++e) {
            const double x = a[i][e];
            const double y = b[i][e];
            if (std::isnan(x) || std::isnan(y)) {
                // NaN never equals anything — under the exact policy a
                // NaN mismatch is an identity mismatch (Axiom 14.17).
                return std::numeric_limits<double>::infinity();
            }
            worst = std::max(worst, std::fabs(x - y));
        }
    }
    return worst;
}

/// Benchmarks one callable: warmups then timed reps; returns
/// (median, min, max) — the declared benchmark-certificate method.
template <typename Fn>
[[nodiscard]] Result<std::tuple<double, double, double>> benchmark(
    Fn&& fn, const uint32_t warmupReps, const uint32_t benchReps) {
    for (uint32_t w = 0; w < warmupReps; ++w) {
        auto r = fn();
        if (!r.has_value()) return std::unexpected<Error>(r.error());
    }
    SmallVector<double, 16> times{};
    for (uint32_t r = 0; r < benchReps; ++r) {
        const double t0 = nowSeconds();
        auto run = fn();
        const double t1 = nowSeconds();
        if (!run.has_value()) return std::unexpected<Error>(run.error());
        times.push_back(t1 - t0);
    }
    double lo = times.front();
    double hi = times.front();
    for (const double t : times) {
        lo = std::min(lo, t);
        hi = std::max(hi, t);
    }
    return std::make_tuple(medianOf(std::move(times)), lo, hi);
}

/// The fingerprint binds EVERY cache-relevant component (Axiom 14.22):
/// kernel source hash, artifact kind, compiler, policy, budget mode,
/// environment descriptor, cache format version — and, for the Cuda
/// path, the DECLARED GPU compiler + arch (two arch declarations are
/// different artifacts and must never share a cache entry).
[[nodiscard]] std::string artifactFingerprint(const KernelModule& kernel,
                                              const ExecPath exec,
                                              const BackendDriverConfig&
                                                  driver,
                                              const GpuBackendDriverConfig&
                                                  gpu,
                                              const BudgetMode mode,
                                              const EnvironmentModel& env) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint32_t version = kFastKernelCacheVersion;
    h = fnv1a64(&version, sizeof(version), h);
    const HashValue kh = kernel.hash();
    h = fnv1a64(&kh, sizeof(kh), h);
    const uint8_t execV = static_cast<uint8_t>(exec);
    h = fnv1a64(&execV, sizeof(execV), h);
    h = fnv1a64String(driver.compiler, h);
    if (exec == ExecPath::NativeCuda) {
        h = fnv1a64String(gpu.compiler, h);
        h = fnv1a64String(gpu.arch, h);
    }
    const char* policy = kFastKernelPolicy;
    h = fnv1a64(policy, std::strlen(policy), h);
    const uint8_t modeV = static_cast<uint8_t>(mode);
    h = fnv1a64(&modeV, sizeof(modeV), h);
    h = fnv1a64String(env.descriptor, h);
    // The environment VALUES are fingerprint material too (Axiom 14.22:
    // the environment h itself must be unchanged, not merely its label;
    // two environments sharing a descriptor but differing in measured
    // peaks are different environments).
    const double bw = env.bandwidthBytesPerSec;
    const double tp = env.throughputOpsPerSec;
    const uint32_t hw = env.hardwareThreads;
    h = fnv1a64(&bw, sizeof(bw), h);
    h = fnv1a64(&tp, sizeof(tp), h);
    h = fnv1a64(&hw, sizeof(hw), h);
    return fingerprintHex(h);
}

[[nodiscard]] std::string joinPathLocal(const std::string& a,
                                        const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

[[nodiscard]] bool readTextFile(const std::string& path,
                                std::string& out) noexcept {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char buf[256];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    std::fclose(f);
    return true;
}

[[nodiscard]] bool writeTextFile(const std::string& path,
                                 const std::string& content) noexcept {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const std::size_t wrote =
        std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
    return wrote == content.size();
}

/// Native path resolution: cache probe (M^reuse) or build (M^first).
/// The Cuda path builds through the dedicated GPU entry points (the
/// plain buildKernelArtifact rejects Cuda by contract so the arch flag
/// can never be silently defaulted — spec #GPU-backend).
[[nodiscard]] Result<std::pair<LoadedKernel, VariantCertificate>>
obtainNativeKernel(const KernelModule& kernel, SymbolTable& symbols,
                   const ExecPath exec,
                   const FastKernelSearchConfig& config,
                   const EnvironmentModel& env, uint32_t& cacheHits,
                   uint32_t& cacheMisses) {
    const ArtifactKind kind = exec == ExecPath::NativeAsm
                                  ? ArtifactKind::Asm
                                  : ArtifactKind::Cpp;
    VariantCertificate cert;
    const std::string fp =
        artifactFingerprint(kernel, exec, config.driver, config.gpu,
                            config.budget.mode, env);
    cert.fingerprint = fp;
    const bool caching = !config.cacheDir.empty();
    std::string cacheDirName;
    std::string libPath;
    if (caching) {
        // The cache ROOT is this layer's responsibility (the driver
        // creates only the per-fingerprint leaf). The root itself may
        // be nested under a not-yet-existing base (fresh checkout,
        // fresh --cache-dir): createDirs walks the full parent chain —
        // a single-level mkdir here turned every first run without
        // pre-existing leftovers into a spurious IoError.
        if (auto rootMade = mlk::createDirs(config.cacheDir);
            !rootMade.has_value()) {
            return err(ErrorCode::IoError,
                       "fastkernel: cannot create cache dir: " +
                           config.cacheDir + " — " +
                           rootMade.error().message);
        }
        cacheDirName = joinPathLocal(config.cacheDir, "fk-" + fp);
        libPath = joinPathLocal(cacheDirName, "libmlk_kernel.so");
        std::string recorded;
        if (readTextFile(joinPathLocal(cacheDirName, "fingerprint.txt"),
                         recorded) &&
            recorded == fp + "\n") {
            const double l0 = nowSeconds();
            auto reused = loadKernelLibrary(kernel, symbols, libPath);
            const double l1 = nowSeconds();
            if (reused.has_value()) {
                cert.comptimeFromCache = true;  // M^reuse (Axiom 15.8)
                cert.loadSec = l1 - l0;
                cert.libraryPath = reused->libraryPath();
                ++cacheHits;
                return std::make_pair(std::move(*reused), std::move(cert));
            }
            // Fall through to a rebuild; the stale entry is overwritten.
        }
    }
    const double t0 = nowSeconds();
    std::string artifactDir;
    if (caching) {
        artifactDir = cacheDirName;
    }
    auto built =
        exec == ExecPath::NativeCuda
            ? (artifactDir.empty()
                   ? buildGpuKernelArtifact(kernel, symbols, config.gpu)
                   : buildGpuKernelArtifactInDir(kernel, symbols,
                                                 config.gpu, artifactDir))
            : (artifactDir.empty()
                   ? buildKernelArtifact(kernel, symbols, kind, config.driver)
                   : buildKernelArtifactInDir(kernel, symbols, kind,
                                              config.driver, artifactDir));
    const double t1 = nowSeconds();
    if (!built.has_value()) {
        return err(built.error().code,
                   "fastkernel: artifact build failed: " +
                       built.error().message);
    }
    cert.buildSec = t1 - t0;
    cert.libraryPath = built->libraryPath();
    if (caching) {
        // Record the entry key; a later probe must match EXACTLY.
        std::string probe;
        if (!readTextFile(joinPathLocal(cacheDirName, "fingerprint.txt"),
                          probe) ||
            probe != fp + "\n") {
            if (!writeTextFile(
                    joinPathLocal(cacheDirName, "fingerprint.txt"),
                    fp + "\n")) {
                return err(ErrorCode::IoError,
                           "fastkernel: cannot write cache key under " +
                               cacheDirName);
            }
        }
    }
    ++cacheMisses;
    return std::make_pair(std::move(*built), std::move(cert));
}

}  // namespace (search helpers)


// --- Report serialization (Axiom 14.20: the search space is reported) ----

/// Budget gate + objective finalization shared by both execution paths
/// (Axioms 14.6/15.6: a candidate over B_allowed is INADMISSIBLE — the
/// overrun is recorded as a structured rejection, never a silent pass;
/// Axiom 15.1: the amortized objective adds M^kernel / N).
void finishOutcome(VariantOutcome& o, const FastKernelSearchConfig& config,
                   const double b0, const double bAllowed) {
    const double m = o.cert.comptimeTotalSec;
    if (m > bAllowed) {
        o.reject = RejectReason::ComptimeBudgetExceeded;
        o.rejectDetail = "M^kernel=" + std::to_string(m) +
                         "s exceeds B_allowed=" + std::to_string(bAllowed) +
                         "s (mode " +
                         budgetModeName(config.budget.mode) + ")";
        return;
    }
    o.usedExtraComptime = m > b0;
    o.objectiveSec =
        config.budget.mode == BudgetMode::Amortized
            ? o.cert.runtimeUpperBoundSec +
                  m / static_cast<double>(config.budget.amortizedExecutions)
            : o.cert.runtimeUpperBoundSec;
    o.admissible = true;
}

json::Value FastKernelReport::toJson() const {
    json::Value doc = json::Object{};
    doc.set("spec", json::Value{"OEIA/Refined/FastKernels"});
    doc.set("policy", json::Value{kFastKernelPolicy});
    doc.set("cache_version",
            json::Value{static_cast<int64_t>(kFastKernelCacheVersion)});

    json::Value envDoc = json::Object{};
    envDoc.set("descriptor", json::Value{env.descriptor});
    envDoc.set("bandwidth_bytes_per_sec",
               json::Value{env.bandwidthBytesPerSec});
    envDoc.set("throughput_ops_per_sec",
               json::Value{env.throughputOpsPerSec});
    envDoc.set("hardware_threads",
               json::Value{static_cast<int64_t>(env.hardwareThreads)});
    doc.set("environment", std::move(envDoc));

    json::Value workDoc = json::Object{};
    workDoc.set("min_bytes", json::Value{work.minBytes});
    workDoc.set("min_ops", json::Value{work.minOps});
    workDoc.set("bytes_known", json::Value{work.bytesKnown});
    workDoc.set("ops_known", json::Value{work.opsKnown});
    workDoc.set("counting_rules",
                json::Value{
                    "B_min: non-temp buffers moved once; O_min: "
                    "MatMul=2*M*K*N, ReduceSum=M*K, Softmax=5*M*K "
                    "(max/sub/exp/add/div), Compute expr=1, accumulate "
                    "store=1"});
    doc.set("essential_work", std::move(workDoc));

    json::Value roof = json::Object{};
    roof.set("lower_bound_sec", json::Value{rooflineLowerSec});
    roof.set("winner_gap_over_lower", json::Value{winnerGapOverLower});
    roof.set("axiom", json::Value{"14.11/14.12"});
    doc.set("roofline", std::move(roof));

    json::Value budgetDoc = json::Object{};
    budgetDoc.set("mode", json::Value{""});  // replaced below (const ctx)
    doc.set("budget", std::move(budgetDoc));

    doc.set("baseline_runtime_sec", json::Value{baselineRuntimeSec});
    doc.set("baseline_comptime_sec", json::Value{baselineComptimeSec});
    doc.set("budget_provenance", json::Value{budgetProvenance});

    json::Value cache = json::Object{};
    cache.set("hits", json::Value{static_cast<int64_t>(cacheHits)});
    cache.set("misses", json::Value{static_cast<int64_t>(cacheMisses)});
    doc.set("cache", std::move(cache));

    json::Value cands = json::Array{};
    for (const VariantOutcome& o : candidates) {
        json::Value c = json::Object{};
        json::Value cfg = json::Object{};
        cfg.set("tile_size", json::Value{o.config.tileSize});
        cfg.set("exec", json::Value{execPathName(o.config.exec)});
        cfg.set("threads", json::Value{static_cast<int64_t>(
                               o.config.threads)});
        c.set("config", std::move(cfg));
        c.set("admissible", json::Value{o.admissible});
        if (o.reject != RejectReason::None) {
            json::Value rej = json::Object{};
            rej.set("reason", json::Value{rejectReasonName(o.reject)});
            rej.set("detail", json::Value{o.rejectDetail});
            c.set("rejected", std::move(rej));
        }
        json::Value cert = json::Object{};
        json::Value ident = json::Object{};
        ident.set("certified", json::Value{o.cert.identityCertified});
        ident.set("max_abs_diff", json::Value{o.cert.identityMaxAbsDiff});
        cert.set("identity", std::move(ident));
        json::Value run = json::Object{};
        run.set("upper_bound_sec",
                json::Value{o.cert.runtimeUpperBoundSec});
        run.set("min_sec", json::Value{o.cert.runtimeMinSec});
        run.set("max_sec", json::Value{o.cert.runtimeMaxSec});
        run.set("reps",
                json::Value{static_cast<int64_t>(o.cert.benchReps)});
        run.set("warmup",
                json::Value{static_cast<int64_t>(o.cert.warmupReps)});
        cert.set("runtime", std::move(run));
        json::Value comp = json::Object{};
        comp.set("compile_sec", json::Value{o.cert.compileSec});
        comp.set("emit_sec", json::Value{o.cert.emitSec});
        comp.set("build_sec", json::Value{o.cert.buildSec});
        comp.set("load_sec", json::Value{o.cert.loadSec});
        comp.set("total_sec", json::Value{o.cert.comptimeTotalSec});
        comp.set("from_cache", json::Value{o.cert.comptimeFromCache});
        cert.set("comptime", std::move(comp));
        json::Value launch = json::Object{};
        launch.set("covered", json::Value{o.cert.launch.covered});
        launch.set("threads", json::Value{static_cast<int64_t>(
                                 o.cert.launch.threads)});
        json::Value slabs = json::Array{};
        for (const LaunchSlab& s : o.cert.launch.slabs) {
            json::Value sd = json::Object{};
            sd.set("begin", json::Value{s.begin});
            sd.set("end_inclusive", json::Value{s.endInclusive});
            slabs.push(std::move(sd));
        }
        launch.set("slabs", std::move(slabs));
        cert.set("launch", std::move(launch));
        if (!o.cert.fingerprint.empty()) {
            cert.set("fingerprint", json::Value{o.cert.fingerprint});
        }
        if (!o.cert.libraryPath.empty()) {
            cert.set("library", json::Value{o.cert.libraryPath});
        }
        c.set("certificate", std::move(cert));
        c.set("objective_sec", json::Value{o.objectiveSec});
        c.set("used_extra_comptime", json::Value{o.usedExtraComptime});
        c.set("justification_margin_sec",
              json::Value{o.justificationMarginSec});
        cands.push(std::move(c));
    }
    doc.set("candidates", std::move(cands));
    doc.set("winner_index", json::Value{winnerIndex});
    doc.set("budget_failure", json::Value{budgetFailure});
    doc.set("claim", json::Value{claim});
    return doc;
}

Result<FastKernelReport> runFastKernelSearch(MathGraph& graph,
                                             SymbolTable& symbols,
                                             const MathDomainProfile&
                                                 profile,
                                             const FastKernelSearchConfig&
                                                 config) {
    // --- Declaration validation ------------------------------------------
    if (config.tileSizes.empty() || config.execPaths.empty() ||
        config.threadCounts.empty()) {
        return err(ErrorCode::InvalidArgument,
                   "fastkernel: the declared search space must be "
                   "non-empty (tiles x paths x threads)");
    }
    if (config.benchReps == 0) {
        return err(ErrorCode::InvalidArgument,
                   "fastkernel: benchReps must be >= 1 (a certificate "
                   "needs at least one observation)");
    }

    // --- Environment h (Axiom 0.3) ----------------------------------------
    FastKernelReport report;
    if (config.measureEnvironment) {
        auto env = measureEnvironment();
        if (!env.has_value()) return std::unexpected<Error>(env.error());
        report.env = *env;
    } else {
        report.env = config.env;
    }

    // --- The specification's reference execution (Axiom 14.2 oracle) -----
    KernelModule refKernel;
    {
        DiagnosticEngine diag;
        TelemetrySink telemetry;
        AccuracyContract contract;
        KernelModule tier1;
        mlk::PipelineRunner runner(symbols, &telemetry);
        mlk::PassContext ctx;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier1;
        ctx.kernelOut = &tier1;
        auto r = runner.run(mlk::Tier::Tier1, ctx, graph, &tier1);
        if (!r.has_value()) {
            return err(ErrorCode::Internal,
                       std::string("fastkernel: Tier1 reference compile "
                                   "failed: ") +
                           r.error().message);
        }
        refKernel = std::move(tier1);
    }
    auto refBindings = makeBindings(refKernel, 0);
    if (!refBindings.has_value()) {
        return std::unexpected<Error>(refBindings.error());
    }
    auto refRun = executeKernelOnBuffers(refKernel, symbols,
                                         refBindings->io, nullptr);
    if (!refRun.has_value()) {
        return err(ErrorCode::Internal,
                   "fastkernel: reference execution failed: " +
                       refRun.error().message);
    }
    SmallVector<SmallVector<double, 8>, 8> refOutputs{};
    for (const SmallVector<double, 8>& buf : refBindings->ownedOutputs) {
        refOutputs.push_back(buf);
    }

    // Roofline model from the SPECIFICATION module (Axiom 14.12: the
    // specification's essential movement and operations).
    report.work = essentialWorkModel(refKernel);
    report.rooflineLowerSec = rooflineLowerBound(report.work, report.env);

    // --- The specification's own runtime (the baseline column) -----------
    {
        auto base = benchmark(
            [&] {
                return executeKernelOnBuffers(refKernel, symbols,
                                              refBindings->io, nullptr);
            },
            config.warmupReps, config.benchReps);
        if (base.has_value()) {
            report.baselineRuntimeSec = std::get<0>(*base);
        }
        // A failed baseline benchmark does not invalidate the search;
        // the certificate column simply stays empty (recorded honestly).
    }

    // --- Budget baseline B_K^0 (Axiom 14.7; auto = default pipeline) -----
    double b0 = config.budget.baselineComptimeSec;
    if (b0 > 0.0) {
        report.budgetProvenance = "policy-set baseline comptime budget";
    } else {
        // Auto-measurement methodology (declared): 3 samples of the
        // default-pipeline compile, B_K^0 = max — a single sample
        // carries ~0.3% compile noise, which would decide admissibility
        // by measurement luck at the budget boundary. The max is the
        // conservative baseline estimate.
        constexpr uint32_t kAutoBaselineSamples = 3;
        double worst = 0.0;
        for (uint32_t s = 0; s < kAutoBaselineSamples; ++s) {
            auto defaultCompile =
                compileVariant(graph, symbols, profile,
                               constants::kPolyDefaultTileSize);
            if (!defaultCompile.has_value()) {
                return std::unexpected<Error>(defaultCompile.error());
            }
            worst = std::max(worst, defaultCompile->second);
        }
        b0 = worst;
        report.budgetProvenance =
            "auto-measured: max of 3 default Tier2 pipeline compiles";
    }
    report.baselineComptimeSec = b0;
    const double bAllowed =
        b0 + (config.budget.mode == BudgetMode::SameComptime
                  ? 0.0
                  : config.budget.extraComptimeMaxSec);

    // --- Candidate enumeration (declared order; Axiom 14.3) ---------------
    // The declared tile axes are deduplicated in declaration order: a
    // repeated declaration is one candidate, not two (Axiom 14.3).
    SmallVector<int64_t, 8> tiles{};
    for (const int64_t t : config.tileSizes) {
        bool seen = false;
        for (const int64_t u : tiles) seen = seen || u == t;
        if (!seen) tiles.push_back(t);
    }
    const std::size_t execCap = constants::kKernelExecMaxThreads;
    for (const int64_t tile : tiles) {
        for (const ExecPath exec : config.execPaths) {
            for (const uint32_t threads : config.threadCounts) {
                VariantOutcome o;
                o.config = VariantConfig{tile, exec, threads};

                // Resource gate before any work (Axiom 14.16): a walker
                // thread override above the executor cap can never run.
                if (exec == ExecPath::Walker &&
                    threads > static_cast<uint32_t>(execCap) &&
                    threads != 0) {
                    o.reject = RejectReason::ResourceLimit;
                    o.rejectDetail =
                        "threads=" + std::to_string(threads) +
                        " exceeds the executor cap kKernelExecMaxThreads=" +
                        std::to_string(execCap);
                    report.candidates.push_back(std::move(o));
                    continue;
                }

                // Compile (stage-accounted; Axiom 15.2).
                auto compiled = compileVariant(graph, symbols, profile,
                                               tile);
                if (!compiled.has_value()) {
                    o.reject = RejectReason::BuildFailed;
                    o.rejectDetail = compiled.error().message;
                    report.candidates.push_back(std::move(o));
                    continue;
                }
                KernelModule kernel = std::move(compiled->first);
                o.cert.compileSec = compiled->second;

                // Launch-config selection (Axiom 14.13): the thread
                // override is part of the kernel object.
                if (exec == ExecPath::Walker && threads > 0) {
                    *kernel.scheduleParams.findOrInsert(
                        symbols.intern("threads"), nullptr) =
                        static_cast<int64_t>(threads);
                }

                // Native artifact (emit + out-of-process build + load,
                // or the fingerprint-validated cache reuse; Axiom 14.22).
                if (exec != ExecPath::Walker) {
                    auto native = obtainNativeKernel(
                        kernel, symbols, exec, config, report.env,
                        report.cacheHits, report.cacheMisses);
                    if (!native.has_value()) {
                        o.reject = RejectReason::BuildFailed;
                        o.rejectDetail = native.error().message;
                        if (native.error().code ==
                            ErrorCode::UnsupportedCapability) {
                            o.reject =
                                RejectReason::CapabilityUnsupported;
                        }
                        report.candidates.push_back(std::move(o));
                        continue;
                    }
                    LoadedKernel& loaded = std::get<0>(*native);
                    o.cert = std::get<1>(*native);
                    o.cert.compileSec = compiled->second;
                    o.cert.comptimeTotalSec =
                        o.cert.compileSec + o.cert.emitSec +
                        o.cert.buildSec + o.cert.loadSec;
                    o.cert.launch = certifyLaunchCoverage(
                        kernel, symbols, refBindings->outputElements);

                    // Identity (Axiom 14.2): bit-exact vs the reference.
                    auto io = makeBindings(kernel, 0);
                    if (!io.has_value()) {
                        o.reject = RejectReason::BuildFailed;
                        o.rejectDetail = io.error().message;
                        report.candidates.push_back(std::move(o));
                        continue;
                    }
                    auto run = loaded.run(kernel, symbols, io->io);
                    if (!run.has_value()) {
                        o.reject = RejectReason::BuildFailed;
                        o.rejectDetail = "native run failed: " +
                                         run.error().message;
                        report.candidates.push_back(std::move(o));
                        continue;
                    }
                    const double diff =
                        maxAbsDiff(io->ownedOutputs, refOutputs);
                    o.cert.identityMaxAbsDiff = diff;
                    if (diff != 0.0) {
                        o.reject = RejectReason::IdentityMismatch;
                        o.rejectDetail =
                            "native artifact diverged from the "
                            "reference (max|diff| != 0)";
                        report.candidates.push_back(std::move(o));
                        continue;
                    }
                    o.cert.identityCertified = true;

                    // Runtime upper bound (Axiom 14.4).
                    auto bench =
                        benchmark([&] {
                            return loaded.run(kernel, symbols, io->io);
                        }, config.warmupReps, config.benchReps);
                    if (!bench.has_value()) {
                        o.reject = RejectReason::BuildFailed;
                        o.rejectDetail = "native benchmark failed: " +
                                         bench.error().message;
                        report.candidates.push_back(std::move(o));
                        continue;
                    }
                    o.cert.runtimeCertified = true;
                    o.cert.runtimeUpperBoundSec = std::get<0>(*bench);
                    o.cert.runtimeMinSec = std::get<1>(*bench);
                    o.cert.runtimeMaxSec = std::get<2>(*bench);
                    o.cert.benchReps = config.benchReps;
                    o.cert.warmupReps = config.warmupReps;
                    finishOutcome(o, config, b0, bAllowed);
                    report.candidates.push_back(std::move(o));
                    continue;
                }

                // Walker path: identity, launch, runtime.
                auto io = makeBindings(kernel, 0);
                if (!io.has_value()) {
                    o.reject = RejectReason::BuildFailed;
                    o.rejectDetail = io.error().message;
                    report.candidates.push_back(std::move(o));
                    continue;
                }
                o.cert.comptimeTotalSec = o.cert.compileSec;
                o.cert.launch = certifyLaunchCoverage(
                    kernel, symbols, refBindings->outputElements);
                auto run = executeKernelOnBuffers(kernel, symbols,
                                                  io->io, nullptr);
                if (!run.has_value()) {
                    o.reject = RejectReason::BuildFailed;
                    o.rejectDetail = "walker run failed: " +
                                     run.error().message;
                    report.candidates.push_back(std::move(o));
                    continue;
                }
                const double diff = maxAbsDiff(io->ownedOutputs,
                                               refOutputs);
                o.cert.identityMaxAbsDiff = diff;
                if (diff != 0.0) {
                    o.reject = RejectReason::IdentityMismatch;
                    o.rejectDetail =
                        "transformed kernel diverged from the "
                        "reference (max|diff| != 0)";
                    report.candidates.push_back(std::move(o));
                    continue;
                }
                o.cert.identityCertified = true;
                auto bench = benchmark(
                    [&] {
                        return executeKernelOnBuffers(kernel, symbols,
                                                      io->io, nullptr);
                    },
                    config.warmupReps, config.benchReps);
                if (!bench.has_value()) {
                    o.reject = RejectReason::BuildFailed;
                    o.rejectDetail = "walker benchmark failed: " +
                                     bench.error().message;
                    report.candidates.push_back(std::move(o));
                    continue;
                }
                o.cert.runtimeCertified = true;
                o.cert.runtimeUpperBoundSec = std::get<0>(*bench);
                o.cert.runtimeMinSec = std::get<1>(*bench);
                o.cert.runtimeMaxSec = std::get<2>(*bench);
                o.cert.benchReps = config.benchReps;
                o.cert.warmupReps = config.warmupReps;
                finishOutcome(o, config, b0, bAllowed);
                report.candidates.push_back(std::move(o));
            }
        }
    }

    // --- Justification gate for extra-comptime candidates (Axiom 14.9) ---
    if (config.budget.mode == BudgetMode::BoundedExtra) {
        double bestSameU = std::numeric_limits<double>::infinity();
        for (const VariantOutcome& o : report.candidates) {
            if (o.admissible && !o.usedExtraComptime) {
                bestSameU = std::min(bestSameU,
                                     o.cert.runtimeUpperBoundSec);
            }
        }
        if (std::isinf(bestSameU)) {
            // No same-comptime candidate is admissible: the fallback
            // U_base is the specification's own certified runtime.
            bestSameU = report.baselineRuntimeSec;
        }
        for (VariantOutcome& o : report.candidates) {
            if (!o.admissible || !o.usedExtraComptime) continue;
            const double u = o.cert.runtimeUpperBoundSec;
            const double dM = o.cert.comptimeTotalSec - b0;
            const JustificationDecision d =
                evaluateExtraComptimeJustification(
                    u, bestSameU, dM,
                    config.budget.minRuntimeImprovementSec,
                    config.budget.alpha,
                    config.budget.amortizedExecutions);
            if (d.justified) {
                o.justificationMarginSec = d.marginSec;
            } else {
                o.admissible = false;
                o.reject = RejectReason::NoJustification;
                o.rejectDetail =
                    "extra comptime dM=" + std::to_string(dM) +
                    "s without a certified payoff (u=" +
                    std::to_string(u) + "s, base=" +
                    std::to_string(bestSameU) + "s, tau=" +
                    std::to_string(config.budget.minRuntimeImprovementSec) +
                    "s)";
            }
        }
    }

    // --- Winner (Axiom 14.10) + honest claim (Axiom 14.21) ----------------
    int64_t winner = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(
                                report.candidates.size());
         ++i) {
        const VariantOutcome& o = report.candidates[static_cast<
            std::size_t>(i)];
        if (!o.admissible) continue;
        if (winner < 0 ||
            o.objectiveSec <
                report.candidates[static_cast<std::size_t>(winner)]
                    .objectiveSec) {
            winner = i;
        }
    }
    report.winnerIndex = winner;
    if (winner < 0) {
        report.budgetFailure = true;
        report.claim =
            "budget failure: no certified feasible kernel within the "
            "declared compile-time budget (Axiom 15.6)";
    } else {
        const VariantOutcome& w =
            report.candidates[static_cast<std::size_t>(winner)];
        if (report.rooflineLowerSec > 0.0 &&
            w.cert.runtimeUpperBoundSec > 0.0) {
            report.winnerGapOverLower =
                w.cert.runtimeUpperBoundSec / report.rooflineLowerSec -
                1.0;
        }
        bool everyCertified = true;
        for (const VariantOutcome& o : report.candidates) {
            if (o.reject == RejectReason::BuildFailed ||
                o.reject == RejectReason::CapabilityUnsupported) {
                everyCertified = false;
            }
        }
        if (everyCertified) {
            report.claim =
                "fastest certified kernel within the declared searched "
                "space (exhaustive finite search; Axiom 14.21)";
        } else {
            report.claim =
                "best certified kernel in searched space (some "
                "candidates could not be certified; Axiom 14.21)";
        }
    }
    return report;
}

}  // namespace mlk::fastkernel
