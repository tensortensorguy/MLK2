// MLK+ math benchmark runner (bench/math/).
//
// Protocol (Rule 49 — statistically valid; Rule 50/58 — correctness BEFORE
// benchmarking; Rule 30 — no silent fallbacks, every event recorded):
//   1. Build the workload MathGraph.
//   2. Compile at Tier 2 twice:
//        - "libm"  : exact profile, no approximation allowed.
//        - "poly7" : approximation permitted under an explicit accuracy
//                    contract (max 2 ULP) — the superoptimizer-verified
//                    sin family may then be selected by approx.function_lower.
//   3. Verify the executed kernel against the reference semantics
//      (std::sin / std::tanh / ... composed per the workload definition).
//   4. Measure: warmup, reps, min/median/stddev, GB/s.
//   5. Emit JSON to bench/results/mlk_math.json for the cross-framework
//      report (vs torch.compile / JAX / C baselines).
#include "math_workloads.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/pass/register_all.h"
#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/proof/accuracy_contract.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/telemetry.h"
#include "mlk/type/domain_profile.h"

namespace {

using namespace mlk;
using namespace mlk::bench;

// Rule 27: protocol constants are named and documented.
constexpr int64_t kElementCount = 1 << 20;   // 1,048,576 f64 elements
constexpr int64_t kMatDim = 512;             // GEMM: 512x512x512
constexpr uint32_t kWarmupReps = 10;
constexpr uint32_t kMeasureReps = 30;
constexpr double kVerifyMaxAbsExact = 1e-12;   // libm path vs reference
constexpr double kVerifyMaxRelExact = 1e-12;
constexpr double kVerifyMaxRelPoly = 5e-13;    // poly sin = <=2 ULP claimed

struct Stats {
    double minMs{};
    double medianMs{};
    double stddevMs{};
    double gbps{};
    double compileMs{};
    double maxRelError{};
};

[[nodiscard]] MathDomainProfile makeProfile(SymbolTable& symbols,
                                            const bool allowApprox) {
    MathDomainProfile p;
    p.name = symbols.intern("tensor_f64_cpu");
    p.version = symbols.intern("1.0.0");
    p.capabilities = CapabilityFlags{};
    p.capabilities.set(Capability::HasNumericValues);
    p.capabilities.set(Capability::HasFloatingPoint);
    p.capabilities.set(Capability::HasTensorDomain);
    p.capabilities.set(Capability::HasDynamicShapes);
    p.capabilities.set(Capability::HasKernelFusion);
    p.capabilities.set(Capability::HasAutotuning);
    p.capabilities.set(Capability::HasCPUBackend);
    p.capabilities.set(Capability::HasCalculus);
    p.capabilities.set(Capability::HasDerivatives);
    p.capabilities.set(Capability::HasGradients);
    if (allowApprox) {
        p.capabilities.set(Capability::HasApproximation);
        p.approximation.allowApproximation = true;
        p.approximation.defaultMaxUlps = 2.0;
    }
    return p;
}

[[nodiscard]] AccuracyContract makeContract(const bool allowApprox) {
    AccuracyContract c;
    if (allowApprox) {
        c.maxUlps = 2.0;
        c.maxAbsError = 1e-12;
        c.maxRelError = 1e-12;
        c.allowFastMath = true;  // scoped: profile + contract gate (Rule 91)
    }
    return c;
}

[[nodiscard]] Result<KernelModule> compileKernel(
    MathGraph& graph, const MathDomainProfile& profile,
    const AccuracyContract& contract, SymbolTable& symbols,
    TelemetrySink& telemetry) {
    KernelModule kernel;
    DiagnosticEngine diag;
    CancellationToken cancel;
    PipelineRunner runner(symbols, &telemetry);
    PassContext ctx;
    ctx.domainProfile = &profile;
    ctx.accuracy = &contract;
    ctx.diag = &diag;
    ctx.telemetry = &telemetry;
    ctx.cancel = &cancel;
    ctx.symbols = &symbols;
    ctx.tier = Tier::Tier2;
    ctx.kernelOut = &kernel;
    auto runResult = runner.run(Tier::Tier2, ctx, graph, &kernel);
    if (!runResult.has_value()) {
        for (const auto& d : diag.entries()) {
            std::fprintf(stderr, "diag[node %u]: %s | expected: %s | "
                                 "actual: %s | fix: %s\n",
                         d.nodeId, d.message.c_str(),
                         d.expected.c_str(), d.actual.c_str(),
                         d.suggestedFix.c_str());
        }
        return std::unexpected<Error>(std::move(runResult).error());
    }
    (void)*runResult;
    if (diag.hasErrors()) {
        std::string first = "tier-2 compilation diagnostics";
        for (const auto& d : diag.entries()) {
            if (d.severity == Severity::Error ||
                d.severity == Severity::Fatal) {
                first = d.message;
                break;
            }
        }
        return err(ErrorCode::VerificationFailed, first, 47);
    }
    return kernel;
}

[[nodiscard]] double nowMs() noexcept {
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

[[nodiscard]] Result<Stats> measure(KernelModule& kernel,
                                    SymbolTable& symbols,
                                    const KernelBufferBindings& io) {
    Stats s;
    CancellationToken cancel;
    std::vector<double> samples;
    samples.reserve(kMeasureReps);

    // Warmup (Rule 49: caches, allocator, page-ins stabilize first).
    for (uint32_t w = 0; w < kWarmupReps; ++w) {
        MLK_TRYV(executeKernelOnBuffers(kernel, symbols, io, &cancel));
    }
    for (uint32_t r = 0; r < kMeasureReps; ++r) {
        const double t0 = nowMs();
        MLK_TRYV(executeKernelOnBuffers(kernel, symbols, io, &cancel));
        samples.push_back(nowMs() - t0);
    }
    std::sort(samples.begin(), samples.end());
    s.minMs = samples.front();
    s.medianMs = samples[samples.size() / 2];
    double mean = 0.0;
    for (const double v : samples) mean += v;
    mean /= static_cast<double>(samples.size());
    double var = 0.0;
    for (const double v : samples) var += (v - mean) * (v - mean);
    s.stddevMs = std::sqrt(var / static_cast<double>(samples.size()));
    return s;
}


[[nodiscard]] double maxRel(const double* got, const double* ref,
                            const int64_t n) {
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double denom = std::fabs(ref[i]) > 1.0 ? std::fabs(ref[i]) : 1.0;
        const double e = std::fabs(got[i] - ref[i]) / denom;
        if (e > worst) worst = e;
    }
    return worst;
}

void setGbps(Stats& s, const int64_t bytes) {
    s.gbps = (static_cast<double>(bytes) / 1e9) / (s.medianMs / 1e3);
}

[[nodiscard]] std::string statsJson(const char* name, const char* variant,
                                    const int64_t n, const Stats& s) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  R"({"name":"%s","variant":"%s","n":%lld,"dtype":"f64",)"
                  R"("min_ms":%.6f,"median_ms":%.6f,"stddev_ms":%.6f,)"
                  R"("gbps":%.3f,"compile_ms":%.3f,"max_rel_err":%.3e})",
                  name, variant, static_cast<long long>(n), s.minMs,
                  s.medianMs, s.stddevMs, s.gbps, s.compileMs,
                  s.maxRelError);
    return std::string(buf);
}

/// One workload: compile both variants, verify both, measure both.
/// The graph is REBUILT per variant (compilation mutates the graph through
/// the pass pipeline; the IR is versioned, not destructively reused).
[[nodiscard]] Result<std::string> runElementwiseWorkload(
    const char* name, SymbolTable& symbols, TelemetrySink& telemetry,
    const std::function<Result<MathGraph>(SymbolTable&, int64_t)>& builder,
    double* x, double* y, const int64_t n,
    const std::vector<double>& refExact, const std::vector<double>& refPoly) {
    std::string out;
    const MathDomainProfile exactProfile = makeProfile(symbols, false);
    const AccuracyContract exactContract = makeContract(false);
    const MathDomainProfile polyProfile = makeProfile(symbols, true);
    const AccuracyContract polyContract = makeContract(true);

    struct Variant {
        const char* label;
        const MathDomainProfile* profile;
        const AccuracyContract* contract;
        const std::vector<double>* ref;
        const double tol;
    };
    const Variant variants[] = {
        {"libm", &exactProfile, &exactContract, &refExact,
         kVerifyMaxRelExact},
        {"poly7", &polyProfile, &polyContract, &refPoly, kVerifyMaxRelPoly},
    };

    for (const Variant& v : variants) {
        MLK_TRY_VAR(work, builder(symbols, n));
        const double c0 = nowMs();
        MLK_TRY_VAR(kernel, compileKernel(work, *v.profile, *v.contract,
                                          symbols, telemetry));
        const double compileMs = nowMs() - c0;

        KernelBufferBindings io;
        io.inputs.push_back(x);
        io.outputs.push_back(y);
        io.elements = n;
        MLK_TRYV(executeKernelOnBuffers(kernel, symbols, io, nullptr));
        const double rel = maxRel(y, v.ref->data(), n);
        if (rel > v.tol) {
            return err(ErrorCode::VerificationFailed,
                       std::string(name) + " [" + v.label +
                           "] FAILED verification: max rel err " +
                           std::to_string(rel) + " > " +
                           std::to_string(v.tol) +
                           " (Rule 58: no benchmarking of wrong kernels)",
                       58);
        }
        MLK_TRY_VAR(st, measure(kernel, symbols, io));
        st.compileMs = compileMs;
        st.maxRelError = rel;
        setGbps(st, 2 * n * 8);  // read x + write y
        out += statsJson(name, v.label, n, st);
        out += ",\n";
    }
    return out;
}

[[nodiscard]] Result<std::string> runMatMul(SymbolTable& symbols,
                                            TelemetrySink& telemetry) {
    const int64_t m = kMatDim, k = kMatDim, n = kMatDim;
    std::vector<double> a(static_cast<std::size_t>(m * k));
    std::vector<double> b(static_cast<std::size_t>(k * n));
    fillLowDiscrepancy(a.data(), m * k, -1.0, 1.0);
    fillLowDiscrepancy(b.data(), k * n, -1.0, 1.0);
    std::vector<double> c(static_cast<std::size_t>(m * n), 0.0);
    std::vector<double> ref(static_cast<std::size_t>(m * n), 0.0);

    MLK_TRY_VAR(work, Workloads::matMul(symbols, m, k, n));

    const MathDomainProfile exactProfile = makeProfile(symbols, false);
    const AccuracyContract exactContract = makeContract(false);
    const double c0 = nowMs();
    MLK_TRY_VAR(kernel,
                compileKernel(work, exactProfile, exactContract, symbols,
                              telemetry));
    const double compileMs = nowMs() - c0;

    KernelBufferBindings io;
    io.inputs.push_back(a.data());
    io.inputs.push_back(b.data());
    io.outputs.push_back(c.data());
    io.elements = m * n;
    MLK_TRYV(executeKernelOnBuffers(kernel, symbols, io, nullptr));

    // Reference GEMM (naive ikj, single-thread, verified before timing).
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const double aik = a[static_cast<std::size_t>(i * k + p)];
            for (int64_t j = 0; j < n; ++j) {
                ref[static_cast<std::size_t>(i * n + j)] +=
                    aik * b[static_cast<std::size_t>(p * n + j)];
            }
        }
    }
    const double rel = maxRel(c.data(), ref.data(), m * n);
    if (rel > 1e-10) {
        return err(ErrorCode::VerificationFailed,
                   "matmul FAILED verification: max rel err " +
                       std::to_string(rel) +
                       " (Rule 58: no benchmarking of wrong kernels)",
                   58);
    }
    MLK_TRY_VAR(st, measure(kernel, symbols, io));
    st.compileMs = compileMs;
    st.maxRelError = rel;
    setGbps(st, (2LL * m * k + m * n) * 8);
    return statsJson("matmul_512", "blocked_gemm", m * n, st) + "\n";
}

}  // namespace

int main() {
    SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    TelemetrySink telemetry;

    const int64_t n = kElementCount;
    std::vector<double> x(static_cast<std::size_t>(n));
    std::vector<double> y(static_cast<std::size_t>(n));
    fillLowDiscrepancy(x.data(), n, -3.0, 3.0);

    std::string json = "{\"format\":\"mlk-math-bench\",\"protocol\":{"
        "\"warmup\":10,\"reps\":30,\"dtype\":\"f64\"},"
        "\"results\":[\n";

    auto emit = [&](const std::string& s) { json += s; };

    struct Wl {
        const char* name;
        double lo, hi;
    };
    // Domain tuning per workload (Rule 56: realistic ranges; exp over
    // [-2,2) never overflows; tanh saturates only at the tails).
    const Wl wls[] = {
        {"sin_sq_3x", -3.0, 3.0},
        {"sin_sq_3x_fused", -3.0, 3.0},
        {"tanh", -3.0, 3.0},
        {"exp", -2.0, 2.0},
    };

    for (const Wl& w : wls) {
        fillLowDiscrepancy(x.data(), n, w.lo, w.hi);
        std::vector<double> refExact(static_cast<std::size_t>(n));
        std::vector<double> refPoly(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const double xi = x[static_cast<std::size_t>(i)];
            double r = 0.0;
            if (std::strcmp(w.name, "sin_sq_3x") == 0 ||
                std::strcmp(w.name, "sin_sq_3x_fused") == 0) {
                r = std::sin(xi * xi + 3.0 * xi);
                if (std::strcmp(w.name, "sin_sq_3x_fused") == 0) {
                    r = r * 1.5 + 0.25;
                }
            } else if (std::strcmp(w.name, "tanh") == 0) {
                r = std::tanh(xi);
            } else {
                r = std::exp(xi);
            }
            refExact[static_cast<std::size_t>(i)] = r;
            refPoly[static_cast<std::size_t>(i)] = r;
        }

        std::function<Result<MathGraph>(SymbolTable&, int64_t)> builder;
        if (std::strcmp(w.name, "sin_sq_3x") == 0) {
            builder = Workloads::sinSq3x;
        } else if (std::strcmp(w.name, "sin_sq_3x_fused") == 0) {
            builder = Workloads::sinSq3xFused;
        } else if (std::strcmp(w.name, "tanh") == 0) {
            builder = Workloads::tanhWl;
        } else {
            builder = Workloads::expWl;
        }
        auto r = runElementwiseWorkload(w.name, symbols, telemetry, builder,
                                        x.data(), y.data(), n, refExact,
                                        refPoly);
        if (!r.has_value()) {
            std::fprintf(stderr, "workload %s failed: %s\n", w.name,
                         r.error().message.c_str());
            return 1;
        }
        emit(*r);
    }

    // Derivative workload: y = d/dx[x²·sin(x)].
    {
        fillLowDiscrepancy(x.data(), n, -3.0, 3.0);
        std::vector<double> ref(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const double xi = x[static_cast<std::size_t>(i)];
            // Symbolic derivative: 2x·sin(x) + x²·cos(x).
            ref[static_cast<std::size_t>(i)] =
                2.0 * xi * std::sin(xi) + xi * xi * std::cos(xi);
        }
        auto r = runElementwiseWorkload(
            "derivative_x2_sin_x", symbols, telemetry,
            Workloads::derivativeX2SinX, x.data(), y.data(), n, ref, ref);
        if (!r.has_value()) {
            std::fprintf(stderr, "derivative workload failed: %s\n",
                         r.error().message.c_str());
            return 1;
        }
        emit(*r);
    }

    // GEMM workload.
    {
        auto r = runMatMul(symbols, telemetry);
        if (!r.has_value()) {
            std::fprintf(stderr, "matmul workload failed: %s\n",
                         r.error().message.c_str());
            return 1;
        }
        emit(*r);
    }

    // Trim the trailing comma/newline and close the JSON array.
    while (!json.empty() && (json.back() == '\n' || json.back() == ',')) {
        json.pop_back();
    }
    json += "\n]}\n";
    std::printf("%s", json.c_str());
    return 0;
}
