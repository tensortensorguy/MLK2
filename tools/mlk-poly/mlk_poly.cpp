// mlk-poly — polyhedral pipeline driver (layout spec §13):
//   mlk-poly demo [--backend=asm|cpp]
//                 self-contained MatMul walkthrough: compiles a GEMM
//                 graph at Tier1 (baseline) and Tier2 (polyhedral),
//                 dumps both kernel JSONs, and executes baseline vs
//                 transformed over dense buffers with a max-diff report;
//                 with --backend it additionally lowers the transformed
//                 kernel to a native artifact (x86-64 assembly or C++),
//                 builds it OUT-OF-PROCESS (ADR-0003/0006: no in-process
//                 machine codegen), loads it, executes it, and reports
//                 the three-way max-diff (walker vs artifact bit-exact)
//   mlk-poly show <graph.mlk>
//                 compiles any graph at Tier2 and dumps the kernel
//                 JSON (non-affine kernels report baseline fallback)
//   mlk-poly autotune [--m=M --k=K --n=N] [--tiles=0,16,32]
//                 [--exec=walker,asm,cpp] [--budget-mode=same|extra|
//                 amortized] [--comptime-budget-ms=X] [--extra-budget-ms=X]
//                 [--tau-ms=X] [--alpha=A] [--executions=N] [--warmup=W]
//                 [--reps=R] [--cache-dir=D] [--workdir=D] [--json-out=F]
//                 runs the OEIA/Refined/FastKernels budgeted search over
//                 the declared GEMM variant space: per-variant Tier2
//                 compile, bit-exact identity certificates, benchmarked
//                 runtime upper bounds, roofline lower bound + gap,
//                 budget gates, and the honest winner claim (Axiom 14.21).
//                 --json-out writes the full machine-readable report.
//   mlk-poly emit <graph.mlk> [--asm|--cpp] [--out=<path>]
//                 [--workdir=<dir>] [--compile]
//                 compiles the graph at Tier2 and emits the standalone
//                 artifact text (kernel.s | kernel.cpp) to stdout or
//                 --out; --compile additionally builds a shared object
//                 with the system toolchain and prints its path.
//                 Boundary: the mlk-graph text format cannot represent
//                 tensor descriptors yet (upstream serializer), so tensor
//                 graphs enter through the GraphBuilder API (demo) —
//                 emit covers every graph the Tier2 lowering accepts
//   mlk-poly bench [--reps=R] [--warmup=W]
//                  [--suites=gemm,softmax,reducesum] [--json-out=F]
//                  [--workdir=D]
//                 runs the polyhedral runtime benchmark: per suite case
//                 (declared shapes and flop models) the Tier1 call
//                 baseline, the Tier2 walker, and the C++/assembly
//                 native artifacts over the same seeded inputs, with a
//                 bit-exact gate before any timing and Rule 49 stats
//                 (median/min/max); skipped paths are reported, never
//                 guessed (Rule 30).
// Deterministic (Rule 53); every fallback is reported (Rule 30).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "mlk/backend/asm_emitter.h"
#include "mlk/backend/backend_driver.h"
#include "mlk/backend/cpp_emitter.h"
#include "mlk/fastkernel/fast_kernel.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"
#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/telemetry.h"
#include "mlk/core/small_vector.h"
#include "mlk/support/json.h"
#include "mlk/type/domain_profile.h"

#include "poly_bench.h"

namespace {

std::string readFile(const char* path, bool& ok) {
    ok = false;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return {};
    std::string data;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
    std::fclose(f);
    ok = true;
    return data;
}

/// Builds a MatMul graph: C[8,7] = A[8,6] * B[6,7].
mlk::MathGraph buildGemmGraph(mlk::SymbolTable& symbols) {
    mlk::GraphBuilder b(symbols);  // builds into its internal graph_
    const auto tA = mlk::MathType::tensorValue(mlk::Dtype::F64, {8, 6});
    const auto tB = mlk::MathType::tensorValue(mlk::Dtype::F64, {6, 7});
    const mlk::ValueId pa = b.placeholder("A", tA);
    const mlk::ValueId pb = b.placeholder("B", tB);
    auto mm = b.op(mlk::MathOp::MatMul, {pa, pb});
    if (mm.has_value()) b.output(*mm);
    return b.graph();
}

/// Builds an M*K*N MatMul graph for the autotune verb (the searched
/// specification s; Axiom 14.2).
mlk::MathGraph buildGemmGraphShaped(mlk::SymbolTable& symbols,
                                    const int64_t m, const int64_t k,
                                    const int64_t n) {
    mlk::GraphBuilder b(symbols);
    const auto tA = mlk::MathType::tensorValue(mlk::Dtype::F64,
                                               {m, k});
    const auto tB = mlk::MathType::tensorValue(mlk::Dtype::F64,
                                               {k, n});
    const mlk::ValueId pa = b.placeholder("A", tA);
    const mlk::ValueId pb = b.placeholder("B", tB);
    auto mm = b.op(mlk::MathOp::MatMul, {pa, pb});
    if (mm.has_value()) b.output(*mm);
    return b.graph();
}

/// The tensor CPU domain profile shared by the demo and emit verbs.
mlk::MathDomainProfile tensorCpuProfile(mlk::SymbolTable& symbols) {
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("tensor_f64_cpu");
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    profile.capabilities.set(mlk::Capability::HasTensorDomain);
    profile.capabilities.set(mlk::Capability::HasKernelFusion);
    profile.capabilities.set(mlk::Capability::HasAlgebraicRewriting);
    profile.capabilities.set(mlk::Capability::HasDerivatives);
    profile.capabilities.set(mlk::Capability::HasApproximation);
    return profile;
}

mlk::KernelModule compileAt(mlk::MathGraph& graph, mlk::SymbolTable& symbols,
                            mlk::MathDomainProfile& profile,
                            mlk::DiagnosticEngine& diag,
                            mlk::TelemetrySink& telemetry, mlk::Tier tier,
                            bool& ok) {
    mlk::PipelineRunner runner(symbols, &telemetry);
    mlk::KernelModule kernel;
    mlk::PassContext ctx;
    ctx.domainProfile = &profile;
    ctx.diag = &diag;
    ctx.symbols = &symbols;
    ctx.tier = tier;
    ctx.kernelOut = &kernel;
    auto r = runner.run(tier, ctx, graph, &kernel);
    ok = r.has_value();
    if (!ok) {
        std::fprintf(stderr, "mlk-poly: compile failed: %s\n",
                     r.error().message.c_str());
    }
    return kernel;
}

int runDemo(const std::string& backendFlag) {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::DiagnosticEngine diag;
    mlk::TelemetrySink telemetry;
    mlk::MathGraph graph = buildGemmGraph(symbols);
    mlk::AccuracyContract contract;
    mlk::MathDomainProfile profile = tensorCpuProfile(symbols);

    bool ok = false;
    mlk::KernelModule baseline =
        compileAt(graph, symbols, profile, diag, telemetry, mlk::Tier::Tier1,
                  ok);
    if (!ok) return 1;
    mlk::KernelModule poly =
        compileAt(graph, symbols, profile, diag, telemetry, mlk::Tier::Tier2,
                  ok);
    if (!ok) return 1;

    std::printf("== mlk-poly report ==\n");
    std::printf("baseline (Tier1): %zu nodes\n",
                static_cast<std::size_t>(baseline.nodes.size()));
    std::printf("polyhedral (Tier2): %zu nodes\n",
                static_cast<std::size_t>(poly.nodes.size()));
    std::printf("\n-- polyhedral kernel JSON --\n%s\n",
                mlk::json::serializePretty(poly.toJson(symbols)).c_str());

    // Execute baseline vs transformed over dense buffers.
    constexpr int64_t M = 8, K = 6, N = 7;
    std::string err;
    (void)err;
    mlk::KernelBufferBindings ioBase, ioPoly;
    mlk::SmallVector<double, 8> bufA(static_cast<std::size_t>(M * K));
    mlk::SmallVector<double, 8> bufB(static_cast<std::size_t>(K * N));
    mlk::SmallVector<double, 8> c1(static_cast<std::size_t>(M * N), 0.0);
    mlk::SmallVector<double, 8> c2(static_cast<std::size_t>(M * N), 0.0);
    for (std::size_t i = 0; i < bufA.size(); ++i) {
        bufA[i] = static_cast<double>((i * 7) % 13) * 0.25;
    }
    for (std::size_t i = 0; i < bufB.size(); ++i) {
        bufB[i] = static_cast<double>((i * 5) % 11) * 0.5;
    }
    ioBase.inputs.push_back(bufA.data());
    ioBase.inputs.push_back(bufB.data());
    ioBase.outputs.push_back(c1.data());
    ioBase.elements = M * K;
    ioPoly = ioBase;
    ioPoly.outputs[0] = c2.data();

    auto r1 = mlk::executeKernelOnBuffers(baseline, symbols, ioBase, nullptr);
    auto r2 = mlk::executeKernelOnBuffers(poly, symbols, ioPoly, nullptr);
    if (!r1.has_value() || !r2.has_value()) {
        std::fprintf(stderr, "mlk-poly: execution failed\n");
        return 1;
    }
    double maxDiff = 0.0;
    for (std::size_t i = 0; i < c1.size(); ++i) {
        const double d = std::fabs(c1[i] - c2[i]);
        if (d > maxDiff) maxDiff = d;
    }
    std::printf("executed baseline vs transformed: %zu outputs, max|diff| = "
                "%g\n",
                c1.size(), maxDiff);
    if (maxDiff != 0.0) {
        std::printf("mlk-poly: DIFF detected\n");
        return 1;
    }
    std::printf("mlk-poly: PASS (walker bit-exact)\n");

    // Native artifact path (opt-in): lower the transformed kernel to
    // assembly or C++, build it out-of-process, load, execute, and
    // compare against the walker bit-exactly (ADR-0003/0006).
    if (!backendFlag.empty()) {
        const mlk::ArtifactKind kind =
            backendFlag == "cpp" ? mlk::ArtifactKind::Cpp
                                 : mlk::ArtifactKind::Asm;
        std::printf("\n-- native artifact (%s) --\n",
                    backendFlag.c_str());
        mlk::BackendDriverConfig cfg;
        auto loaded = mlk::buildKernelArtifact(poly, symbols, kind, cfg);
        if (!loaded.has_value()) {
            std::fprintf(stderr, "mlk-poly: artifact failed: %s\n",
                         loaded.error().message.c_str());
            return 1;
        }
        std::printf("artifact: %s\nlibrary:  %s\n",
                    loaded->artifactPath().c_str(),
                    loaded->libraryPath().c_str());
        mlk::SmallVector<double, 8> c3(static_cast<std::size_t>(M * N),
                                       0.0);
        mlk::KernelBufferBindings ioNative = ioBase;
        ioNative.outputs[0] = c3.data();
        auto r3 = loaded->run(poly, symbols, ioNative);
        if (!r3.has_value()) {
            std::fprintf(stderr, "mlk-poly: native run failed: %s\n",
                         r3.error().message.c_str());
            return 1;
        }
        double nativeDiff = 0.0;
        for (std::size_t i = 0; i < c3.size(); ++i) {
            const double d = std::fabs(c3[i] - c2[i]);
            if (d > nativeDiff) nativeDiff = d;
        }
        std::printf("executed native vs walker: %zu outputs, max|diff| = "
                    "%g\n",
                    c3.size(), nativeDiff);
        if (nativeDiff != 0.0) {
            std::printf("mlk-poly: NATIVE DIFF detected\n");
            return 1;
        }
        std::printf("mlk-poly: PASS (native bit-exact)\n");
    }
    return 0;
}

/// Compiles a graph file at Tier2 (the `show` pipeline) and emits the
/// standalone artifact text; --compile additionally builds and loads the
/// shared object out-of-process and prints the library path.
int runEmit(const std::string& path, const bool useAsm,
            const std::string& outPath, const std::string& workdir,
            const bool compile) {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::DiagnosticEngine diag;
    bool ok = false;
    const std::string text = readFile(path.c_str(), ok);
    if (!ok) {
        std::fprintf(stderr, "mlk-poly: cannot read %s\n", path.c_str());
        return 2;
    }
    auto graph = mlk::parseGraphFile(text, symbols);
    if (!graph.has_value()) {
        std::fprintf(stderr, "mlk-poly: parse error: %s\n",
                     graph.error().message.c_str());
        return 2;
    }
    mlk::TelemetrySink telemetry;
    mlk::AccuracyContract contract;
    mlk::MathDomainProfile profile = tensorCpuProfile(symbols);
    mlk::KernelModule kernel;
    mlk::PipelineRunner runner(symbols, &telemetry);
    mlk::PassContext ctx;
    ctx.domainProfile = &profile;
    ctx.accuracy = &contract;
    ctx.diag = &diag;
    ctx.symbols = &symbols;
    ctx.tier = mlk::Tier::Tier2;
    ctx.kernelOut = &kernel;
    auto r = runner.run(mlk::Tier::Tier2, ctx, *graph, &kernel);
    if (!r.has_value()) {
        std::fprintf(stderr, "mlk-poly: compile failed: %s\n",
                     r.error().message.c_str());
        return 1;
    }
    auto src = useAsm ? mlk::emitAsmSource(kernel, symbols)
                      : mlk::emitCppSource(kernel, symbols);
    if (!src.has_value()) {
        std::fprintf(stderr, "mlk-poly: emit failed: %s\n",
                     src.error().message.c_str());
        return 1;
    }
    if (!outPath.empty()) {
        FILE* f = std::fopen(outPath.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "mlk-poly: cannot write %s\n",
                         outPath.c_str());
            return 1;
        }
        std::fwrite(src->data(), 1, src->size(), f);
        std::fclose(f);
        std::printf("mlk-poly: artifact written to %s\n", outPath.c_str());
    } else {
        std::fwrite(src->data(), 1, src->size(), stdout);
    }
    if (compile) {
        mlk::BackendDriverConfig cfg;
        if (!workdir.empty()) cfg.workdirBase = workdir;
        auto loaded = mlk::buildKernelArtifact(
            kernel, symbols,
            useAsm ? mlk::ArtifactKind::Asm : mlk::ArtifactKind::Cpp, cfg);
        if (!loaded.has_value()) {
            std::fprintf(stderr, "mlk-poly: artifact failed: %s\n",
                         loaded.error().message.c_str());
            return 1;
        }
        std::printf("mlk-poly: compiled artifact: %s\n",
                    loaded->artifactPath().c_str());
        std::printf("mlk-poly: shared object:    %s\n",
                    loaded->libraryPath().c_str());
    }
    return 0;
}


/// Parses a comma-separated int64 list ("0,16,32").
bool parseIntList(const std::string& s,
                  mlk::SmallVector<int64_t, 8>& out) {
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t next = s.find(',', pos);
        const std::string tok = s.substr(
            pos, next == std::string::npos ? std::string::npos : next - pos);
        if (tok.empty()) return false;
        char* end = nullptr;
        const long long v = std::strtoll(tok.c_str(), &end, 10);
        if (end == nullptr || *end != '\0') return false;
        out.push_back(static_cast<int64_t>(v));
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return !out.empty();
}

/// The OEIA/Refined/FastKernels budgeted search over a GEMM variant
/// space (Axioms 14.1-15.10); every claim in the printed summary traces
/// to a certificate in the JSON report.
int runAutotune(int argc, char** argv) {
    int64_t m = 64, k = 48, n = 56;
    mlk::SmallVector<int64_t, 8> tiles{0, 16, 32};
    bool execSet = false;
    std::string execArg;
    mlk::fastkernel::FastKernelSearchConfig cfg;
    std::string jsonOut, cacheDir, workdir;
    double comptimeBudgetMs = 0.0, extraBudgetMs = 0.0, tauMs = 0.0,
           alpha = 1.0;
    int64_t executions = 1;
    std::string mode = "same";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&a](const char* key) -> std::string {
            const std::string prefix = std::string("--") + key + "=";
            return a.rfind(prefix, 0) == 0 ? a.substr(prefix.size())
                                           : std::string();
        };
        if (!val("m").empty()) {
            m = std::atoll(val("m").c_str());
        } else if (!val("k").empty()) {
            k = std::atoll(val("k").c_str());
        } else if (!val("n").empty()) {
            n = std::atoll(val("n").c_str());
        } else if (!val("tiles").empty()) {
            tiles.clear();
            if (!parseIntList(val("tiles"), tiles)) {
                std::fprintf(stderr, "mlk-poly: bad --tiles\n");
                return 2;
            }
        } else if (!val("exec").empty()) {
            execSet = true;
            execArg = val("exec");
        } else if (!val("budget-mode").empty()) {
            mode = val("budget-mode");
        } else if (!val("comptime-budget-ms").empty()) {
            comptimeBudgetMs = std::atof(val("comptime-budget-ms").c_str());
        } else if (!val("extra-budget-ms").empty()) {
            extraBudgetMs = std::atof(val("extra-budget-ms").c_str());
        } else if (!val("tau-ms").empty()) {
            tauMs = std::atof(val("tau-ms").c_str());
        } else if (!val("alpha").empty()) {
            alpha = std::atof(val("alpha").c_str());
        } else if (!val("executions").empty()) {
            executions = std::atoll(val("executions").c_str());
        } else if (!val("warmup").empty()) {
            cfg.warmupReps =
                static_cast<uint32_t>(std::atoll(val("warmup").c_str()));
        } else if (!val("reps").empty()) {
            cfg.benchReps =
                static_cast<uint32_t>(std::atoll(val("reps").c_str()));
        } else if (!val("cache-dir").empty()) {
            cacheDir = val("cache-dir");
        } else if (!val("workdir").empty()) {
            workdir = val("workdir");
        } else if (!val("json-out").empty()) {
            jsonOut = val("json-out");
        } else {
            std::fprintf(stderr, "mlk-poly: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (m <= 0 || k <= 0 || n <= 0) {
        std::fprintf(stderr, "mlk-poly: --m/--k/--n must be positive\n");
        return 2;
    }
    if (mode != "same" && mode != "extra" && mode != "amortized") {
        std::fprintf(stderr,
                     "mlk-poly: --budget-mode must be same|extra|amortized\n");
        return 2;
    }
    if (execSet) {
        mlk::SmallVector<mlk::fastkernel::ExecPath, 4> paths{};
        std::size_t pos = 0;
        while (pos <= execArg.size()) {
            const std::size_t next = execArg.find(',', pos);
            const std::string tok = execArg.substr(
                pos,
                next == std::string::npos ? std::string::npos : next - pos);
            if (tok == "walker") {
                paths.push_back(mlk::fastkernel::ExecPath::Walker);
            } else if (tok == "asm") {
                paths.push_back(mlk::fastkernel::ExecPath::NativeAsm);
            } else if (tok == "cpp") {
                paths.push_back(mlk::fastkernel::ExecPath::NativeCpp);
            } else if (!tok.empty()) {
                std::fprintf(stderr, "mlk-poly: bad --exec token %s\n",
                             tok.c_str());
                return 2;
            }
            if (next == std::string::npos) break;
            pos = next + 1;
        }
        if (paths.empty()) {
            std::fprintf(stderr, "mlk-poly: --exec must be non-empty\n");
            return 2;
        }
        cfg.execPaths = std::move(paths);
    }
    cfg.tileSizes = std::move(tiles);
    cfg.budget.mode = mode == "same"
                          ? mlk::fastkernel::BudgetMode::SameComptime
                          : mode == "extra"
                                ? mlk::fastkernel::BudgetMode::BoundedExtra
                                : mlk::fastkernel::BudgetMode::Amortized;
    cfg.budget.baselineComptimeSec = comptimeBudgetMs / 1000.0;
    cfg.budget.extraComptimeMaxSec = extraBudgetMs / 1000.0;
    cfg.budget.minRuntimeImprovementSec = tauMs / 1000.0;
    cfg.budget.alpha = alpha;
    cfg.budget.amortizedExecutions = executions;
    cfg.driver.workdirBase = workdir;
    cfg.cacheDir = cacheDir;

    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::DiagnosticEngine diag;
    mlk::TelemetrySink telemetry;
    mlk::MathGraph graph = buildGemmGraphShaped(symbols, m, k, n);
    mlk::AccuracyContract contract;
    mlk::MathDomainProfile profile = tensorCpuProfile(symbols);

    auto report =
        mlk::fastkernel::runFastKernelSearch(graph, symbols, profile, cfg);
    if (!report.has_value()) {
        std::fprintf(stderr, "mlk-poly: autotune failed: %s\n",
                     report.error().message.c_str());
        return 1;
    }

    // Human summary (every line traces to a certificate in the JSON).
    const mlk::fastkernel::FastKernelReport& r = *report;
    std::printf("== mlk-poly fast-kernel search ==\n");
    std::printf("policy: %s\nenvironment: %s\n",
                mlk::fastkernel::kFastKernelPolicy,
                r.env.descriptor.c_str());
    std::printf("budget mode: %s (B_K^0 = %.3f ms, %s)\n",
                mlk::fastkernel::budgetModeName(cfg.budget.mode),
                r.baselineComptimeSec * 1000.0,
                r.budgetProvenance.c_str());
    if (r.rooflineLowerSec > 0.0) {
        std::printf("roofline lower bound: %.3f ms (B_min=%lld B, "
                    "O_min=%lld ops); winner gap: %.2fx over L\n",
                    r.rooflineLowerSec * 1000.0,
                    static_cast<long long>(r.work.minBytes),
                    static_cast<long long>(r.work.minOps),
                    1.0 + r.winnerGapOverLower);
    }
    std::printf("\n%-24s %-11s %-12s %-10s %-9s %s\n",
                "variant(tile/exec/thr)", "identity", "U_run(ms)",
                "M(ms)", "objective", "status");
    for (const auto& o : r.candidates) {
        char row[256];
        if (o.admissible) {
            std::snprintf(row, sizeof(row), "%lld/%s/%u",
                          static_cast<long long>(o.config.tileSize),
                          mlk::fastkernel::execPathName(o.config.exec),
                          o.config.threads);
            std::printf("%-24s %-11s %-12.3f %-10.3f %-9.3f %s\n", row,
                        o.cert.identityCertified ? "bit-exact" : "-",
                        o.cert.runtimeUpperBoundSec * 1000.0,
                        o.cert.comptimeTotalSec * 1000.0,
                        o.objectiveSec * 1000.0, "admissible");
        } else {
            std::snprintf(row, sizeof(row), "%lld/%s/%u",
                          static_cast<long long>(o.config.tileSize),
                          mlk::fastkernel::execPathName(o.config.exec),
                          o.config.threads);
            std::printf("%-24s %-11s %-12s %-10s %-9s %s: %s\n", row, "-",
                        "-", "-", "-",
                        mlk::fastkernel::rejectReasonName(o.reject),
                        o.rejectDetail.c_str());
        }
    }
    if (r.winnerIndex >= 0) {
        const auto& w = r.candidates[static_cast<std::size_t>(
            r.winnerIndex)];
        std::printf("\nwinner: tile=%lld exec=%s threads=%u "
                    "(U_run = %.3f ms, speedup vs spec = %.2fx, "
                    "cache hits=%u misses=%u)\n",
                    static_cast<long long>(w.config.tileSize),
                    mlk::fastkernel::execPathName(w.config.exec),
                    w.config.threads,
                    w.cert.runtimeUpperBoundSec * 1000.0,
                    report->baselineRuntimeSec > 0.0
                        ? report->baselineRuntimeSec /
                              w.cert.runtimeUpperBoundSec
                        : 0.0,
                    r.cacheHits, r.cacheMisses);
    }
    std::printf("claim: %s\n", r.claim.c_str());

    if (!jsonOut.empty()) {
        FILE* f = std::fopen(jsonOut.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "mlk-poly: cannot write %s\n",
                         jsonOut.c_str());
            return 1;
        }
        const std::string text = mlk::json::serializePretty(r.toJson());
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
        std::printf("report: %s\n", jsonOut.c_str());
    }
    return r.budgetFailure ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("usage: mlk-poly demo [--backend=asm|cpp]\n"
                   "       mlk-poly show <graph.mlk>\n"
                   "       mlk-poly autotune [--m=M --k=K --n=N] "
                   "[--tiles=..] [--exec=..] [--budget-mode=..]\n"
                   "                         [--comptime-budget-ms=X] "
                   "[--extra-budget-ms=X] [--tau-ms=X] [--json-out=F]\n"
                   "       mlk-poly emit <graph.mlk> [--asm|--cpp] "
                   "[--out=<path>] [--workdir=<dir>] [--compile]\n",
                   stderr);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "demo") {
        std::string backend;
        for (int i = 2; i < argc; ++i) {
            if (std::strncmp(argv[i], "--backend=", 10) == 0) {
                backend = argv[i] + 10;
            } else {
                std::fprintf(stderr, "mlk-poly: unknown option %s\n",
                             argv[i]);
                return 2;
            }
        }
        if (!backend.empty() && backend != "asm" && backend != "cpp") {
            std::fprintf(stderr, "mlk-poly: --backend must be asm|cpp\n");
            return 2;
        }
        return runDemo(backend);
    }
    if (mode == "autotune") {
        return runAutotune(argc, argv);
    }
    if (mode == "bench") {
        return polybench::runBench(argc, argv);
    }
    if (mode == "emit" && argc >= 3) {
        std::string outPath, workdir;
        bool useAsm = true;   // the assembly form is the default artifact
        bool compile = false;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--asm") {
                useAsm = true;
            } else if (a == "--cpp") {
                useAsm = false;
            } else if (a.rfind("--out=", 0) == 0) {
                outPath = a.substr(6);
            } else if (a.rfind("--workdir=", 0) == 0) {
                workdir = a.substr(10);
            } else if (a == "--compile") {
                compile = true;
            } else {
                std::fprintf(stderr, "mlk-poly: unknown option %s\n",
                             argv[i]);
                return 2;
            }
        }
        return runEmit(argv[2], useAsm, outPath, workdir, compile);
    }
    if (mode == "show" && argc >= 3) {
        mlk::SymbolTable symbols;
        mlk::passes::registerAllPasses(symbols);
        mlk::DiagnosticEngine diag;
        bool ok = false;
        const std::string text = readFile(argv[2], ok);
        if (!ok) {
            std::fprintf(stderr, "mlk-poly: cannot read %s\n", argv[2]);
            return 2;
        }
        auto graph = mlk::parseGraphFile(text, symbols);
        if (!graph.has_value()) {
            std::fprintf(stderr, "mlk-poly: parse error: %s\n",
                         graph.error().message.c_str());
            return 2;
        }
        mlk::TelemetrySink telemetry;
        mlk::AccuracyContract contract;
        mlk::MathDomainProfile profile;
        profile.name = symbols.intern("default");
        mlk::KernelModule kernel;
        mlk::PipelineRunner runner(symbols, &telemetry);
        mlk::PassContext ctx;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier2;
        ctx.kernelOut = &kernel;
        auto r = runner.run(mlk::Tier::Tier2, ctx, *graph, &kernel);
        if (!r.has_value()) {
            std::fprintf(stderr, "mlk-poly: compile failed: %s\n",
                         r.error().message.c_str());
            return 1;
        }
        std::printf("== mlk-poly report ==\n%s\n",
                    mlk::json::serializePretty(kernel.toJson(symbols)).c_str());
        return 0;
    }
    std::fputs("usage: mlk-poly demo [--backend=asm|cpp] | show "
               "<graph.mlk> | autotune [...]\n"
               " | bench [--reps=R] [--warmup=W] [--suites=...] "
               "[--json-out=F] [--workdir=D]\n"
               " | emit <graph.mlk> [--asm|--cpp] [--out=<path>] "
               "[--workdir=<dir>] [--compile]\n",
               stderr);
    return 2;
}
