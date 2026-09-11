// mlk-poly — polyhedral pipeline driver (layout spec §13):
//   mlk-poly demo   self-contained MatMul walkthrough: compiles a GEMM
//                   graph at Tier1 (baseline) and Tier2 (polyhedral),
//                   dumps both kernel JSONs, and executes baseline vs
//                   transformed over dense buffers with a max-diff report
//   mlk-poly show <graph.mlk>
//                   compiles any graph at Tier2 and dumps the kernel
//                   JSON (non-affine kernels report baseline fallback)
// Deterministic (Rule 53); every fallback is reported (Rule 30).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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

int runDemo() {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::DiagnosticEngine diag;
    mlk::TelemetrySink telemetry;
    mlk::MathGraph graph = buildGemmGraph(symbols);
    mlk::AccuracyContract contract;
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("tensor_f64_cpu");
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    profile.capabilities.set(mlk::Capability::HasTensorDomain);
    profile.capabilities.set(mlk::Capability::HasKernelFusion);
    profile.capabilities.set(mlk::Capability::HasAlgebraicRewriting);
    profile.capabilities.set(mlk::Capability::HasDerivatives);
    profile.capabilities.set(mlk::Capability::HasApproximation);

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
    std::printf(maxDiff == 0.0 ? "mlk-poly: PASS (bit-exact)\n"
                               : "mlk-poly: DIFF detected\n");
    return maxDiff == 0.0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("usage: mlk-poly demo | mlk-poly show <graph.mlk>\n",
                   stderr);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "demo") return runDemo();
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
    std::fputs("usage: mlk-poly demo | mlk-poly show <graph.mlk>\n", stderr);
    return 2;
}
