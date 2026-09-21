// mlkc — MLK+ compiler driver (layout spec §13):
//   mlkc compile graph.mlk --tier=2 --profile=tensor_f32_cpu [--emit=...]
// Deterministic, cancellable; falls back gracefully (Rule 139); never
// opaque in its diagnostics (Rule 67).
#include <cstdlib>
#include <cstring>
#include <string>

#include "mlk/autotune/searcher.h"
#include "mlk/backend/cpp_emitter.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/register_all.h"
#include "mlk/pipeline/pipeline_runner.h"
#include "mlk/runtime/cache.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/telemetry.h"
#include "mlk/superopt/enumerator.h"
#include "mlk/type/domain_profile.h"
#include "mlk/verifier/graph_verifier.h"

namespace {

int usage() {
    std::fputs(
        "mlkc — MLK+ compiler driver\n"
        "usage:\n"
        "  mlkc compile <graph.mlk> --tier=0..3 --profile=<name> "
        "[--emit=ir|kernel|cpp] [--out=<path>] [--telemetry[=path]]\n"
        "  mlkc run     <graph.mlk> --tier=0..3 --profile=<name> "
        "[--x1=1.0 --x2=2.0 ...]\n"
        "  mlkc list-passes\n"
        "telemetry: --telemetry dumps the structured event stream "
        "(schemas/telemetry.schema.json) to <path> or stderr; pipe it "
        "into mlk-profile for aggregation.\n",
        stderr);
    return 2;
}

mlk::Tier parseTier(const std::string& s, bool& ok) {
    ok = true;
    if (s == "0" || s == "tier0") return mlk::Tier::Tier0;
    if (s == "1" || s == "tier1") return mlk::Tier::Tier1;
    if (s == "2" || s == "tier2") return mlk::Tier::Tier2;
    if (s == "3" || s == "tier3") return mlk::Tier::Tier3;
    ok = false;
    return mlk::Tier::Tier1;
}

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

int runCompile(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* graphPath = argv[2];
    std::string tierArg = "1";
    std::string profileName = "scalar_f64";
    std::string emit = "ir";
    std::string outPath;
    std::string telemetryPath;
    bool dumpTelemetry = false;
    for (int i = 3; i < argc; ++i) {
        const char* a = argv[i];
        auto starts = [&](const char* p) {
            return std::strncmp(a, p, std::strlen(p)) == 0;
        };
        if (starts("--tier=")) tierArg = a + 7;
        else if (starts("--profile=")) profileName = a + 10;
        else if (starts("--emit=")) emit = a + 7;
        else if (starts("--out=")) outPath = a + 6;
        else if (starts("--telemetry=")) {
            dumpTelemetry = true;
            telemetryPath = a + 12;
        } else if (std::strcmp(a, "--telemetry") == 0) {
            dumpTelemetry = true;
        }
    }
    bool ok = false;
    const mlk::Tier tier = parseTier(tierArg, ok);
    if (!ok) {
        std::fprintf(stderr, "mlkc: invalid tier '%s'\n", tierArg.c_str());
        return 2;
    }

    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::TelemetrySink telemetry;
    mlk::DiagnosticEngine diag;

    const std::string text = readFile(graphPath, ok);
    if (!ok) {
        std::fprintf(stderr, "mlkc: cannot read %s\n", graphPath);
        return 2;
    }
    mlk::MathGraph graph{&symbols};
    {
        auto parsed = mlk::parseGraphFile(text, symbols);
        if (!parsed.has_value()) {
            std::fprintf(stderr, "mlkc: parse error: %s\n",
                         parsed.error().message.c_str());
            return 2;
        }
        graph = std::move(*parsed);
    }

    // Domain profile: MVP ships embedded defaults (hermetic; Rule 160);
    // file-based profiles load through mlk-generate-profiles + cache.
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern(profileName);
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    profile.capabilities.set(mlk::Capability::HasTensorDomain);
    profile.capabilities.set(mlk::Capability::HasCalculus);
    profile.capabilities.set(mlk::Capability::HasDerivatives);
    profile.capabilities.set(mlk::Capability::HasAlgebraicRewriting);
    profile.capabilities.set(mlk::Capability::HasKernelFusion);
    profile.capabilities.set(mlk::Capability::HasCPUBackend);
    profile.lawCommutativeMul = mlk::TriState::True;
    // FP reassociation stays FORBIDDEN by default (Rule 33/90).
    profile.lawAssociativeAdd = mlk::TriState::Unknown;
    mlk::AccuracyContract contract;

    mlk::KernelModule kernel;
    mlk::PipelineRunner runner(symbols, &telemetry);
    mlk::PassContext ctx;
    ctx.domainProfile = &profile;
    ctx.accuracy = &contract;
    ctx.diag = &diag;
    ctx.telemetry = &telemetry;
    ctx.symbols = &symbols;
    ctx.tier = tier;
    ctx.kernelOut = &kernel;

    mlk::PassResult result;
    {
        auto r = runner.run(tier, ctx, graph, &kernel);
        if (!r.has_value()) {
            std::fprintf(stderr, "mlkc: compilation failed: %s\n",
                         r.error().message.c_str());
            // Rule 139: graceful degradation — the artifact is rejected,
            // nothing installed; exit code signals failure to the caller.
            return 1;
        }
        result = *r;
    }

    if (emit == "ir") {
        mlk::PrintOptions opts;
        opts.annotateTypes = true;
        std::printf("%s\n", mlk::printGraph(graph, opts, symbols).c_str());
    } else if (emit == "kernel") {
        std::printf("%s\n",
                    mlk::json::serializePretty(kernel.toJson(symbols)).c_str());
    } else if (emit == "cpp") {
        auto src = mlk::emitCppSource(kernel, symbols);
        if (!src.has_value()) {
            std::fprintf(stderr, "mlkc: emit failed: %s\n",
                         src.error().message.c_str());
            return 1;
        }
        if (outPath.empty()) {
            std::fputs(src->c_str(), stdout);
        } else {
            FILE* f = std::fopen(outPath.c_str(), "wb");
            if (f == nullptr) return 2;
            std::fwrite(src->c_str(), 1, src->size(), f);
            std::fclose(f);
        }
    }
    std::fprintf(stderr,
                 "mlkc: compiled tier=%s profile=%s changed=%d nodes=%u\n",
                 mlk::tierName(tier), profileName.c_str(),
                 result.changed ? 1 : 0, result.nodesAfter);
    if (dumpTelemetry) {
        // Tool-boundary dump (Rule 157): the structured event stream, with
        // pass/reason resolved through THIS table (ids are table-scoped).
        const std::string json = mlk::json::serializePretty(
            telemetry.toJson(&symbols));
        if (telemetryPath.empty()) {
            std::fputs(json.c_str(), stderr);
            std::fputc('\n', stderr);
        } else {
            FILE* f = std::fopen(telemetryPath.c_str(), "wb");
            if (f == nullptr) {
                std::fprintf(stderr, "mlkc: cannot write %s\n",
                             telemetryPath.c_str());
                return 2;
            }
            std::fwrite(json.c_str(), 1, json.size(), f);
            std::fclose(f);
        }
    }
    return 0;
}

int runExec(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* graphPath = argv[2];
    std::string tierArg = "0";
    std::string profileName = "scalar_f64";
    mlk::SmallVector<double, 8> inputs;
    for (int i = 3; i < argc; ++i) {
        const char* a = argv[i];
        auto starts = [&](const char* p) {
            return std::strncmp(a, p, std::strlen(p)) == 0;
        };
        if (starts("--tier=")) tierArg = a + 7;
        else if (starts("--profile=")) profileName = a + 10;
        else if (starts("--x")) {
            // Accept both documented forms: --x2.0 and --x=2.0 (the
            // latter used to parse strtod("=2.0") == 0 silently).
            const char* v = a + 3;
            if (*v == '=') ++v;
            inputs.push_back(std::strtod(v, nullptr));
        }
    }
    bool ok = false;
    const mlk::Tier tier = parseTier(tierArg, ok);
    if (!ok) return 2;

    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::TelemetrySink telemetry;
    const std::string text = readFile(graphPath, ok);
    if (!ok) return 2;
    mlk::MathGraph graph{&symbols};
    {
        auto parsed = mlk::parseGraphFile(text, symbols);
        if (!parsed.has_value()) {
            std::fprintf(stderr, "mlkc: parse error: %s\n",
                         parsed.error().message.c_str());
            return 2;
        }
        graph = std::move(*parsed);
    }

    mlk::ExecutionEngine engine(symbols, telemetry);
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern(profileName);
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    mlk::AccuracyContract contract;
    auto r = engine.execute(graph, profile, contract, tier, inputs);
    if (!r.has_value()) {
        std::fprintf(stderr, "mlkc: execution failed: %s\n",
                     r.error().message.c_str());
        return 1;
    }
    for (std::size_t i = 0; i < r->outputScalars.size(); ++i) {
        std::printf("out[%zu] = %.17g\n", i, r->outputScalars[i]);
    }
    std::fprintf(stderr, "mlkc: executed tier=%s\n",
                 mlk::tierName(r->executedTier));
    return 0;
}

int runListPasses() {
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    for (const auto& entry : mlk::PassRegistry::instance().entries()) {
        std::printf("%-36s %-9s tiers=%zu\n",
                    symbols.text(entry.contract.name).c_str(),
                    mlk::passKindName(entry.contract.kind),
                    entry.contract.supportedTiers.size());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (std::strcmp(argv[1], "compile") == 0) return runCompile(argc, argv);
    if (std::strcmp(argv[1], "run") == 0) return runExec(argc, argv);
    if (std::strcmp(argv[1], "list-passes") == 0) return runListPasses();
    return usage();
}
