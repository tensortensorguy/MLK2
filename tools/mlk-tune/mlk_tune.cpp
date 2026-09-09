// mlk-tune — autotuner driver (layout spec §13):
//   mlk-tune tune graph.mlk --budget=medium
#include <cstdio>
#include <cstring>
#include <string>

#include "mlk/autotune/searcher.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"
#include "mlk/runtime/cache.h"
#include "mlk/runtime/telemetry.h"
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
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[1], "tune") != 0) {
        std::fputs("usage: mlk-tune tune <graph.mlk>\n", stderr);
        return 2;
    }
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::TelemetrySink telemetry;
    bool ok = false;
    const std::string text = readFile(argv[2], ok);
    if (!ok) return 2;
    auto graph = mlk::parseGraphFile(text, symbols);
    if (!graph.has_value()) {
        std::fprintf(stderr, "mlk-tune: parse error: %s\n",
                     graph.error().message.c_str());
        return 2;
    }

    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("scalar_f64");
    profile.capabilities.set(mlk::Capability::HasNumericValues);
    profile.capabilities.set(mlk::Capability::HasFloatingPoint);
    profile.capabilities.set(mlk::Capability::HasAutotuning);
    profile.approximation.allowApproximation = false;
    mlk::AccuracyContract contract;
    mlk::TuningContext ctx;
    ctx.profile = &profile;
    ctx.accuracy = &contract;
    ctx.hardware = mlk::HardwareInfo::detect();

    mlk::RealizationCache cache(symbols);
    mlk::Autotuner tuner(symbols, telemetry);
    auto entry = tuner.tune(*graph, ctx, cache);
    if (!entry.has_value()) {
        std::fprintf(stderr, "mlk-tune: failed: %s\n",
                     entry.error().message.c_str());
        return 1;
    }
    std::printf("mlk-tune: winner strategy=%s measured=%.4f ms\n",
                symbols.text(entry->strategy).c_str(), entry->measuredMs);
    std::printf("%s\n",
                mlk::json::serializePretty(entry->toJson(symbols)).c_str());
    return 0;
}
