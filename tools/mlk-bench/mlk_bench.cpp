// mlk-bench — benchmark harness driver (Rule 49 protocol).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mlk/autotune/benchmarker.h"
#include "mlk/core/constants.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"
#include "mlk/runtime/execution.h"
#include "mlk/runtime/telemetry.h"

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
    if (argc < 3 || std::strcmp(argv[1], "run") != 0) {
        std::fputs("usage: mlk-bench run <graph.mlk> [--reps=30] [--warmup=5]\n",
                   stderr);
        return 2;
    }
    uint32_t reps = mlk::constants::kBenchDefaultReps;
    uint32_t warmup = mlk::constants::kBenchDefaultWarmup;
    for (int i = 3; i < argc; ++i) {
        if (std::strncmp(argv[i], "--reps=", 7) == 0) {
            reps = static_cast<uint32_t>(std::atoi(argv[i] + 7));
        } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
            warmup = static_cast<uint32_t>(std::atoi(argv[i] + 9));
        }
    }
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    mlk::TelemetrySink telemetry;
    bool ok = false;
    const std::string text = readFile(argv[2], ok);
    if (!ok) return 2;
    auto graph = mlk::parseGraphFile(text, symbols);
    if (!graph.has_value()) {
        std::fprintf(stderr, "mlk-bench: parse error: %s\n",
                     graph.error().message.c_str());
        return 2;
    }
    mlk::BenchmarkProtocol p;
    p.reps = reps;
    p.warmup = warmup;
    auto m = mlk::benchmarkGraph(*graph, symbols, p);
    if (!m.has_value()) {
        std::fprintf(stderr, "mlk-bench: failed: %s\n",
                     m.error().message.c_str());
        return 1;
    }
    std::printf("mlk-bench: min=%.4f ms median=%.4f ms stddev=%.4f ms "
                "ops/s=%.2f reps=%u\n",
                m->minMs, m->medianMs, m->stddevMs, m->opsPerSecond, m->reps);
    return 0;
}
