// MLK+ benchmark runner: executes suite JSON files via the Rule 49
// protocol (warmup, reps, min/median/stddev, noise detection).
#include "mlk/autotune/benchmarker.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"

#include <cstdio>
#include <cstring>
#include <string>

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
    if (argc < 2) {
        std::fputs("usage: mlk_bench_runner <graph.mlk> [--reps=30]\n", stderr);
        return 2;
    }
    mlk::SymbolTable symbols;
    mlk::passes::registerAllPasses(symbols);
    bool ok = false;
    const std::string text = readFile(argv[1], ok);
    if (!ok) return 2;
    auto graph = mlk::parseGraphFile(text, symbols);
    if (!graph.has_value()) return 2;
    mlk::BenchmarkProtocol p;
    auto m = mlk::benchmarkGraph(*graph, symbols, p);
    if (!m.has_value()) {
        std::fprintf(stderr, "bench failed: %s\n", m.error().message.c_str());
        return 1;
    }
    std::printf("min=%.4f ms median=%.4f ms stddev=%.4f ms ops/s=%.2f\n",
                m->minMs, m->medianMs, m->stddevMs, m->opsPerSecond);
    (void)argc; (void)argv;
    return 0;
}
