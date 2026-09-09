// mlk-replay — replay failed compile/tune artifacts (Rule 45/158: debugging
// starts from replay, not reproduction).
#include <cstdio>
#include <cstring>
#include <string>

#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"
#include "mlk/runtime/graph_state.h"
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
    if (argc < 2) {
        std::fputs("usage: mlk-replay <artifact.json>\n", stderr);
        return 2;
    }
    mlk::SymbolTable symbols;
    bool ok = false;
    const std::string text = readFile(argv[1], ok);
    if (!ok) return 2;
    auto doc = mlk::json::parse(text);
    if (!doc.has_value()) {
        std::fprintf(stderr, "mlk-replay: malformed artifact\n");
        return 2;
    }
    // Rule 124: artifacts are untrusted; parse + validate or reject.
    if (const mlk::json::Value* gs = doc->find("graph_state")) {
        auto state = mlk::GraphState::fromJson(*gs);
        if (!state.has_value()) {
            std::fprintf(stderr, "mlk-replay: invalid GraphState: %s\n",
                         state.error().message.c_str());
            return 2;
        }
        std::printf("mlk-replay: GraphState ok (resume_node=%u, %zu "
                    "bindings)\n", state->resumeNode, state->bindings.size());
    }
    if (const mlk::json::Value* g = doc->find("graph")) {
        auto graph = mlk::graphFromJson(*g, symbols);
        std::printf("mlk-replay: graph %s\n",
                    graph.has_value() ? "restored" : "rejected");
    }
    return 0;
}
