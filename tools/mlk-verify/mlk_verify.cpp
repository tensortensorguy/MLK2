// mlk-verify — verifier/proof checker driver (layout spec §13):
//   mlk-verify graph.mlk
#include <cstdio>
#include <cstring>
#include <string>

#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/pass/register_all.h"
#include "mlk/type/domain_profile.h"
#include "mlk/verifier/graph_verifier.h"

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
        std::fputs("usage: mlk-verify <graph.mlk>\n", stderr);
        return 2;
    }
    mlk::SymbolTable symbols;
    mlk::DiagnosticEngine diag;
    bool ok = false;
    const std::string text = readFile(argv[1], ok);
    if (!ok) {
        std::fprintf(stderr, "mlk-verify: cannot read %s\n", argv[1]);
        return 2;
    }
    auto graph = mlk::parseGraphFile(text, symbols);
    if (!graph.has_value()) {
        std::fprintf(stderr, "mlk-verify: parse error: %s\n",
                     graph.error().message.c_str());
        return 2;
    }
    mlk::MathDomainProfile profile;
    profile.name = symbols.intern("default");
    mlk::VerifyOptions opts;
    const bool passed = mlk::verifyGraph(*graph, &profile, opts, diag);
    for (const auto& d : diag.entries()) {
        std::fprintf(stderr, "mlk-verify: %s: %s", mlk::severityName(d.severity),
                     d.message.c_str());
        if (!d.rule.empty()) std::fprintf(stderr, " [%s]", d.rule.c_str());
        std::fprintf(stderr, "\n");
    }
    std::printf("mlk-verify: %s (%zu checks reported)\n",
                passed ? "PASS" : "FAIL", diag.entries().size());
    return passed ? 0 : 1;
}
