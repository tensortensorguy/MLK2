// Golden tests (Rule 42): checked-in input/expected IR pairs under
// tests/pass_golden/. Deterministic printer output.
#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/register_all.h"
#include "mlk/pass/pass_registry.h"
#include "mlk/type/domain_profile.h"

#include "mlk_test.h"

#include <cstdlib>
#include <string>

namespace {
std::string readFileCompat(const std::string& path, bool& ok) {
    ok = false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    std::string data;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
    std::fclose(f);
    ok = true;
    return data;
}
struct Golden {
    std::string name;
    std::string pass;
};
}  // namespace

MLK_TEST(golden, pass_golden_pairs) {
    const char* dir = std::getenv("MLK_TEST_DIR");
    MLK_CHECK(dir != nullptr);
    if (dir == nullptr) return;
    const std::string base = std::string(dir) + "/pass_golden";
    const Golden cases[] = {
        {"math/constant_fold/scalar_add_const", "math.constant_fold"},
        {"math/constant_fold/nan_mul", "math.constant_fold"},
        {"math/identity_elim/mul_one", "math.identity_elim"},
    };
    for (const auto& tc : cases) {
        mlk::SymbolTable symbols;
        mlk::passes::registerAllPasses(symbols);
        mlk::DiagnosticEngine diag;
        mlk::MathDomainProfile profile;
        profile.name = symbols.intern("golden");
        profile.capabilities.set(mlk::Capability::HasNumericValues);
        profile.capabilities.set(mlk::Capability::HasFloatingPoint);
        mlk::AccuracyContract contract;
        mlk::PassContext ctx;
        ctx.domainProfile = &profile;
        ctx.accuracy = &contract;
        ctx.diag = &diag;
        ctx.symbols = &symbols;
        ctx.tier = mlk::Tier::Tier1;

        bool ok = false;
        const std::string in = readFileCompat(base + "/" + tc.name + ".mlk", ok);
        MLK_CHECK(ok);
        if (!ok) continue;
        auto graph = mlk::parseGraphFile(in, symbols);
        MLK_CHECK(graph.has_value());
        if (!graph.has_value()) continue;
        // Realistic prefix: inference before transforms (spec section 9).
        MLK_CHECK(mlk::PassRegistry::instance()
                      .byName(symbols, symbols.intern("type.infer"))
                      ->run(ctx, *graph)
                      .has_value());
        MLK_CHECK(mlk::PassRegistry::instance()
                      .byName(symbols, symbols.intern("property.infer"))
                      ->run(ctx, *graph)
                      .has_value());
        mlk::Pass* pass =
            mlk::PassRegistry::instance().byName(symbols, symbols.intern(tc.pass));
        MLK_CHECK(pass != nullptr);
        if (pass == nullptr) continue;
        MLK_CHECK(pass->run(ctx, *graph).has_value());

        mlk::PrintOptions opts;
        opts.annotateTypes = true;
        const std::string actual =
            mlk::printGraph(*graph, opts, symbols);
        const std::string expected =
            readFileCompat(base + "/" + tc.name + ".expected.mlk", ok);
        MLK_CHECK(ok);
        if (!ok) continue;
        // Tolerate the trailing newline editors add; compare content.
        std::string expectedTrimmed = expected;
        while (!expectedTrimmed.empty() &&
               (expectedTrimmed.back() == '\n' ||
                expectedTrimmed.back() == '\r')) {
            expectedTrimmed.pop_back();
        }
        MLK_CHECK_EQ(actual, expectedTrimmed);
    }
}

MLK_TEST_MAIN("golden")
