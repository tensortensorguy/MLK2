// Rule 44/152: fuzz the serialized graph loader — malformed inputs must be
// rejected safely with no crashes and no UB. Deterministic (seeded).
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_json.h"

#include "mlk_test.h"

#include <cstdint>
#include <string>

namespace {
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};
}  // namespace

MLK_TEST(fuzz, loader_survives_malformed_graphs) {
    mlk::SymbolTable symbols;
    Rng rng(0x5EED1234);
    const char* fragments[] = {
        "{", "}", "null", "[]", "{\"format\":\"mlk-graph\"}",
        "{\"format\":\"mlk-graph\",\"version\":1}",
        "{\"format\":\"mlk-graph\",\"version\":1,\"values\":{}}",
        "{\"format\":\"mlk-graph\",\"version\":1,\"values\":[],\"nodes\":5}",
        "{\"format\":\"mlk-graph\",\"version\":1,\"values\":[{\"id\":0,"
        "\"kind\":\"constant\"}]}",
        "{\"format\":\"mlk-graph\",\"version\":1,\"values\":[],"
        "\"nodes\":[{\"id\":0,\"op\":\"not_an_op\",\"inputs\":[]}]}",
        "{\"format\":\"mlk-graph\",\"version\":1,\"values\":[],"
        "\"nodes\":[{\"id\":0,\"op\":\"sin\",\"inputs\":[999]}]}",
        "{\"format\":\"mlk-graph\",\"version\":1,\"values\":[],"
        "\"nodes\":[],\"outputs\":[-5]}"};
    int accepted = 0;
    int rejected = 0;
    for (int iter = 0; iter < 500; ++iter) {
        std::string text;
        if (iter < static_cast<int>(sizeof(fragments) / sizeof(fragments[0]))) {
            text = fragments[iter];
        } else {
            // Mutate a valid document with random junk.
            text = "{\"format\":\"mlk-graph\",\"version\":1,\"values\":[";
            const int junk = static_cast<int>(rng.next() % 4);
            for (int j = 0; j < junk; ++j) {
                text += std::string("{\"id\":") +
                        std::to_string(static_cast<int>(rng.next() % 10)) +
                        ",\"kind\":\"" +
                        (rng.next() % 2 ? "constant" : "weird") + "\"},";
            }
            text += "],\"nodes\":[],\"outputs\":[]}";
            if (rng.next() % 3 == 0) {
                const std::size_t cut =
                    static_cast<std::size_t>(rng.next()) % text.size();
                text.resize(cut);  // truncate
            }
        }
        auto r = mlk::parseGraphFile(text, symbols);
        if (r.has_value()) {
            ++accepted;
        } else {
            ++rejected;
            // Rejections must carry actionable diagnostics (Rule 67).
            MLK_CHECK(!r.error().message.empty());
        }
    }
    MLK_CHECK(rejected > 0);
    (void)accepted;
}

MLK_TEST_MAIN("fuzz")
