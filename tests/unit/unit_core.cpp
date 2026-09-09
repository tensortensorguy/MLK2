// Core unit tests: SmallVector, SparseSet, BitVector, OpenHashMap, Flags,
// hash stability, SymbolTable, JSON (Rules 16-19, 24).
#include "mlk/core/bitvector.h"
#include "mlk/core/flags.h"
#include "mlk/core/hash.h"
#include "mlk/core/hash_map.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/sparse_set.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/json.h"

#include "mlk_test.h"

namespace {
enum class TestFlag : uint8_t { A, B, C };
}  // namespace

MLK_TEST(core, small_vector_inline) {
    mlk::SmallVector<int, 4> v;
    for (int i = 0; i < 4; ++i) v.push_back(i);
    MLK_CHECK_EQ(v.size(), 4u);
    MLK_CHECK_EQ(v[3], 3);
    v.push_back(4);  // spills to heap
    MLK_CHECK_EQ(v.size(), 5u);
    MLK_CHECK_EQ(v[4], 4);
    mlk::SmallVector<int, 4> copy = v;
    MLK_CHECK(copy == v);
    mlk::SmallVector<int, 4> moved = std::move(copy);
    MLK_CHECK(moved == v);
    v.eraseAt(0);
    MLK_CHECK_EQ(v[0], 1);
}

MLK_TEST(core, sparse_set) {
    mlk::SparseSet s(16);
    s.insert(3);
    s.insert(7);
    MLK_CHECK(s.contains(3));
    MLK_CHECK(s.contains(7));
    MLK_CHECK(!s.contains(4));
    s.remove(3);
    MLK_CHECK(!s.contains(3));
    MLK_CHECK(s.contains(7));
    MLK_CHECK_EQ(s.size(), 1u);
}

MLK_TEST(core, bitvector) {
    mlk::BitVector b(128);
    b.set(0);
    b.set(127);
    MLK_CHECK(b.test(0));
    MLK_CHECK(b.test(127));
    MLK_CHECK(!b.test(64));
    MLK_CHECK_EQ(b.popcount(), 2u);
    mlk::BitVector c(128);
    c.set(64);
    MLK_CHECK(b.unionWith(c));
    MLK_CHECK_EQ(b.popcount(), 3u);
}

MLK_TEST(core, open_hash_map) {
    mlk::OpenHashMap<uint32_t, uint32_t> m;
    for (uint32_t i = 0; i < 1000; ++i) {
        bool inserted = false;
        uint32_t* slot = m.findOrInsert(i, &inserted, i * 2);
        MLK_CHECK(inserted);
        MLK_CHECK_EQ(*slot, i * 2);
    }
    MLK_CHECK_EQ(m.size(), 1000u);
    for (uint32_t i = 0; i < 1000; ++i) {
        const uint32_t* v = m.find(i);
        MLK_CHECK(v != nullptr);
        MLK_CHECK_EQ(*v, i * 2);
    }
    m.remove(500);
    MLK_CHECK(m.find(500) == nullptr);
    MLK_CHECK(m.find(501) != nullptr);
}

MLK_TEST(core, flags) {
    mlk::Flags<TestFlag> f;
    MLK_CHECK(f.none());
    f.set(TestFlag::A);
    f.set(TestFlag::C);
    MLK_CHECK(f.test(TestFlag::A));
    MLK_CHECK(!f.test(TestFlag::B));
    MLK_CHECK(f.any());
    const auto g = f | mlk::Flags<TestFlag>{TestFlag::B};
    MLK_CHECK(g.allOf(mlk::Flags<TestFlag>{TestFlag::A}));
    MLK_CHECK(g.test(TestFlag::B));
}

MLK_TEST(core, hash_stability) {
    // Rule 24: hashes must be run-stable and content-derived.
    MLK_CHECK_EQ(mlk::hashText("mlk"), mlk::hashText("mlk"));
    MLK_CHECK_EQ(mlk::hashF64(3.14), mlk::hashF64(3.14));
    MLK_CHECK_EQ(mlk::hashF64(-0.0), mlk::hashF64(0.0));  // -0.0 normalized
    MLK_CHECK(mlk::hashText("a") != mlk::hashText("b"));
    const mlk::HashValue combined =
        mlk::hashCombine(mlk::hashU64(1), mlk::hashU64(2));
    MLK_CHECK_EQ(combined,
                 mlk::hashCombine(mlk::hashU64(1), mlk::hashU64(2)));
}

MLK_TEST(core, symbol_table_interning) {
    mlk::SymbolTable t;
    const mlk::SymbolId a = t.intern("matmul");
    const mlk::SymbolId b = t.intern("matmul");
    const mlk::SymbolId c = t.intern("conv");
    MLK_CHECK_EQ(a, b);
    MLK_CHECK(a != c);
    MLK_CHECK_EQ(t.text(a), "matmul");
}

MLK_TEST(core, json_roundtrip) {
    mlk::json::Value doc = mlk::json::Object{};
    doc.set("name", mlk::json::Value{"graph"});
    doc.set("n", mlk::json::Value{static_cast<int64_t>(42)});
    doc.set("x", mlk::json::Value{2.5});
    doc.set("ok", mlk::json::Value{true});
    mlk::json::Value arr = mlk::json::Array{};
    arr.push(mlk::json::Value{static_cast<int64_t>(1)});
    arr.push(mlk::json::Value{static_cast<int64_t>(2)});
    doc.set("list", std::move(arr));
    const std::string text = mlk::json::serialize(doc);
    auto parsed = mlk::json::parse(text);
    MLK_CHECK(parsed.has_value());
    MLK_CHECK_EQ(parsed->find("name")->asString(), "graph");
    MLK_CHECK_EQ(parsed->find("n")->asInt(), 42);
    MLK_CHECK_NEAR(parsed->find("x")->asDouble(), 2.5, 1e-12);
    MLK_CHECK(parsed->find("list")->asArray().size() == 2);
}

MLK_TEST(core, json_rejects_malformed) {
    // Rule 124: malformed input rejected safely.
    MLK_CHECK(!mlk::json::parse("{\"a\":}").has_value());
    MLK_CHECK(!mlk::json::parse("[1,2").has_value());
    MLK_CHECK(!mlk::json::parse("{\"a\":1}trailing").has_value());
    std::string deep;
    for (int i = 0; i < 300; ++i) deep += "[";
    deep += "1";
    for (int i = 0; i < 300; ++i) deep += "]";
    MLK_CHECK(!mlk::json::parse(deep).has_value());  // depth bound
}

MLK_TEST_MAIN("core")
