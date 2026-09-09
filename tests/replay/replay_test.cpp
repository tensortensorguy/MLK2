// Rule 45/158: replay artifacts are sufficient for debugging from replay.
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/ir/graph_json.h"
#include "mlk/runtime/graph_state.h"
#include "mlk/support/json.h"

#include "mlk_test.h"

MLK_TEST(replay, artifact_carries_graph_and_state) {
    mlk::SymbolTable symbols;
    mlk::GraphBuilder b(symbols);
    const mlk::MathType f64 =
        mlk::MathType::scalar(mlk::Domain::Float, mlk::Dtype::F64);
    const mlk::ValueId x = b.placeholder("x", f64);
    auto s = b.op(mlk::MathOp::Sin, {x});
    MLK_CHECK(s.has_value());
    b.output(*s);
    mlk::GraphState state;
    state.resumeNode = 0;
    state.graphVersion = b.graph().version();

    mlk::json::Value artifact = mlk::json::Object{};
    artifact.set("graph", mlk::graphToJson(b.graph(), symbols));
    artifact.set("graph_state", state.toJson());
    const std::string text = mlk::json::serializePretty(artifact);

    // Replay: parse artifact, restore graph, restore state (Rule 158).
    auto parsed = mlk::json::parse(text);
    MLK_CHECK(parsed.has_value());
    auto graph = mlk::graphFromJson(*parsed->find("graph"), symbols);
    MLK_CHECK(graph.has_value());
    auto restoredState = mlk::GraphState::fromJson(*parsed->find("graph_state"));
    MLK_CHECK(restoredState.has_value());
    MLK_CHECK_EQ(restoredState->graphVersion, b.graph().version());
}

MLK_TEST_MAIN("replay")
