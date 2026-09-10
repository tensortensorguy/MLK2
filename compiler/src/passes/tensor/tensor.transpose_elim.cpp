// tensor.transpose_elim — t(t(x)) -> x (spec §8.5; Rule 21: equivalence
// recorded, original recoverable).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class TransposeElimPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::Transpose) {
                continue;
            }
            const Value& in = graph.value(n.inputs[0]);
            if (in.kind == ValueKind::NodeResult &&
                graph.node(in.producer).op == MathOp::Transpose &&
                !graph.isNodeDead(in.producer)) {
                const ValueId inner = graph.node(in.producer).inputs[0];
                graph.replaceOperandUses(n.results[0], inner);
                graph.recordEquivalent(n.results[0], inner);
                (void)graph.killNode(nid);
                ++edits;
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_tensor_transpose_elim_pass(SymbolTable& symbols) {
    static TransposeElimPass pass(symbols, "tensor.transpose_elim",
                                  PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"type.inferred"},
                 {"tensor.layout"}, {"analysis.cse"},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
