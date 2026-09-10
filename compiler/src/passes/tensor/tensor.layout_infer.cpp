// tensor.layout_infer — abstract layout policy (spec §8.5; Rule 23/40:
// layout is a physical concern, decided here and accounted by the cost
// model; domain knowledge enters via the profile, never if(domain==Tensor)
// in generic passes, Rule 28).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class LayoutInferPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // MVP layout policy: contiguous row-major everywhere unless a
        // profile/layout constraint says otherwise; transform costs are
        // accounted by the cost model (Rule 40: physicalization preserves
        // meaning; layout is a physical concern — Rule 23).
        const SymbolId layoutName = ctx.symbols->intern("row_major");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            if (n.op == MathOp::LayoutTransform) continue;
            Attr a;
            a.name = ctx.symbols->intern("layout");
            a.value = AttrValue{layoutName};
            bool hasLayout = false;
            for (const auto& existing : n.attrs) {
                if (existing.name == a.name) hasLayout = true;
            }
            if (!hasLayout) {
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_tensor_layout_infer_pass(SymbolTable& symbols) {
    static LayoutInferPass pass(symbols, "tensor.layout_infer",
                                PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"type.inferred"},
                 {"tensor.layout"}, {},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
