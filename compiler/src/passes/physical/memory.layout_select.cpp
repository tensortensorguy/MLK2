// memory.layout_select — concrete physical layout (spec §8.8; Rule 27/31:
// constants from mlk::constants, target params from the profile).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kPhysTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class LayoutSelectPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Concrete layout: contiguous row-major + 64B alignment (constants;
        // Rule 27/31).
        const SymbolId layoutAttr = ctx.symbols->intern("concrete_layout");
        const SymbolId layout = ctx.symbols->intern("contiguous_aligned");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const Value& res = graph.value(n.results[0]);
            if (!res.type.tensor) continue;
            bool has = false;
            for (const auto& a : n.attrs) {
                if (a.name == layoutAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = layoutAttr;
                a.value = AttrValue{layout};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_memory_layout_select_pass(SymbolTable& symbols) {
    static LayoutSelectPass pass(symbols, "memory.layout_select",
                                 PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"memory.placed"},
                 {"memory.layout"}, {},
                 {kPhysTiers[0], kPhysTiers[1], kPhysTiers[2]});
}

}  // namespace mlk::passes
