// memory.buffer_plan — buffer decisions for output tensors (spec §8.8;
// intermediates inside fused regions stay register-resident: fusion is
// traffic savings, spec §12).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kPhysTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class BufferPlanPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Every output-tensor value gets a buffer decision; intermediate
        // values inside fused regions stay register-resident (fusion =
        // traffic savings, spec §12).
        const SymbolId bufferAttr = ctx.symbols->intern("buffer");
        const SymbolId alloc = ctx.symbols->intern("allocate");
        for (const ValueId vid : graph.outputs()) {
            const Value& v = graph.value(vid);
            if (!v.type.tensor) continue;
            // buffer decision rides on the node producing it
            if (v.kind != ValueKind::NodeResult) continue;
            Node& n = graph.node(v.producer);
            bool has = false;
            for (const auto& a : n.attrs) {
                if (a.name == bufferAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = bufferAttr;
                a.value = AttrValue{alloc};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_memory_buffer_plan_pass(SymbolTable& symbols) {
    static BufferPlanPass pass(symbols, "memory.buffer_plan",
                               PassKind::Transform);
    // Contract prerequisite fixed (ADR-0009): the plan's true inputs are
    // the shape/effect facts and output set — the former "memory.liveness"
    // prerequisite named a pass that the pipelines never ran and that no
    // consumer ever read (a phantom dependency, Rule 142 honesty).
    registerPass(symbols, pass, PassKind::Transform, {"effect.inferred"},
                 {"memory.planned"}, {},
                 {kPhysTiers[0], kPhysTiers[1], kPhysTiers[2]});
}

}  // namespace mlk::passes
