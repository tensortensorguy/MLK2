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
    registerPass(symbols, pass, PassKind::Transform, {"memory.liveness"},
                 {"memory.planned"}, {},
                 {kPhysTiers[0], kPhysTiers[1], kPhysTiers[2]});
}

}  // namespace mlk::passes
