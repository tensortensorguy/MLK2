// memory.place — memory-space assignment (spec §8.8; Rule 31: hardware
// params come from Target/HardwareInfo — here host DRAM for the CPU
// backend; Scalar->Register / Tensor HBM->SRAM moves are the same
// mechanism specialized per profile).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kPhysTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class PlacePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Memory space assignment: host DRAM for the CPU backend (Rule 31:
        // hardware params from Target/HardwareInfo, not hardcoded GPU
        // assumptions).
        const SymbolId spaceAttr = ctx.symbols->intern("memory_space");
        const SymbolId dram = ctx.symbols->intern("dram");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const Value& res = graph.value(n.results[0]);
            if (!res.type.tensor) continue;
            bool has = false;
            for (const auto& a : n.attrs) {
                if (a.name == spaceAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = spaceAttr;
                a.value = AttrValue{dram};
                n.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_memory_place_pass(SymbolTable& symbols) {
    static PlacePass pass(symbols, "memory.place", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"memory.planned"},
                 {"memory.placed"}, {},
                 {kPhysTiers[0], kPhysTiers[1], kPhysTiers[2]});
}

}  // namespace mlk::passes
