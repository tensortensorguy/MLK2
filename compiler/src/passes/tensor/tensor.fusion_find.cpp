// tensor.fusion_find — marks maximal elementwise chains as fusion groups
// (spec §8.5/§12; profile gate: HasKernelFusion, Part 0).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kTensorTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class FusionFindPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        if (!ctx.domainProfile->has(Capability::HasKernelFusion)) {
            return r;  // profile gate (Part 0)
        }
        // Marks maximal elementwise chains: producer-consumer edges between
        // elementwise ops with single users fuse into one kernel region
        // (spec §12: fusion is graph-level).
        const SymbolId groupAttr = ctx.symbols->intern("fusion_group");
        int64_t groupId = 0;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            if (!isElementwiseMath(n.op)) continue;
            if (findAttr(n.attrs, groupAttr) != nullptr) continue;
            const Value& result = graph.value(n.results[0]);
            // Only fuse chains whose result has exactly one elementwise user.
            bool fuseWithUser = false;
            if (graph.users(n.results[0]).size() == 1) {
                const NodeId user = graph.users(n.results[0])[0];
                if (isElementwiseMath(graph.node(user).op) &&
                    !graph.node(user).flags.test(NodeFlag::Dead)) {
                    fuseWithUser = true;
                }
            }
            if (fuseWithUser || result.facts.isTrue(PropertyId::Pure)) {
                Attr a;
                a.name = groupAttr;
                a.value = AttrValue{groupId};
                n.attrs.push_back(a);
                r.changed = true;
            }
            if (graph.users(n.results[0]).size() != 1) ++groupId;
        }
        return r;
    }
};

void register_tensor_fusion_find_pass(SymbolTable& symbols) {
    static FusionFindPass pass(symbols, "tensor.fusion_find",
                               PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"effect.inferred"},
                 {"schedule.fusable"}, {},
                 {kTensorTiers[0], kTensorTiers[1], kTensorTiers[2]});
}

}  // namespace mlk::passes
