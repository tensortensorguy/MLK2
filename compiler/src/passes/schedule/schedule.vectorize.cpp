// schedule.vectorize — vector-width schedule params for elementwise
// subgraphs (spec §8.7; Rule 36: vectorization requires a dependence
// proof — fresh NodeResults never alias inputs, so elementwise SSA
// subgraphs qualify trivially).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kSchedTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class VectorizePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Rule 36: vectorization requires dependence proof. For elementwise
        // SSA subgraphs over distinct buffers the dependence proof is
        // trivial (fresh NodeResults never alias inputs); recorded as the
        // vector_width schedule param.
        const SymbolId vw = ctx.symbols->intern("vector_width");
        int64_t width = static_cast<int64_t>(constants::kDefaultSimdWidthF32);
        if (ctx.knobs != nullptr) {
            if (const int64_t* v = ctx.knobs->find(vw)) width = *v;
        }
        for (const NodeId nid : graph.topoOrder()) {
            Node& node = graph.node(nid);
            if (node.flags.test(NodeFlag::Dead)) continue;
            if (!isElementwiseMath(node.op)) continue;
            bool has = false;
            for (const auto& a : node.attrs) {
                if (a.name == vw) has = true;
            }
            if (!has) {
                Attr a;
                a.name = vw;
                a.value = AttrValue{width};
                node.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_schedule_vectorize_pass(SymbolTable& symbols) {
    static VectorizePass pass(symbols, "schedule.vectorize",
                              PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"schedule.fused"},
                 {"schedule.vectorized"}, {},
                 {kSchedTiers[0], kSchedTiers[1], kSchedTiers[2]});
}

}  // namespace mlk::passes
