// memory.liveness — reverse-order liveness over node results (spec §8.8;
// Rule 18: SparseSet for dataflow sets; Rule 23: physical state references
// the math graph, never the reverse).
#include "../passes_common.h"
#include "mlk/core/sparse_set.h"

namespace mlk::passes {

namespace {
constexpr Tier kPhysTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class LivenessPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)ctx;
        // Reverse-order mark liveness over node results (Rule 18: SparseSet
        // for dataflow sets).
        SparseSet live(graph.numValues());
        for (const ValueId out : graph.outputs()) live.insert(out);
        const auto order = graph.topoOrder();
        for (std::size_t i = order.size(); i-- > 0;) {
            const Node& n = graph.node(order[i]);
            bool used = false;
            for (const ValueId res : n.results) used = used || live.contains(res);
            if (used) {
                for (const ValueId in : n.inputs) live.insert(in);
            }
        }
        r.changed = false;
        return r;
    }
};

void register_memory_liveness_pass(SymbolTable& symbols) {
    static LivenessPass pass(symbols, "memory.liveness",
                             PassKind::Analysis);
    registerPass(symbols, pass, PassKind::Analysis, {"effect.inferred"},
                 {"memory.liveness"}, {},
                 {kPhysTiers[0], kPhysTiers[1], kPhysTiers[2]});
}

}  // namespace mlk::passes
