// schedule.parallelize — declarative parallel mapping (spec §8.7; the
// executor decides thread counts: Rule 12 thread-local allocation, Rule 137
// no global locks on hot paths).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kSchedTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class ParallelizePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Parallel mapping is recorded declaratively; the executor decides
        // thread counts (Rule 12: thread-local allocation; Rule 137: no
        // global locks on hot paths).
        const SymbolId threadsAttr = ctx.symbols->intern("parallel");
        const int64_t defaultParallel = 1;
        for (const NodeId nid : graph.topoOrder()) {
            Node& node = graph.node(nid);
            if (node.flags.test(NodeFlag::Dead)) continue;
            bool has = false;
            for (const auto& a : node.attrs) {
                if (a.name == threadsAttr) has = true;
            }
            if (!has) {
                Attr a;
                a.name = threadsAttr;
                a.value = AttrValue{defaultParallel};
                node.attrs.push_back(a);
                r.changed = true;
            }
        }
        return r;
    }
};

void register_schedule_parallelize_pass(SymbolTable& symbols) {
    static ParallelizePass pass(symbols, "schedule.parallelize",
                                PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform,
                 {"schedule.vectorized"}, {"schedule.parallel"}, {},
                 {kSchedTiers[0], kSchedTiers[1], kSchedTiers[2]});
}

}  // namespace mlk::passes
