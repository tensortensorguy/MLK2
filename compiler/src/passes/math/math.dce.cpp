// math.dce — mark-and-sweep dead code elimination (spec §8.2; Rule 92:
// observable values — outputs, effecting nodes, solver state — survive).
#include "../passes_common.h"
#include "math_rewrite_utils.h"
#include "mlk/core/hash_map.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class DcePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        // Mark-reach from outputs through operands. Never remove effecting
        // nodes, tracing hooks, or solver state (Rule 92: observable values
        // must be materialized).
        OpenHashSet<ValueId> live;
        SmallVector<ValueId, 16> worklist;
        for (const ValueId out : graph.outputs()) {
            (void)live.insert(out);
            worklist.push_back(out);
        }
        for (const NodeId nid : graph.effectChain().order()) {
            const Node& n = graph.node(nid);
            for (const ValueId out : n.results) {
                if (!live.contains(out)) {
                    (void)live.insert(out);
                    worklist.push_back(out);
                }
            }
        }
        while (!worklist.empty()) {
            const ValueId v = worklist.back();
            worklist.pop_back();
            const Value& val = graph.value(v);
            if (val.kind != ValueKind::NodeResult) continue;
            const Node& producer = graph.node(val.producer);
            for (const ValueId in : producer.inputs) {
                if (!live.contains(in)) {
                    (void)live.insert(in);
                    worklist.push_back(in);
                }
            }
        }
        uint32_t killed = 0;
        for (const NodeId nid : graph.topoOrder()) {
            const Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            if (!live.contains(n.results[0])) {
                (void)graph.killNode(nid);
                ++killed;
                ctx.budget.consume();
                if (ctx.budget.exhausted()) {
                    return err(ErrorCode::BudgetExceeded, "dce budget exhausted",
                               131);
                }
            }
        }
        r.changed = killed != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_dce_pass(SymbolTable& symbols) {
    static DcePass pass(symbols, "math.dce", PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"effect.inferred"},
                 {"math.dce"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
