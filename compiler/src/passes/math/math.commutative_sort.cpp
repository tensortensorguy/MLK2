// math.commutative_sort — canonical operand order for commutative ops by
// subtree hash (spec §8.2; legality from property facts, Rule 33).
#include "../passes_common.h"
#include "math_rewrite_utils.h"
#include "mlk/ir/graph_hash.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class CommutativeSortPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            const Value& result = graph.value(n.results[0]);
            if (!result.facts.isTrue(PropertyId::Commutative)) continue;
            // Sort operands by canonical subtree hash (deterministic;
            // helps CSE/GVN/pattern matching per spec §8.2).
            bool sorted = false;
            for (std::size_t i = 1; i < n.inputs.size() && !sorted; ++i) {
                if (valueHash(graph, n.inputs[i - 1]) >
                    valueHash(graph, n.inputs[i])) {
                    SmallVector<ValueId, 4> ins;
                    for (const ValueId in : n.inputs) ins.push_back(in);
                    // insertion-sort by hash
                    for (std::size_t a = 1; a < ins.size(); ++a) {
                        std::size_t bIdx = a;
                        while (bIdx > 0 &&
                               valueHash(graph, ins[bIdx - 1]) >
                                   valueHash(graph, ins[bIdx])) {
                            ValueId tmp = ins[bIdx - 1];
                            ins[bIdx - 1] = ins[bIdx];
                            ins[bIdx] = tmp;
                            --bIdx;
                        }
                    }
                    // Rebuild as new node (Rule 21: no destructive rewrite).
                    MLK_TRY_VAR(newV, graph.addNode(n.op, ins, n.attrs));
                    MLK_TRYV(replaceResult(graph, nid, newV));
                    ++edits;
                    sorted = true;
                    ctx.budget.consume();
                    if (ctx.budget.exhausted()) {
                        return err(ErrorCode::BudgetExceeded,
                                   "commutative_sort budget exhausted", 131);
                    }
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_commutative_sort_pass(SymbolTable& symbols) {
    static CommutativeSortPass pass(symbols, "math.commutative_sort",
                                    PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.canonical"}, {"analysis.cse"},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
