// math.normalize_ops — pow(x,2) -> mul(x,x) canonical square form (spec §8.2).
#include "../passes_common.h"
#include "math_rewrite_utils.h"
#include "mlk/core/cancellation.h"

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class NormalizeOpsPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        (void)ctx;
        for (const NodeId nid : graph.topoOrder()) {
            if (ctx.cancel != nullptr && ctx.cancel->cancelled()) {
                return err(ErrorCode::Cancelled, "compilation cancelled");
            }
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            // pow(x, 2) -> mul(x, x): canonical square form (spec §8.2).
            if (n.op == MathOp::Pow && n.numInputs() == 2) {
                double e = 0.0;
                if (isConst(graph, n.inputs[1], &e) && e == 2.0) {
                    SmallVector<ValueId, 4> ins;
                    ins.push_back(n.inputs[0]);
                    ins.push_back(n.inputs[0]);                    MLK_TRY_VAR(newV, graph.addNode(MathOp::Mul, ins));
                    MLK_TRYV(replaceResult(graph, nid, newV));
                    ++edits;
                    ctx.budget.consume();
                    if (ctx.budget.exhausted()) {
                        return err(ErrorCode::BudgetExceeded,
                                   "normalize_ops edit budget exhausted",
                                   131);
                    }
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_normalize_ops_pass(SymbolTable& symbols) {
    static NormalizeOpsPass pass(symbols, "math.normalize_ops",
                                 PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.normalized"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
