// math.identity_elim — algebraic identity elimination (spec §8.2):
// x+0, x*1, int x*0, t(t(x)). FP-semantics gated (Rule 87/90).
#include "../passes_common.h"
#include "math_rewrite_utils.h"

#include <cmath>

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class IdentityElimPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        const bool preserveNegZero =
            ctx.domainProfile->numeric.preserveNegativeZero;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            ValueId replacement = kInvalidValueId;
            if (n.op == MathOp::Add && n.numInputs() == 2) {
                double c = 0.0;
                // x + 0 == x except (-0.0) + (+0.0) == +0.0: gated.
                if (isConst(graph, n.inputs[0], &c) && c == 0.0 &&
                    (!preserveNegZero || !std::signbit(c))) {
                    replacement = n.inputs[1];
                } else if (isConst(graph, n.inputs[1], &c) && c == 0.0 &&
                           (!preserveNegZero || !std::signbit(c))) {
                    replacement = n.inputs[0];
                }
            } else if (n.op == MathOp::Mul && n.numInputs() == 2) {
                double c = 0.0;
                if (isConst(graph, n.inputs[0], &c) && c == 1.0) {
                    replacement = n.inputs[1];
                } else if (isConst(graph, n.inputs[1], &c) && c == 1.0) {
                    replacement = n.inputs[0];
                }
                // x * 0 -> 0 is FORBIDDEN for FP unless x is provably
                // non-NaN (NaN*0 = NaN) — Rule 87/90. Integers fold.
                int64_t ci = 0;
                if (replacement == kInvalidValueId &&
                    isConst(graph, n.inputs[0], nullptr, &ci) && ci == 0 &&
                    isIntDtype(graph, n.inputs[1])) {
                    replacement = n.inputs[0];
                } else if (replacement == kInvalidValueId &&
                           isConst(graph, n.inputs[1], nullptr, &ci) &&
                           ci == 0 && isIntDtype(graph, n.inputs[0])) {
                    replacement = n.inputs[1];
                }
            } else if (n.op == MathOp::Transpose) {
                const Value& in = graph.value(n.inputs[0]);
                if (in.kind == ValueKind::NodeResult &&
                    graph.node(in.producer).op == MathOp::Transpose &&
                    !graph.isNodeDead(in.producer)) {
                    // t(t(x)) -> x
                    replacement = graph.node(in.producer).inputs[0];
                }
            }
            if (replacement != kInvalidValueId) {
                MLK_TRYV(replaceResult(graph, nid, replacement));
                ++edits;
                ctx.budget.consume();
                if (ctx.budget.exhausted()) {
                    return err(ErrorCode::BudgetExceeded,
                               "identity_elim budget exhausted", 131);
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_identity_elim_pass(SymbolTable& symbols) {
    static IdentityElimPass pass(symbols, "math.identity_elim",
                                 PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.canonical"}, {"analysis.cse"},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
