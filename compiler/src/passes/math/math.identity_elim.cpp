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
                // INT path: integer arithmetic is exact; x + 0 -> x for any
                // int constant payload. The previous code probed int
                // constants through the FP slot of isConst(), which leaves
                // the caller's double untouched — so add(x, int 5) read as
                // "constant 0.0" and folded to x (certified miscompile).
                int64_t ci = 0;
                double c = 0.0;
                if (isIntConst(graph, n.inputs[0], &ci) && ci == 0) {
                    replacement = n.inputs[1];
                } else if (isIntConst(graph, n.inputs[1], &ci) && ci == 0) {
                    replacement = n.inputs[0];
                }
                // FP path. The sign that matters is the OPERAND's, not the
                // constant's: x + (-0.0) == x for EVERY x (both zeros:
                // (+0)+(-0)=+0, (-0)+(-0)=-0), so this folds under any
                // profile. x + (+0.0) == x fails for x = -0.0 (IEEE gives
                // +0), so it folds only when the domain drops -0
                // (preserveNegativeZero=false). The previous gate tested
                // signbit(constant) — exactly inverted: it blocked the
                // sound x+(-0) form and fired the unsound x+(+0) form
                // (certified: execution at x=-0 flipped +0 -> -0).
                else if (isFpConst(graph, n.inputs[0], &c) && c == 0.0) {
                    if (std::signbit(c) || !preserveNegZero) {
                        replacement = n.inputs[1];
                    }
                } else if (isFpConst(graph, n.inputs[1], &c) && c == 0.0) {
                    if (std::signbit(c) || !preserveNegZero) {
                        replacement = n.inputs[0];
                    }
                }
            } else if (n.op == MathOp::Mul && n.numInputs() == 2) {
                double c = 0.0;
                int64_t ci = 0;
                // x * 1 -> x: exact for FP (any x) and for INT (exact).
                // Int constants are probed through the typed helper (see
                // the Add arm note; mul(x, int 1) never fired before).
                if (isFpConst(graph, n.inputs[0], &c) && c == 1.0) {
                    replacement = n.inputs[1];
                } else if (isFpConst(graph, n.inputs[1], &c) && c == 1.0) {
                    replacement = n.inputs[0];
                } else if (isIntConst(graph, n.inputs[0], &ci) && ci == 1) {
                    replacement = n.inputs[1];
                } else if (isIntConst(graph, n.inputs[1], &ci) && ci == 1) {
                    replacement = n.inputs[0];
                }
                // x * 0 -> 0 is FORBIDDEN for FP unless x is provably
                // non-NaN (NaN*0 = NaN) — Rule 87/90. Integers fold.
                if (replacement == kInvalidValueId &&
                    isIntConst(graph, n.inputs[0], &ci) && ci == 0 &&
                    isIntDtype(graph, n.inputs[1])) {
                    replacement = n.inputs[0];
                } else if (replacement == kInvalidValueId &&
                           isIntConst(graph, n.inputs[1], &ci) && ci == 0 &&
                           isIntDtype(graph, n.inputs[0])) {
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
