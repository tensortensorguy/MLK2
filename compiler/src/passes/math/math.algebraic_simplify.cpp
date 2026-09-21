// math.algebraic_simplify — bit-exact FP sign/division identities
// (spec §8.2; Rule 33: every rewrite carries its exactness argument).
//
// Scope is deliberately NARROW: only rewrites whose result is bit-identical
// to the original for EVERY input under IEEE-754 round-to-nearest. The
// reassociation family is NOT here (Rule 33 forbids it without an accuracy
// contract; no shipped profile opens that gate — recorded in ADR-0009).
// Identity elimination (x+0, x*1) lives in math.identity_elim; this pass
// owns the complement set that identity elimination does not cover.
#include "../passes_common.h"
#include "math_rewrite_utils.h"

#include <cmath>

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class AlgebraicSimplifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            ValueId replacement = kInvalidValueId;
            // Sub(x, +0.0) -> x. Exact for every x: subtracting a positive
            // zero only changes the exponent when x is zero, and there
            // (+0)-(+0)=+0=x and (-0)-(+0)=-0=x (IEEE round-to-nearest).
            if (n.op == MathOp::Sub && n.numInputs() == 2 &&
                !isIntDtype(graph, n.results[0])) {
                double c = 0.0;
                if (isFpConst(graph, n.inputs[1], &c) && c == 0.0 &&
                    !std::signbit(c)) {
                    replacement = n.inputs[0];
                }
            }
            // Div(x, 1.0) -> x and Div(x, -1.0) -> Neg(x). Division by ±1
            // is exact (the magnitude is unchanged; the sign flip of the
            // -1 form is the exact FP negate). Replacing a division with a
            // negate is the classic strength win; both are bit-identical.
            else if (n.op == MathOp::Div && n.numInputs() == 2 &&
                     !isIntDtype(graph, n.results[0])) {
                double c = 0.0;
                if (isFpConst(graph, n.inputs[1], &c) && c == 1.0) {
                    replacement = n.inputs[0];
                } else if (isFpConst(graph, n.inputs[1], &c) && c == -1.0 &&
                           !isIntDtype(graph, n.inputs[0])) {
                    // Appended nodes are renumbered topologically by the
                    // pipeline runner (Rule 77: one renumber site) and the
                    // replacement's type transfers in replaceResult.
                    MLK_TRY_VAR(negV, graph.addNode(MathOp::Neg,
                                                    {n.inputs[0]}));
                    replacement = negV;
                }
            }
            // Neg(Neg(x)) -> x. FP negate is an exact sign flip; two flips
            // restore every bit, including zeros, infinities, and NaN
            // payloads.
            else if (n.op == MathOp::Neg && n.numInputs() == 1) {
                const Value& in = graph.value(n.inputs[0]);
                if (in.kind == ValueKind::NodeResult &&
                    graph.node(in.producer).op == MathOp::Neg &&
                    !graph.isNodeDead(in.producer)) {
                    replacement = graph.node(in.producer).inputs[0];
                }
            }
            if (replacement != kInvalidValueId) {
                MLK_TRYV(replaceResult(graph, nid, replacement));
                ++edits;
                ctx.budget.consume();
                if (ctx.budget.exhausted()) {
                    return err(ErrorCode::BudgetExceeded,
                               "algebraic_simplify budget exhausted", 131);
                }
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_algebraic_simplify_pass(SymbolTable& symbols) {
    static AlgebraicSimplifyPass pass(symbols, "math.algebraic_simplify",
                                      PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.algebraic"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
