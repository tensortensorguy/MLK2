// math.constant_fold — defined-only, pure-only constant folding (spec §8.2;
// Rule 78: fold only defined, domain-safe ops; Rule 87/90: never create
// NaN/Inf, preserve input NaNs; integer arithmetic folds exactly).
#include "../passes_common.h"
#include "math_rewrite_utils.h"

#include <cmath>

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class ConstantFoldPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        r.nodesBefore = graph.liveNodeCount();
        uint32_t edits = 0;
        const NumericSemantics& num = ctx.domainProfile->numeric;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || !isPureNode(graph, nid)) {
                continue;
            }
            if (n.op == MathOp::Lambda || n.op == MathOp::Apply) {
                continue;  // symbolic: never folded (Rule 87)
            }
            bool allConst = n.numInputs() > 0;
            for (const ValueId in : n.inputs) {
                if (!isConst(graph, in, nullptr, nullptr)) {
                    allConst = false;
                    break;
                }
            }
            if (!allConst) continue;
            double a = 0.0;
            double b2 = 0.0;
            int64_t ai = 0, bi = 0;
            (void)isConst(graph, n.inputs[0], &a, &ai);
            if (n.numInputs() == 2) {
                (void)isConst(graph, n.inputs[1], &b2, &bi);
            }
            bool allIntConst = true;
            for (const ValueId in : n.inputs) {
                if (!graph.value(in).constant.isInt) {
                    allIntConst = false;
                    break;
                }
            }
            if (allIntConst && n.numInputs() == 2 &&
                (n.op == MathOp::Add || n.op == MathOp::Sub ||
                 n.op == MathOp::Mul)) {
                // Integer arithmetic folds exactly.
                int64_t out = 0;
                if (n.op == MathOp::Add) out = ai + bi;
                if (n.op == MathOp::Sub) out = ai - bi;
                if (n.op == MathOp::Mul) out = ai * bi;
                MathType t = graph.value(n.results[0]).type;
                // addIntConstant is infallible: returns a plain ValueId.
                const ValueId newV = graph.addIntConstant(out, t);
                MLK_TRYV(replaceResult(graph, nid, newV));
                ++edits;
                continue;
            }
            double out = 0.0;
            bool defined = true;
            switch (n.op) {  // Rule 78: fold only defined, pure, domain-safe
                case MathOp::Add: out = a + b2; break;
                case MathOp::Sub: out = a - b2; break;
                case MathOp::Mul: out = a * b2; break;
                case MathOp::Div:
                    if (b2 == 0.0) defined = false;  // singularity: no fold
                    else out = a / b2;
                    break;
                case MathOp::Neg: out = -a; break;
                case MathOp::Exp: out = std::exp(a); break;
                case MathOp::Log:
                    if (a <= 0.0) defined = false;  // domain violation
                    else out = std::log(a);
                    break;
                case MathOp::Sin: out = std::sin(a); break;
                case MathOp::Cos: out = std::cos(a); break;
                case MathOp::Tan: out = std::tan(a); break;
                case MathOp::Tanh: out = std::tanh(a); break;
                case MathOp::Sqrt:
                    if (a < 0.0) defined = false;
                    else out = std::sqrt(a);
                    break;
                case MathOp::Rsqrt:
                    if (a <= 0.0) defined = false;
                    else out = 1.0 / std::sqrt(a);
                    break;
                case MathOp::Erf: out = std::erf(a); break;
                default:
                    defined = false;
                    break;
            }
            if (!defined) continue;
            // Rule 87/90: never CREATE new NaN/Inf; preserve input NaNs.
            if (a != a || (n.numInputs() == 2 && b2 != b2)) continue;
            if (out != out || std::isinf(out)) continue;
            if (num.preserveInf && std::isinf(a)) continue;
            MathType t = graph.value(n.results[0]).type;            MLK_TRY_VAR(newV, makeConstant(graph, out, t));
            MLK_TRYV(replaceResult(graph, nid, newV));
            ++edits;
            ctx.budget.consume();
            if (ctx.budget.exhausted()) {
                return err(ErrorCode::BudgetExceeded,
                           "constant_fold budget exhausted", 131);
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_constant_fold_pass(SymbolTable& symbols) {
    static ConstantFoldPass pass(symbols, "math.constant_fold",
                                 PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"effect.inferred"},
                 {"math.canonical"}, {"analysis.cse"},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
