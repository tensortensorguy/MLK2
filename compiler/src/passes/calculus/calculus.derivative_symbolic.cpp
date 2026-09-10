// calculus.derivative_symbolic — symbolic differentiation lowering
// (spec §8.4, spec example: d/dx x² sin(x) = 2x sin(x) + x² cos(x)).
//
// Calculus is first-class: Derivative nodes remain symbolic until this
// strategy-lowering pass fires, and the original expression is never
// destroyed (Rule 21 — recordEquivalent keeps it recoverable).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kCalcTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};

/// Symbolic differentiation rules. Returns kInvalidValueId when no rule
/// applies (reported as Unimplemented — never guessed, Rule 28).
Result<ValueId> diffValue(MathGraph& g, ValueId v, ValueId x,
                          uint32_t depth) {
    if (depth > constants::kMaxEquivalenceDepth) {
        return err(ErrorCode::BudgetExceeded,
                   "symbolic differentiation depth exceeded", 10);
    }
    const Value& val = g.value(v);
    // d/dx c = 0 ; d/dx x = 1
    if (val.kind == ValueKind::Constant) {
        MathType t = val.type;
        return g.addConstant(0.0, t);
    }
    if (val.kind == ValueKind::Placeholder || val.kind == ValueKind::Variable ||
        val.kind == ValueKind::Symbol) {
        MathType t = val.type;
        return g.addConstant(v == x ? 1.0 : 0.0, t);
    }
    if (val.kind != ValueKind::NodeResult) {
        return err(ErrorCode::InvalidGraph, "differentiating unknown value");
    }
    const Node& n = g.node(val.producer);
    MathType t = g.value(n.results[0]).type;

    auto c = [&](double d) -> Result<ValueId> {
        return g.addConstant(d, t);
    };
    auto bin = [&](MathOp op, ValueId a, ValueId b) {
        SmallVector<ValueId, 4> ins;
        ins.push_back(a);
        ins.push_back(b);
        return g.addNode(op, ins);
    };
    auto un = [&](MathOp op, ValueId a) {
        SmallVector<ValueId, 4> ins;
        ins.push_back(a);
        return g.addNode(op, ins);
    };

    switch (n.op) {  // Rule 78: exhaustive (rules cover the MVP op set)
        case MathOp::Add: {
            MLK_TRY_VAR(da, diffValue(g, n.inputs[0], x, depth + 1));
            MLK_TRY_VAR(db, diffValue(g, n.inputs[1], x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(da);
            ins.push_back(db);
            return g.addNode(MathOp::Add, ins);
        }
        case MathOp::Sub: {
            MLK_TRY_VAR(da, diffValue(g, n.inputs[0], x, depth + 1));
            MLK_TRY_VAR(db, diffValue(g, n.inputs[1], x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(da);
            ins.push_back(db);
            return g.addNode(MathOp::Sub, ins);
        }
        case MathOp::Mul: {
            // (uv)' = u'v + uv'
            const ValueId u = n.inputs[0];
            const ValueId w = n.inputs[1];
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            MLK_TRY_VAR(dw, diffValue(g, w, x, depth + 1));
            MLK_TRY_VAR(duw, bin(MathOp::Mul, du, w));
            MLK_TRY_VAR(udw, bin(MathOp::Mul, u, dw));
            SmallVector<ValueId, 4> ins;
            ins.push_back(duw);
            ins.push_back(udw);
            return g.addNode(MathOp::Add, ins);
        }
        case MathOp::Div: {
            // (u/w)' = (u'w - u w') / w^2
            const ValueId u = n.inputs[0];
            const ValueId w = n.inputs[1];
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            MLK_TRY_VAR(dw, diffValue(g, w, x, depth + 1));
            MLK_TRY_VAR(duw, bin(MathOp::Mul, du, w));
            MLK_TRY_VAR(udw, bin(MathOp::Mul, u, dw));
            SmallVector<ValueId, 4> ns;
            ns.push_back(duw);
            ns.push_back(udw);
            MLK_TRY_VAR(num, g.addNode(MathOp::Sub, ns));
            MLK_TRY_VAR(w2, bin(MathOp::Mul, w, w));
            SmallVector<ValueId, 4> ds;
            ds.push_back(num);
            ds.push_back(w2);
            return g.addNode(MathOp::Div, ds);
        }
        case MathOp::Neg: {
            MLK_TRY_VAR(du, diffValue(g, n.inputs[0], x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(du);
            return g.addNode(MathOp::Neg, ins);
        }
        case MathOp::Sin: {
            // (sin u)' = cos(u) * u'
            const ValueId u = n.inputs[0];
            MLK_TRY_VAR(cu, un(MathOp::Cos, u));
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(cu);
            ins.push_back(du);
            return g.addNode(MathOp::Mul, ins);
        }
        case MathOp::Cos: {
            // (cos u)' = -sin(u) * u'
            const ValueId u = n.inputs[0];
            MLK_TRY_VAR(su, un(MathOp::Sin, u));
            MLK_TRY_VAR(nsu, un(MathOp::Neg, su));
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(nsu);
            ins.push_back(du);
            return g.addNode(MathOp::Mul, ins);
        }
        case MathOp::Exp: {
            // (exp u)' = exp(u) * u'
            const ValueId u = n.inputs[0];
            MLK_TRY_VAR(eu, un(MathOp::Exp, u));
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(eu);
            ins.push_back(du);
            return g.addNode(MathOp::Mul, ins);
        }
        case MathOp::Log: {
            // (log u)' = u' / u
            const ValueId u = n.inputs[0];
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(du);
            ins.push_back(u);
            return g.addNode(MathOp::Div, ins);
        }
        case MathOp::Pow: {
            // (u^c)' = c * u^(c-1) * u'   (constant-exponent rule; the
            // general u^v rule requires log-derivation and is Tier-3 work).
            const ValueId u = n.inputs[0];
            const ValueId e = n.inputs[1];
            const Value& ev = g.value(e);
            if (ev.kind != ValueKind::Constant || ev.constant.isInt) {
                return err(ErrorCode::Unimplemented,
                           "pow with non-constant exponent requires the "
                           "general rule (tracked in docs roadmap)");
            }
            MLK_TRY_VAR(cm1, c(ev.constant.f64 - 1.0));
            MLK_TRY_VAR(um1, bin(MathOp::Pow, u, cm1));
            MLK_TRY_VAR(cu, bin(MathOp::Mul, e, um1));
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(cu);
            ins.push_back(du);
            return g.addNode(MathOp::Mul, ins);
        }
        case MathOp::Sqrt: {
            // (sqrt u)' = u' / (2 sqrt u)
            const ValueId u = n.inputs[0];
            MLK_TRY_VAR(su, un(MathOp::Sqrt, u));
            MLK_TRY_VAR(two, c(2.0));
            MLK_TRY_VAR(den0, bin(MathOp::Mul, two, su));
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(du);
            ins.push_back(den0);
            return g.addNode(MathOp::Div, ins);
        }
        case MathOp::Tanh: {
            // (tanh u)' = (1 - tanh²(u)) * u'
            const ValueId u = n.inputs[0];
            MLK_TRY_VAR(tu, un(MathOp::Tanh, u));
            MLK_TRY_VAR(tu2, bin(MathOp::Mul, tu, tu));
            MLK_TRY_VAR(one, c(1.0));
            SmallVector<ValueId, 4> ms;
            ms.push_back(one);
            ms.push_back(tu2);
            MLK_TRY_VAR(sub, g.addNode(MathOp::Sub, ms));
            MLK_TRY_VAR(du, diffValue(g, u, x, depth + 1));
            SmallVector<ValueId, 4> ins;
            ins.push_back(sub);
            ins.push_back(du);
            return g.addNode(MathOp::Mul, ins);
        }
        case MathOp::Derivative: {
            // d/dx (Derivative(u, x)) — second-order derivative.
            const ValueId u = n.inputs[0];
            return diffValue(g, u, x, depth + 1);
        }
        default:
            return err(ErrorCode::Unimplemented,
                       std::string("no symbolic derivative rule for ") +
                           opName(n.op));
    }
}
}  // namespace

class DerivativeSymbolicPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Capability gate (Part 0): no derivatives, no pass.
        if (!ctx.domainProfile->has(Capability::HasDerivatives)) {
            return err(ErrorCode::UnsupportedCapability,
                       "profile lacks HasDerivatives; cannot lower "
                       "Derivative nodes symbolically", 28);
        }
        r.nodesBefore = graph.liveNodeCount();
        uint32_t lowered = 0;
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::Derivative) {
                continue;
            }
            // n.inputs = {f, x}
            const ValueId f = n.inputs[0];
            const ValueId x = n.inputs[1];
            MLK_TRY_VAR(dfdx, diffValue(graph, f, x, 0));
            graph.replaceOperandUses(n.results[0], dfdx);
            graph.recordEquivalent(n.results[0], dfdx);
            (void)graph.killNode(nid);
            ++lowered;
            ctx.budget.consume();
            if (ctx.budget.exhausted()) {
                return err(ErrorCode::BudgetExceeded,
                           "derivative lowering budget exhausted", 131);
            }
        }
        r.changed = lowered != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_calculus_derivative_symbolic_pass(SymbolTable& symbols) {
    static DerivativeSymbolicPass pass(symbols, "calculus.derivative_symbolic",
                                       PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering, {"type.inferred"},
                 {"calculus.lowered"}, {"analysis.cse"},
                 {kCalcTiers[0], kCalcTiers[1], kCalcTiers[2]});
}

}  // namespace mlk::passes
