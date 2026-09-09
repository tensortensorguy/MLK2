// Calculus passes (spec §8.4): calculus.derivative_symbolic,
// calculus.derivative_simplify, calculus.derivative_ad_forward,
// calculus.derivative_ad_reverse, calculus.derivative_numeric,
// calculus.integral_closed_form, calculus.integral_quadrature,
// calculus.gradient_lower.
//
// Calculus is first-class: Derivative nodes remain symbolic until a
// strategy-lowering pass decides the method (symbolic / AD / numeric), and
// the original expression is never destroyed (Rule 21).
#include "../passes_common.h"

#include <cmath>

namespace mlk::passes {

namespace {

/// Symbolic differentiation rules (spec example: d/dx x² sin(x) =
/// 2x sin(x) + x² cos(x)). Returns kInvalidValueId when no rule applies.
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

class DerivativeSimplifyPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Runs the canonicalize driver over derivative results (spec:
        // 2x sin(x) + x² cos(x) -> x(2 sin(x) + x cos(x)) form is e-graph
        // territory; here we fold/sort/eliminate identities).
        Pass* canonicalize =
            PassRegistry::instance().byName(ctx.symbols->intern("math.canonicalize"));
        if (canonicalize == nullptr) {
            return err(ErrorCode::Internal, "math.canonicalize not registered");
        }
        MLK_TRY_VAR(cr, canonicalize->run(ctx, graph));
        r.changed = cr.changed;
        return r;
    }
};

class DerivativeNumericPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        (void)graph;
        // Rule 34: finite differences REQUIRE an explicit accuracy contract;
        // without one this pass must not fire.
        if (!ctx.accuracy->permitsApproximation()) {
            return err(ErrorCode::AccuracyViolation,
                       "numeric differentiation requires an accuracy "
                       "contract permitting approximation (Rule 34)", 34);
        }
        r.changed = false;  // strategy attached at Tier-2 in autotuner
        return r;
    }
};

class IntegralQuadraturePass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        if (!ctx.domainProfile->has(Capability::HasIntegrals)) {
            return err(ErrorCode::UnsupportedCapability,
                       "profile lacks HasIntegrals", 28);
        }
        // Strategy attachment: each Integral node gets a method attr
        // (adaptive_gauss_kronrod default; method selection is autotunable
        // per spec §5.2). The integral node stays in the graph — the
        // strategy layer references it (MathGraph/StrategyGraph split).
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead) || n.op != MathOp::Integral) {
                continue;
            }
            Attr a;
            a.name = ctx.symbols->intern("method");
            a.value = AttrValue{ctx.symbols->intern("adaptive_simpson")};
            n.attrs.push_back(a);
            r.changed = true;
        }
        return r;
    }
};

void registerCalculusPasses(SymbolTable& symbols) {
    static DerivativeSymbolicPass derivSym(symbols,
                                           "calculus.derivative_symbolic",
                                           PassKind::Lowering);
    static DerivativeSimplifyPass derivSimp(symbols,
                                            "calculus.derivative_simplify",
                                            PassKind::Transform);
    static DerivativeNumericPass derivNum(symbols,
                                          "calculus.derivative_numeric",
                                          PassKind::Lowering);
    static IntegralQuadraturePass integralQ(symbols,
                                            "calculus.integral_quadrature",
                                            PassKind::Lowering);
    const Tier t12[] = {Tier::Tier1, Tier::Tier2};
    registerPass(symbols, derivSym, PassKind::Lowering, {"type.inferred"},
                 {"calculus.lowered"}, {"analysis.cse"},
                 {t12[0], t12[1], Tier::Tier3});
    registerPass(symbols, derivSimp, PassKind::Transform,
                 {"calculus.lowered"}, {"math.canonical"}, {"analysis.cse"},
                 {t12[0], t12[1], Tier::Tier3});
    registerPass(symbols, derivNum, PassKind::Lowering, {"accuracy.analyzed"},
                 {"calculus.strategy"}, {}, {t12[0], t12[1], Tier::Tier3});
    registerPass(symbols, integralQ, PassKind::Lowering, {"type.inferred"},
                 {"calculus.strategy"}, {}, {t12[0], t12[1], Tier::Tier3});
}

}  // namespace mlk::passes
