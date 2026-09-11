// approx.function_lower — implementation-family selection for math
// functions (spec §8.6; libm default, polynomial families only when the
// policy gate allowed approximation — Rule 34).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kApproxTiers[] = {Tier::Tier2, Tier::Tier3};
}  // namespace

class FunctionLowerPass final : public PassBase {
public:
    using PassBase::PassBase;

    Result<PassResult> run(PassContext& ctx, MathGraph& graph) override {
        PassResult r;
        // Implementation family selection: libm is the default; polynomial
        // families are attached ONLY when the policy gate allowed
        // approximation (Rule 34) — the attr carries the candidate family
        // and the ulp bound to verify.
        const bool approxAllowed =
            ctx.domainProfile->has(Capability::HasApproximation) &&
            (ctx.accuracy != nullptr && ctx.accuracy->permitsApproximation());
        const SymbolId familyAttr = ctx.symbols->intern("family");
        const SymbolId libm = ctx.symbols->intern("libm");
        const SymbolId poly = ctx.symbols->intern("poly7");
        for (const NodeId nid : graph.topoOrder()) {
            Node& n = graph.node(nid);
            if (n.flags.test(NodeFlag::Dead)) continue;
            bool isMathFn = false;
            switch (n.op) {  // Rule 78: exhaustive
                case MathOp::Exp: case MathOp::Log: case MathOp::Sin:
                case MathOp::Cos: case MathOp::Tan: case MathOp::Tanh:
                case MathOp::Sqrt: case MathOp::Rsqrt: case MathOp::Erf:
                    isMathFn = true;
                    break;
                default:
                    break;
            }
            if (!isMathFn) continue;
            bool hasFamily = false;
            for (const auto& a : n.attrs) {
                if (a.name == familyAttr) hasFamily = true;
            }
            if (hasFamily) continue;
            Attr a;
            a.name = familyAttr;
            a.value = AttrValue{approxAllowed ? poly : libm};
            n.attrs.push_back(a);
            r.changed = true;
        }
        return r;
    }
};

void register_approx_function_lower_pass(SymbolTable& symbols) {
    static FunctionLowerPass pass(symbols, "approx.function_lower",
                                  PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"approx.gated"},
                 {"approx.lowered"}, {"analysis.cost"},
                 {kApproxTiers[0], kApproxTiers[1]});
}

}  // namespace mlk::passes
