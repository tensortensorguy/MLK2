// calculus.integral_quadrature — quadrature strategy attachment for
// Integral nodes (spec §8.4; method selection is autotunable per §5.2;
// the Integral node stays in the graph — MathGraph/StrategyGraph split).
#include "../passes_common.h"

namespace mlk::passes {

namespace {
constexpr Tier kCalcTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

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

void register_calculus_integral_quadrature_pass(SymbolTable& symbols) {
    static IntegralQuadraturePass pass(symbols,
                                       "calculus.integral_quadrature",
                                       PassKind::Lowering);
    registerPass(symbols, pass, PassKind::Lowering, {"type.inferred"},
                 {"calculus.strategy"}, {},
                 {kCalcTiers[0], kCalcTiers[1], kCalcTiers[2]});
}

}  // namespace mlk::passes
