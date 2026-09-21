// math.strength_reduce — strength reduction with bit-exact legality
// (spec §8.2; Rule 33: no rewrite without its exactness argument).
//
// Div(x, c) -> Mul(x, 1/c) for compile-time FP constants c that are powers
// of two. Legality THEOREM (not a sample claim): for c = ±2^e the reciprocal
// 1/c = ±2^-e is exactly representable, so x/c and x*(1/c) are the SAME
// real number; IEEE rounding is a function of the real result alone, hence
// the two forms are bit-identical for every x — including subnormals (both
// scale the exponent by e), overflow (same real, same saturated result),
// and signed zeros (sign of the real result is identical).
// Guard rails: (a) the reciprocal must itself be exactly representable —
// c = 2^-1074 has reciprocal 2^1074 which OVERFLOWS to +inf, where x/c
// stays finite for small x, so degenerate reciprocals are skipped;
// (b) the rewrite applies to F64 results only today — the evaluation
// precision must make both c and 1/c exact, and that is the profile the
// interpreter and the emitters currently guarantee bit-exactly. pow(x, 2) ->
// x*x is deliberately NOT here: it is bit-exact only if the evaluator's pow
// is correctly rounded for that case, which is a property of the linked
// libm, not of IEEE — the e-graph's pow_square rule stays gated on the
// profile's IEEE declaration (Rule 90) and differential verification.
#include "../passes_common.h"
#include "math_rewrite_utils.h"

#include <cmath>

namespace mlk::passes {

namespace {
constexpr Tier kMathTiers[] = {Tier::Tier1, Tier::Tier2, Tier::Tier3};
}  // namespace

class StrengthReducePass final : public PassBase {
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
            if (n.op != MathOp::Div || n.numInputs() != 2) continue;
            if (graph.value(n.results[0]).type.dtype != Dtype::F64) {
                continue;  // FP theorem, F64 evaluation precision (see header)
            }
            double c = 0.0;
            if (!isFpConst(graph, n.inputs[1], &c)) continue;
            if (c == 0.0) continue;  // division by zero: leave semantics
            int e = 0;
            // frexp decomposes c = m * 2^e with m in [0.5, 1); a power of
            // two has m == ±0.5 exactly. NaN/Inf yield m == c — excluded.
            const double m = std::frexp(c, &e);
            if (m != 0.5 && m != -0.5) continue;
            // 1/c is EXACT for a power of two (the real result 2^(1-e) is
            // representable, so the correctly-rounded division returns it
            // bit-exactly — sign included). The previous ldexp(1.0, -e)
            // was off by one frexp mantissa (8 -> 0.0625) and dropped the
            // sign; caught by the pass's own differential test.
            const double recip = 1.0 / c;
            if (!std::isfinite(recip) || recip == 0.0) {
                continue;  // 2^-1074's reciprocal overflows: keep the div
            }
            // The reciprocal rides with the SAME type as the divisor
            // constant (scalar for the broadcast case, tensor for a
            // constant-tensor divisor); the Mul result's type transfers
            // from the Div result inside replaceResult.
            const MathType& type = graph.value(n.inputs[1]).type;
            const ValueId recipV = graph.addConstant(recip, type);
            MLK_TRY_VAR(mulV,
                        graph.addNode(MathOp::Mul, {n.inputs[0], recipV}));
            MLK_TRYV(replaceResult(graph, nid, mulV));
            ++edits;
            ctx.budget.consume();
            if (ctx.budget.exhausted()) {
                return err(ErrorCode::BudgetExceeded,
                           "strength_reduce budget exhausted", 131);
            }
        }
        r.changed = edits != 0;
        r.nodesAfter = graph.liveNodeCount();
        return r;
    }
};

void register_math_strength_reduce_pass(SymbolTable& symbols) {
    static StrengthReducePass pass(symbols, "math.strength_reduce",
                                   PassKind::Transform);
    registerPass(symbols, pass, PassKind::Transform, {"property.inferred"},
                 {"math.strength_reduced"}, {},
                 {kMathTiers[0], kMathTiers[1], kMathTiers[2]});
}

}  // namespace mlk::passes
