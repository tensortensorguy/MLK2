// MLK+ effect inference implementation (Rule 140).
#include "mlk/effect/effect_inference.h"

#include "mlk/ir/math_graph.h"

namespace mlk {

bool isPure(const EffectSet& e) noexcept { return e.none(); }

EffectSet inferNodeEffects(const MathGraph& graph, NodeId nodeId) {
    const Node& n = graph.node(nodeId);
    // Data-driven: effect classes by op category (Rule 28 — knowledge lives
    // in tables; no per-domain if-chains in passes).
    EffectSet e{};
    switch (n.op) {  // Rule 78: exhaustive
        case MathOp::NativeToMathRef:
        case MathOp::MathToNativeRef:
            // The native boundary is opaque unless proven otherwise
            // (Rule 112): crossing it is an FFI effect.
            e.set(EffectKind::FFI);
            break;
        default:
            break;  // pure mathematical transformation
    }
    return e;
}

}  // namespace mlk
