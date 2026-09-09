// Property inference implementation.
// The rule table below encodes mathematical facts per op. Domain-dependent
// laws (commutativity of mul on matrices, associativity of FP add) are gated
// by the Math Domain Profile — Rule 28/33 honored by construction.
#include "mlk/property/property_inference.h"

#include "mlk/effect/effect_inference.h"

namespace mlk {

namespace {

/// Range facts for math functions (spec §Pass 3 examples: exp(x) > 0,
/// sin(x) in [-1,1], x² >= 0).
void applyFunctionFacts(MathOp op, FactSet& facts) {
    switch (op) {
        case MathOp::Exp:
            facts.setTriState(PropertyId::Positive, TriState::True);
            facts.setTriState(PropertyId::MonotonicIncreasing, TriState::True);
            facts.setRange(Interval::nonNegative());
            facts.setTriState(PropertyId::Differentiable, TriState::True);
            break;
        case MathOp::Sin:
        case MathOp::Cos:
            facts.setRange(Interval::closed(-1.0, 1.0));
            facts.setTriState(PropertyId::Periodic, TriState::True);
            facts.setTriState(PropertyId::Differentiable, TriState::True);
            break;
        case MathOp::Tanh:
            facts.setRange(Interval::closed(-1.0, 1.0));
            facts.setTriState(PropertyId::MonotonicIncreasing, TriState::True);
            facts.setTriState(PropertyId::Differentiable, TriState::True);
            break;
        case MathOp::Sqrt:
            facts.setRange(Interval::nonNegative());
            facts.setTriState(PropertyId::MonotonicIncreasing, TriState::True);
            break;
        case MathOp::Erf:
            facts.setRange(Interval::closed(-1.0, 1.0));
            facts.setTriState(PropertyId::MonotonicIncreasing, TriState::True);
            break;
        case MathOp::Gelu:
            facts.setTriState(PropertyId::Differentiable, TriState::True);
            break;
        default:
            break;
    }
}

}  // namespace

FactSet inferNodeFacts(const MathGraph& graph, NodeId nodeId,
                       const MathDomainProfile& profile) {
    const Node& n = graph.node(nodeId);
    FactSet facts;

    // Purity first (Rule 87): only provably pure expressions may be folded.
    facts.setTriState(PropertyId::Pure,
                      isPure(n.effects) ? TriState::True : TriState::False);

    const bool tensorDomain = [&] {
        for (const ValueId in : n.inputs) {
            if (graph.value(in).type.tensor.has_value()) return true;
        }
        return false;
    }();

    switch (n.op) {  // Rule 78: exhaustive
        case MathOp::Add:
            // Commutativity of add holds in fields; profile gate for exact
            // vs FP (FP add IS commutative; it is associativity that is not).
            facts.setTriState(PropertyId::Commutative, TriState::True);
            // Associativity only when the profile allows reassociation
            // (Rule 33: FP add not associative without contract).
            facts.setTriState(PropertyId::Associative,
                              profile.lawAssociativeAdd);
            break;
        case MathOp::Mul:
            // Scalar/elementwise mul commutes; matrix mul does NOT
            // (Rule 33: "A × B == B × A" is exactly the betrayal).
            facts.setTriState(PropertyId::Commutative,
                              tensorDomain ? TriState::False
                                           : profile.lawCommutativeMul);
            break;
        case MathOp::MatMul:
            facts.setTriState(PropertyId::Commutative, TriState::False);
            break;
        case MathOp::Div:
        case MathOp::Sub:
            facts.setTriState(PropertyId::Commutative, TriState::False);
            break;
        case MathOp::Transpose:
            facts.setTriState(PropertyId::Invertible, TriState::True);
            facts.setTriState(PropertyId::Idempotent, TriState::True);
            break;
        case MathOp::Reshape:
        case MathOp::Broadcast:
            facts.setTriState(PropertyId::Contiguous, TriState::True);
            break;
        default:
            break;
    }

    applyFunctionFacts(n.op, facts);
    return facts;
}

Result<uint32_t> inferProperties(MathGraph& graph,
                                 const MathDomainProfile& profile) {
    uint32_t established = 0;
    for (const NodeId nid : graph.topoOrder()) {
        const Node& n = graph.node(nid);
        if (n.flags.test(NodeFlag::Dead)) continue;
        FactSet facts = inferNodeFacts(graph, nid, profile);
        // Result value inherits node facts (facts are per-value per spec).
        Value& result = graph.value(n.results[0]);
        const uint32_t before = static_cast<uint32_t>(result.facts.size());
        for (const auto& f : facts.all()) {
            result.facts.set(f);
        }
        established += static_cast<uint32_t>(result.facts.size()) - before;
    }
    graph.bumpVersion();
    return established;
}

}  // namespace mlk
