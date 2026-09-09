// MLK+ property inference (property.infer): the pass that makes algebraic
// reasoning possible (spec §8.1: "essential before algebraic rewriting").
//
// Facts are inferred from op classes + operand domains + the Math Domain
// Profile law declarations. Rule 22: anything not definitely known stays
// Unknown — never silently True. Rule 33: floating-point reassociation laws
// come from the profile, not from wishful thinking.
#pragma once

#include "mlk/core/result.h"
#include "mlk/ir/math_graph.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

/// Per-op property table entry (data-driven; Rule 28/77).
struct OpPropertyRule {
    MathOp op;
    /// Property facts derivable when operands are commutative-compatible.
    PropertyId property;
    TriState value;
};

/// Infers facts for every value in the graph; returns facts established.
[[nodiscard]] Result<uint32_t> inferProperties(MathGraph& graph,
                                               const MathDomainProfile& profile);

/// Single-op inference hook (used by the verifier to re-check and by the
/// e-graph to propagate facts into new e-nodes).
[[nodiscard]] FactSet inferNodeFacts(const MathGraph& graph, NodeId node,
                                     const MathDomainProfile& profile);

}  // namespace mlk
