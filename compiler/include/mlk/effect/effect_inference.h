// MLK+ effect inference (Rule 140): derive the effect set of a node from its
// op, its operands' types, and the Math Domain Profile capability gates.
#include "mlk/core/symbol_table.h"
#include "mlk/effect/effect_kind.h"
#include "mlk/ir/math_op.h"
#include "mlk/ir/node_id.h"

namespace mlk {

class MathGraph;

/// Computes the effect set for a node. Pure unless the op class declares
/// otherwise (data-driven, Rule 28: no per-domain if-chains in passes).
[[nodiscard]] EffectSet inferNodeEffects(const MathGraph& graph, NodeId node);

/// Pure ops have no observable effect and are reorderable/cse-able.
[[nodiscard]] bool isPure(const EffectSet& e) noexcept;

}  // namespace mlk
