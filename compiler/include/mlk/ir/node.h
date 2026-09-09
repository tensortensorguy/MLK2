// MLK+ Node: a mathematical operation or transformation.
//
// A Node references its operand values by index (Rule 15) and carries:
//   - op (MathOp)
//   - inputs (SmallVector — Rule 19)
//   - attrs (interned, Rule 16)
//   - effects (Rule 140)
//   - speculation metadata when the node embeds a profile-driven assumption
//     (Rules 5, 141: GraphState attachment + complete guard metadata)
#pragma once

#include "mlk/core/flags.h"
#include "mlk/core/hash.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/effect/effect_kind.h"
#include "mlk/ir/attrs.h"
#include "mlk/ir/node_id.h"
#include "mlk/ir/value_id.h"
#include "mlk/property/fact.h"

namespace mlk {

enum class NodeFlag : uint8_t {
    Canonical,      // already in canonical form (math.canonicalize)
    Folded,         // constant-folded
    Speculative,    // embeds a guarded speculation (Rule 141)
    Dead,           // marked dead by DCE (kept for provenance until purge)
    HasGraphState,  // GraphState attachment present (Rule 5)
    kCount,
};
using NodeFlags = Flags<NodeFlag>;

/// Sources of a speculation's evidence (Rule 141).
enum class SpeculationSource : uint8_t {
    None = 0,
    Profile,
    Benchmark,
    StaticProof,
};

/// Confidence attached to profile/benchmark evidence (Rule 63).
struct SpeculationInfo {
    SpeculationSource source{SpeculationSource::None};
    double confidence{0.0};          // [0,1]
    NodeId guardPlan{kInvalidNodeId};
    uint32_t graphStateId{constants::kInvalidId};  // GraphState attach (Rule 5)
    SymbolId fallbackTarget{kInvalidSymbolId};
    SymbolId invalidationDependency{kInvalidSymbolId};
    double costEstimate{0.0};
};

struct Node {
    NodeId id{kInvalidNodeId};
    MathOp op{MathOp::Add};
    SmallVector<ValueId, 4> inputs{};
    SmallVector<ValueId, 2> results{};
    AttrList attrs{};
    EffectSet effects{};
    NodeFlags flags{};
    FactSet facts{};
    SpeculationInfo speculation{};

    [[nodiscard]] uint32_t numInputs() const noexcept {
        return static_cast<uint32_t>(inputs.size());
    }
};

}  // namespace mlk
