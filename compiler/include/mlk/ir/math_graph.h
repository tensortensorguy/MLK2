// MLK+ MathGraph: the abstract mathematical IR.
//
// Design laws realized here:
//   Rule 15 — index-based graph; no raw pointers for edges.
//   Rule 21 — mathematical equivalence uses equivalence classes; destructive
//             rewriting is forbidden where multiple equivalent forms may be
//             needed. Replacements create NEW values; old ones stay reachable
//             until a versioned, documented lowering decision purges them.
//   Rule 23 — mathematical state only. No hardware details live here.
//   Rule 24 — stable content hashes + versioned serialization.
//   Rule 89 — every mutation bumps a version so specializations can depend on
//             it and be invalidated.
//   Rule 140 — effect chain maintained on mutation.
#pragma once

#include <cassert>
#include <cstdint>
#include <optional>

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/core/hash_map.h"
#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/effect/effect_chain.h"
#include "mlk/effect/effect_kind.h"
#include "mlk/ir/attrs.h"
#include "mlk/ir/math_op.h"
#include "mlk/ir/node.h"
#include "mlk/ir/value.h"
#include "mlk/type/math_type.h"

namespace mlk {

class MathGraph {
public:
    /// The graph interns generated names through a SymbolTable (Rule 16).
    /// A process-wide synchronized default is provided (Rule 144: allowed
    /// global state), but frontends should pass their own.
    /// Well-known interned attribute names (Rule 16: interned once, then
    /// only ids flow — never string comparison in passes).
    struct WellKnownSymbols {
        SymbolId shapeDims{kInvalidSymbolId};
        SymbolId value{kInvalidSymbolId};
        SymbolId axis{kInvalidSymbolId};
        SymbolId epsilon{kInvalidSymbolId};
    };

    explicit MathGraph(SymbolTable* symbols = defaultSymbolTable())
        : symbols_(symbols) {
        wk_.shapeDims = symbols_->intern("shape_dims");
        wk_.value = symbols_->intern("value");
        wk_.axis = symbols_->intern("axis");
        wk_.epsilon = symbols_->intern("epsilon");
        values_.reserve(constants::kInitialValueCapacity);
        nodes_.reserve(constants::kInitialNodeCapacity);
    }

    [[nodiscard]] const WellKnownSymbols& wk() const noexcept { return wk_; }

    [[nodiscard]] static SymbolTable* defaultSymbolTable() {
        static SymbolTable table;  // internally synchronized (Rule 144)
        return &table;
    }

    [[nodiscard]] SymbolTable& symbols() noexcept { return *symbols_; }
    [[nodiscard]] const SymbolTable& symbols() const noexcept { return *symbols_; }

    // --- Construction API ----------------------------------------------------
    [[nodiscard]] ValueId addPlaceholder(SymbolId name, MathType type);
    [[nodiscard]] ValueId addVariable(SymbolId name, MathType type);
    /// Adds a scalar constant (int64 or double payload).
    [[nodiscard]] ValueId addConstant(double v, MathType type);
    [[nodiscard]] ValueId addIntConstant(int64_t v, MathType type);
    [[nodiscard]] ValueId addSymbol(SymbolId name, MathType type);

    [[nodiscard]] Result<ValueId> addNode(MathOp op,
                                          const SmallVector<ValueId, 4>& inputs,
                                          AttrList attrs = {});

    // --- Accessors -------------------------------------------------------------
    [[nodiscard]] Node& node(NodeId id) {
        return nodes_[assertIndex(id, nodes_.size())];
    }
    [[nodiscard]] const Node& node(NodeId id) const {
        return nodes_[assertIndex(id, nodes_.size())];
    }
    [[nodiscard]] Value& value(ValueId id) {
        return values_[assertIndex(id, values_.size())];
    }
    [[nodiscard]] const Value& value(ValueId id) const {
        return values_[assertIndex(id, values_.size())];
    }

    [[nodiscard]] uint32_t numNodes() const noexcept {
        return static_cast<uint32_t>(nodes_.size());
    }
    [[nodiscard]] uint32_t numValues() const noexcept {
        return static_cast<uint32_t>(values_.size());
    }
    [[nodiscard]] const std::vector<Node>& nodes() const noexcept {
        return nodes_;
    }
    [[nodiscard]] const std::vector<Value>& values() const noexcept {
        return values_;
    }
    [[nodiscard]] std::vector<Node>& nodesRef() noexcept { return nodes_; }
    [[nodiscard]] std::vector<Value>& valuesRef() noexcept { return values_; }

    // Outputs
    void addOutput(ValueId v) {
        if (!outputs_.contains(v)) outputs_.push_back(v);
    }
    [[nodiscard]] const SmallVector<ValueId, 4>& outputs() const {
        return outputs_;
    }
    [[nodiscard]] SmallVector<ValueId, 4>& outputsRef() { return outputs_; }

    // Use-def ---------------------------------------------------------------
    /// Users of a value (nodes consuming it). Maintained on construction.
    [[nodiscard]] const SmallVector<NodeId, 4>& users(ValueId v) const {
        return users_[assertIndex(v, users_.size())];
    }
    void rebuildUsers();

    // Versioning (Rule 89) ----------------------------------------------------
    [[nodiscard]] uint64_t version() const noexcept { return version_; }
    void bumpVersion() noexcept { ++version_; }

    // Effect chain (Rule 140) ---------------------------------------------------
    [[nodiscard]] const EffectChain& effectChain() const noexcept {
        return effects_;
    }
    EffectChain& effectChainRef() noexcept { return effects_; }

    // Hashing (Rule 24) ---------------------------------------------------------
    /// Stable structural hash of the graph (cached; invalidated on mutation).
    [[nodiscard]] HashValue hash() const;

    // Replacement machinery (Rule 21: new values, never in-place meaning) ----
    /// Registers `newV` as the current representative for the equivalence of
    /// `oldV` without deleting `oldV`. The original stays recoverable.
    void recordEquivalent(ValueId oldV, ValueId newV);

    /// Rewrites every use of oldV to newV (structural rewiring; semantics
    /// preserved by the caller's legality proof). Rebuilds use-def lists.
    void replaceOperandUses(ValueId oldV, ValueId newV);

    /// Marks a value dead after a versioned, documented lowering decision.
    /// Requires the caller to name the decision (auditability).
    void markLoweredAway(ValueId v, SymbolId decision);

    [[nodiscard]] bool isDead(ValueId v) const {
        return value(v).flags.test(ValueFlag::Dead);
    }
    [[nodiscard]] bool isNodeDead(NodeId n) const {
        return node(n).flags.test(NodeFlag::Dead);
    }
    [[nodiscard]] uint32_t liveNodeCount() const {
        uint32_t count = 0;
        for (const auto& n : nodes_) {
            if (!n.flags.test(NodeFlag::Dead)) ++count;
        }
        return count;
    }

    /// Marks a node dead: splices the effect chain, keeps storage for
    /// provenance (Rule 21). Returns false if already dead.
    bool killNode(NodeId n);
    [[nodiscard]] bool isLoweredAway(ValueId v) const {
        return loweredAway_.contains(v);
    }

    /// Representative mapping for equivalence classes recorded so far.
    [[nodiscard]] ValueId representative(ValueId v) const {
        ValueId cur = v;
        // Path walk with bounded depth (acyclic by construction; the bound
        // is defensive against accidental cycles from buggy passes).
        for (uint32_t depth = 0; depth < constants::kMaxEquivalenceDepth;
             ++depth) {
            const ValueId* next = representative_.find(cur);
            if (next == nullptr) return cur;
            cur = *next;
        }
        return cur;
    }

    [[nodiscard]] const OpenHashMap<ValueId, ValueId>& equivalenceMap() const {
        return representative_;
    }

    // Structural queries -------------------------------------------------------
    /// Topological order of all nodes (producers before consumers).
    [[nodiscard]] SmallVector<NodeId, 16> topoOrder() const;

    /// Renumbers live nodes into dataflow order (producers before
    /// consumers), remapping value.producer, users_ lists, and the effect
    /// chain. Value ids are NOT renumbered — only node ids move.
    /// Restores the "node id order is topological" invariant after rewrite
    /// passes that append replacement nodes and rewire earlier consumers
    /// (Rule 47: the verifier and the Tier-0 interpreter both rely on the
    /// invariant). Returns the number of nodes whose id changed.
    [[nodiscard]] uint32_t renumberTopological();

    [[nodiscard]] SmallVector<NodeId, 4> valueUsers(ValueId v) const {
        SmallVector<NodeId, 4> out;
        if (v < users_.size()) out = users_[v];
        return out;
    }

private:
    [[nodiscard]] static std::size_t assertIndex(uint32_t id,
                                                 std::size_t size) noexcept {
        assert(id < size && "graph index out of range");
        // Rule 25: [[assume]] documents the invariant for release builds.
        [[assume(id < size)]];
        return static_cast<std::size_t>(id);
    }

    [[nodiscard]] ValueId allocateValue(Value v);
    [[nodiscard]] ValueId addInputLike(ValueKind kind, SymbolId name,
                                       MathType type);
    void attachEffects(Node& n);

    SymbolTable* symbols_;
    WellKnownSymbols wk_{};
    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::vector<SmallVector<NodeId, 4>> users_;
    SmallVector<ValueId, 4> outputs_{};
    EffectChain effects_{};
    OpenHashMap<ValueId, ValueId> representative_{};
    OpenHashSet<ValueId> loweredAway_{};
    uint64_t version_{0};
    mutable std::optional<HashValue> hashCache_;
};

}  // namespace mlk
