// MLK+ e-graph (Rule 21: "Mathematical Equivalence Must Use Equivalence
// Classes"). Equality saturation per spec §4/§Pass 4: build classes from the
// graph, saturate with cost-aware rewrite rules under strict budget, extract
// best/many candidates.
//
// This is a compact e-class union-find over enodes: an enode is
// (MathOp, attr-hash, child eclass ids). Congruence closure emerges from
// dedup: merging classes re-keys their enodes in the lookup table.
#pragma once

#include <cstdint>
#include <vector>

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/core/hash_map.h"
#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/ir/math_graph.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

using EClassId = uint32_t;
inline constexpr EClassId kInvalidEClass = constants::kInvalidId;

/// Read-only view of a leaf value (for rule predicates; Rule 16: ids only).
struct LeafView {
    ValueKind kind{ValueKind::Constant};
    bool isInt{false};
    double f64{0.0};
    int64_t i64{0};
};

[[nodiscard]] inline LeafView leafView(const MathGraph& g, ValueId v) {
    const Value& val = g.value(v);
    LeafView lv;
    lv.kind = val.kind;
    lv.isInt = val.constant.isInt;
    lv.f64 = val.constant.f64;
    lv.i64 = val.constant.i64;
    return lv;
}

struct ENode {
    MathOp op{MathOp::Add};
    HashValue attrHash{0};
    SmallVector<EClassId, 4> children{};
    /// Provenance: rule that created this enode (Rule 52/54: traceable).
    SymbolId createdBy{kInvalidSymbolId};

    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashU64(static_cast<uint64_t>(op));
        h = hashCombine(h, attrHash);
        for (std::size_t i = 0; i < children.size(); ++i) {
            h = hashCombine(h, hashU64(children[i]));
        }
        return h;
    }
    [[nodiscard]] bool operator==(const ENode& o) const {
        return op == o.op && attrHash == o.attrHash && children == o.children;
    }
};

struct EGraphConfig {
    std::size_t maxNodes{constants::kEgraphDefaultMaxNodes};
    uint32_t maxIterations{constants::kEgraphDefaultMaxIterations};
};

/// Budgeted, deterministic e-graph.
class EGraph {
public:
    explicit EGraph(const MathDomainProfile& profile,
                    const EGraphConfig& cfg = EGraphConfig{})
        : profile_(profile), config_(cfg) {}

    /// Imports the graph's expression trees (Rule 21: original forms live).
    [[nodiscard]] Result<EClassId> importGraph(const MathGraph& graph,
                                               ValueId root);

    [[nodiscard]] EClassId find(EClassId c) const;
    [[nodiscard]] Result<EClassId> unionClasses(EClassId a, EClassId b,
                                                SymbolId rule);
    [[nodiscard]] Result<EClassId> addENode(ENode enode);
    [[nodiscard]] std::size_t numENodes() const { return enodes_.size(); }
    [[nodiscard]] std::size_t numClasses() const { return liveClasses_; }
    [[nodiscard]] bool saturated() const { return saturated_; }

    /// Applies one rewrite pass over all enodes; returns whether anything
    /// merged. Deterministic order (enode insertion order; Rule 143).
    [[nodiscard]] Result<bool> saturateOnce(SymbolTable& symbols);

    /// Cost of an enode subtree (instruction-count style with op weights;
    /// Rule 55).
    [[nodiscard]] double subtreeCost(EClassId c, std::size_t depth = 0) const;

    /// Greedy bottom-up extraction: picks the cheapest enode per class and
    /// materializes the extracted expression into `out` (a new MathGraph).
    [[nodiscard]] Result<ValueId> extract(SymbolTable& symbols,
                                          MathGraph& out,
                                          EClassId rootClass);

    /// Emits every distinct equivalent form of `rootClass` (compile=INF
    /// multi-candidate extraction per spec §Pass 5).
    [[nodiscard]] Result<SmallVector<ValueId, 8>> extractMany(
        SymbolTable& symbols, MathGraph& out, EClassId rootClass,
        std::size_t maxForms);

private:
    [[nodiscard]] EClassId makeClass();
    [[nodiscard]] Result<EClassId> internENode(const ENode& e);
    [[nodiscard]] Result<EClassId> buildFromValue(const MathGraph& graph,
                                                  ValueId v);
    void rebuildLookup();
    [[nodiscard]] Result<ValueId> materialize(SymbolTable& symbols,
                                              MathGraph& out, EClassId c,
                                              OpenHashMap<EClassId, ValueId>& memo,
                                              uint32_t depth);

    const MathDomainProfile& profile_;
    EGraphConfig config_;

    std::vector<ENode> enodes_{};
    /// Union-find parent array over classes.
    std::vector<EClassId> parent_{};
    /// eclass -> enode ids belonging to it (post-find canonical).
    std::vector<SmallVector<uint32_t, 2>> classMembers_{};
    OpenHashMap<HashValue, EClassId> lookup_{};
    /// Enode id -> owning (canonical) class.
    std::vector<EClassId> enodeClass_{};
    /// Leaf class -> imported source ValueId (Rule 21 provenance).
    OpenHashMap<EClassId, ValueId> leafSource_{};
    std::size_t liveClasses_{0};
    bool saturated_{false};
    EClassId rootClass_{kInvalidEClass};
    ValueId importedRoot_{kInvalidValueId};
    const MathGraph* imported_{nullptr};
};

}  // namespace mlk
