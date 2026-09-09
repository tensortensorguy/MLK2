// MLK+ effect chain (Rule 140: ordering of effects is explicit).
//
// The graph maintains an ordered chain over effecting nodes. Transformations
// that remove or insert effecting nodes must splice the chain via the graph
// API; the effect verifier validates continuity.
#pragma once

#include "mlk/core/small_vector.h"
#include "mlk/ir/value_id.h"

namespace mlk {

class MathGraph;

/// EffectChain: program-ordered list of effecting node ids.
class EffectChain {
public:
    void append(NodeId n) { order_.push_back(n); }

    /// Removes a node, preserving relative order of the rest.
    void remove(NodeId n) {
        for (std::size_t i = 0; i < order_.size(); ++i) {
            if (order_[i] == n) {
                order_.eraseAt(i);
                return;
            }
        }
    }

    [[nodiscard]] bool contains(NodeId n) const {
        for (const NodeId m : order_) {
            if (m == n) return true;
        }
        return false;
    }

    [[nodiscard]] const SmallVector<NodeId, 4>& order() const {
        return order_;
    }
    [[nodiscard]] bool empty() const { return order_.empty(); }
    [[nodiscard]] std::size_t size() const { return order_.size(); }

    void clear() { order_.clear(); }

private:
    SmallVector<NodeId, 4> order_{};
};

}  // namespace mlk
