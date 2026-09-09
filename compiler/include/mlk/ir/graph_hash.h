// MLK+ graph hashing API (Rule 24: every graph has a stable content hash).
// The hash is structural: op + attrs + operand hashes, computed bottom-up.
// It is independent of creation order for identical math (commutative
// operand sorting happens in math.commutative_sort before hashing matters).
#pragma once

#include "mlk/core/hash.h"
#include "mlk/ir/math_graph.h"

namespace mlk {

/// Stable structural hash of the whole graph.
[[nodiscard]] inline HashValue graphHash(const MathGraph& graph) {
    return graph.hash();
}

/// Stable hash of a single value's subtree (cached on the value).
[[nodiscard]] inline HashValue valueHash(const MathGraph& graph, ValueId v) {
    const Value& val = graph.value(v);
    if (val.hashValid) return val.hash;
    // Trigger full-graph hash, which populates per-value caches.
    graphHash(graph);
    return graph.value(v).hash;
}

}  // namespace mlk
