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
    // Trigger full-graph hash, which populates per-value caches for
    // NodeResult values.
    (void)graphHash(graph);
    const Value& after = graph.value(v);
    if (after.hashValid) return after.hash;
    // Constants/placeholders/variables/symbols are never stamped by
    // graphHash(); their identity is kind + type + payload. Returning the
    // unstamped cache slot here previously collapsed all such values to one
    // hash (P0: cse merged mul(x,x) with mul(3,x); see structuralValueHash).
    return structuralValueHash(after);
}

}  // namespace mlk
