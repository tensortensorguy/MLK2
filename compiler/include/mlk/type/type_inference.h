// MLK+ type inference pass interface (type.infer / shape.infer).
// Infers domain, dtype, kind, shape, broadcast compatibility, and promotion
// validity. Rule 39: mismatched types are ERRORS, never implicitly coerced —
// the frontend must insert explicit conversion nodes.
#pragma once

#include "mlk/core/result.h"
#include "mlk/ir/math_graph.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

/// Infers/validates types for every node result in the graph.
/// Returns the number of values whose type was (re)assigned.
[[nodiscard]] Result<uint32_t> inferTypes(MathGraph& graph,
                                          const MathDomainProfile& profile);

/// Shape-only inference (after type.infer has established element dtypes).
[[nodiscard]] Result<uint32_t> inferShapes(MathGraph& graph);

/// Result type of a single op application; used by inference and by the
/// verifier to re-check derived types. nullopt = type error (diagnosed by
/// caller with full context).
[[nodiscard]] std::optional<MathType> inferResultType(
    const MathGraph& graph, NodeId node, const MathDomainProfile* profile);

}  // namespace mlk
