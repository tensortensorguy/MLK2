// MLK+ graph identifiers (Rule 15: Index-Based Graph).
//
// Never use raw pointers (Node*, Value*) for edges in the mathematical
// graph. All node and value references use 32-bit integer indices. This cuts
// memory footprint, increases cache capacity, makes the IR trivially
// serializable, and avoids pointer invalidation during arena reallocation.
#pragma once

#include <cstdint>

#include "mlk/core/constants.h"

namespace mlk {

/// Handle to a mathematical transformation node.
using NodeId = uint32_t;

/// Handle to a typed mathematical object (scalar, tensor, function, ...).
using ValueId = uint32_t;

inline constexpr NodeId kInvalidNodeId = constants::kInvalidId;
inline constexpr ValueId kInvalidValueId = constants::kInvalidId;

}  // namespace mlk
