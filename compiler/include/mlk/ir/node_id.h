// MLK+ node identifiers. See value_id.h (Rule 15); split into its own header
// so pass code can include only what it names.
#pragma once

#include "mlk/ir/value_id.h"

namespace mlk {

using NodeId = uint32_t;  // (NodeId/kInvalidNodeId live in value_id.h)

}  // namespace mlk
