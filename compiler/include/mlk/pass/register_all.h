// MLK+ pass registration entry point (public face of the pass catalog).
#pragma once

#include "mlk/core/symbol_table.h"

namespace mlk::passes {

/// Registers every MLK+ pass with the global PassRegistry. Deterministic;
/// call once per process before building pipelines.
void registerAllPasses(SymbolTable& symbols);

}  // namespace mlk::passes
