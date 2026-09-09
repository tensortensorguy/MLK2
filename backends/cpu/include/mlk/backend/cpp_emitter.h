// MLK+ C++ source emitter (spec §12: KernelIR -> C++ template/source for
// clang/gcc compilation). Emits portable scalar/SIMD-friendly loops from the
// KernelModule. Rule 148: target features are gated and recorded.
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/kernel_ir.h"

namespace mlk {

/// Emits standalone C++ source for a KernelModule (AOT artifact form).
[[nodiscard]] Result<std::string> emitCppSource(const KernelModule& kernel,
                                                SymbolTable& symbols);

}  // namespace mlk
