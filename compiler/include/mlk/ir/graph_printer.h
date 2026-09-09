// MLK+ graph printer (ir/graph_printer.h).
//
// Deterministic S-expression printer used by golden tests, tools, and docs.
// Cold path: std::string lives here, never in the IR (Rule 16).
#pragma once

#include <string>

#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"

namespace mlk {

struct PrintOptions {
    bool annotateTypes{false};
    bool annotateFacts{false};
    bool annotateEffects{false};
    bool printDead{false};  // include dead nodes (Rule 21 provenance)
};

/// Prints the graph as a deterministic S-expression:
///   (sin (add (mul x x) (mul 3 x)))
[[nodiscard]] std::string printGraph(const MathGraph& graph,
                                     const PrintOptions& opts,
                                     SymbolTable& symbols);

/// One-line summary for diagnostics: e.g. `sin(add(mul(x,x), mul(3,x)))`.
[[nodiscard]] std::string printValueExpr(const MathGraph& graph, ValueId v,
                                         SymbolTable& symbols);

}  // namespace mlk
