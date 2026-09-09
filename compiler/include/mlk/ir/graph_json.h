// MLK+ graph serialization (.mlk files are JSON; Rule 24: versioned format;
// Rule 124: untrusted input — the loader validates everything).
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/math_graph.h"
#include "mlk/support/json.h"

namespace mlk {

[[nodiscard]] json::Value graphToJson(const MathGraph& graph,
                                      SymbolTable& symbols);
[[nodiscard]] Result<MathGraph> graphFromJson(const json::Value& doc,
                                              SymbolTable& symbols);
[[nodiscard]] inline Result<MathGraph> parseGraphFile(
    std::string_view text, SymbolTable& symbols) {
    MLK_TRY_VAR(doc, json::parse(text));
    return graphFromJson(doc, symbols);
}

}  // namespace mlk
