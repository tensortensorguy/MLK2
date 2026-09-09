// MLK+ declarative search space (Rule 54: parameter names, domains,
// constraints, priors; serializable; no ad-hoc mutation in benchmark loops).
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/support/json.h"

namespace mlk {

struct SearchParam {
    SymbolId name{kInvalidSymbolId};
    SmallVector<int64_t, 8> domain{};
    int64_t prior{0};
};

struct SearchSpace {
    SmallVector<SearchParam, 8> params{};

    [[nodiscard]] json::Value toJson(SymbolTable& symbols) const;
    [[nodiscard]] static Result<SearchSpace> fromJson(const json::Value& doc,
                                                      SymbolTable& symbols);
    /// Materializes a configuration vector into a schedule map.
    [[nodiscard]] OpenHashMap<SymbolId, int64_t> realize(
        const SmallVector<int64_t, 8>& config) const;
};

}  // namespace mlk
