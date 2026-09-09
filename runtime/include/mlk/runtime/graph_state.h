// MLK+ GraphState (Rules 5, 100, 101): the snapshot that lets the fallback
// engine reconstruct the exact lower-tier execution state. The verifier
// rejects incomplete GraphState.
#pragma once

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/core/result.h"
#include "mlk/core/small_vector.h"
#include "mlk/ir/value_id.h"
#include "mlk/support/json.h"

namespace mlk {

/// One reconstructed value binding at fallback time.
struct ValueBinding {
    ValueId value{kInvalidValueId};
    double f64{0.0};
    int64_t i64{0};
    bool isInt{false};
};

/// GraphState: complete, machine-checkable fallback snapshot (Rule 101).
struct GraphState {
    /// Interpreter/graph position to resume from.
    NodeId resumeNode{constants::kInvalidId};
    uint64_t graphVersion{0};
    uint64_t domainVersion{0};
    SmallVector<ValueBinding, 8> bindings{};
    /// Dependency list for invalidation (Rule 89/120).
    SmallVector<SymbolId, 4> dependencies{};

    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashU64(resumeNode);
        h = hashCombine(h, hashU64(graphVersion));
        for (std::size_t i = 0; i < bindings.size(); ++i) {
            h = hashCombine(h, hashU64(bindings[i].value));
            h = hashCombine(h, bindings[i].isInt ? hashI64(bindings[i].i64)
                                                 : hashF64(bindings[i].f64));
        }
        return h;
    }

    [[nodiscard]] json::Value toJson() const;
    [[nodiscard]] static Result<GraphState> fromJson(const json::Value& doc);
};

/// Guard metadata (Rule 65: complete metadata; no anonymous guards).
struct GuardMetadata {
    SymbolId guardedAssumption{kInvalidSymbolId};
    SymbolId evidenceSource{kInvalidSymbolId};  // profile | benchmark | proof
    double confidence{0.0};
    SymbolId guardKind{kInvalidSymbolId};
    SymbolId fallbackTarget{kInvalidSymbolId};
    uint32_t graphStateId{constants::kInvalidId};
    SymbolId invalidationDependency{kInvalidSymbolId};
    double costEstimate{0.0};
};

}  // namespace mlk
