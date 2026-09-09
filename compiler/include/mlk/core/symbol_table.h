// MLK+ interned symbol table (Rule 16).
//
// Never pass, compare, or store std::string or std::string_view in the IR or
// passes. All identifiers — operator names, domain names, type names,
// property names, layout names, module names, proof-rule names — are
// interned here at the frontend. The IR only uses SymbolId (uint32_t).
//
// Interning is cold-path (frontend) and internally synchronized; lookups by
// id are lock-free vector indexing (Rule 144: allowed global state is an
// interned symbol table with proper synchronization).
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlk/core/constants.h"

namespace mlk {

using SymbolId = uint32_t;

inline constexpr SymbolId kInvalidSymbolId = 0xFFFFFFFFu;

/// A per-process intern pool. Frontends own one SymbolTable and thread it
/// through contexts; the IR stores only ids.
class SymbolTable {
public:
    SymbolTable() { pool_.reserve(constants::kSymbolInitialBuckets); }

    [[nodiscard]] SymbolId intern(std::string_view name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = lookup_.find(name);
        if (it != lookup_.end()) return it->second;
        const auto id = static_cast<SymbolId>(pool_.size());
        pool_.emplace_back(name);
        lookup_.emplace(pool_.back(), id);
        return id;
    }

    /// View of an interned symbol. Cold path: diagnostics, printing, tools.
    /// The IR itself must not store the returned pointer.
    [[nodiscard]] std::string_view view(SymbolId id) const {
        return pool_[id];
    }

    /// Materialized copy for tool output only (never stored in the IR).
    [[nodiscard]] std::string text(SymbolId id) const {
        return std::string(pool_[id]);
    }

    [[nodiscard]] std::size_t size() const { return pool_.size(); }

private:
    std::mutex mutex_;
    std::vector<std::string> pool_;
    std::unordered_map<std::string_view, SymbolId> lookup_;
};

}  // namespace mlk
