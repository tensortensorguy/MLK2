// MLK+ realization cache (Rule 57: complete cache keys; Rule 37: versioned;
// Rule 124: untrusted artifacts validated on load).
#pragma once

#include "mlk/core/constants.h"
#include "mlk/core/hash.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/proof/accuracy_contract.h"
#include "mlk/support/json.h"
#include "mlk/support/kernel_ir.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

/// Complete realization cache key (Rule 57 — incomplete keys are correctness
/// bugs).
struct CacheKey {
    HashValue graphHash{0};
    HashValue propertyFactsHash{0};
    int64_t shapeBucket{0};
    SymbolId dtype{constants::kInvalidId};
    SymbolId domain{constants::kInvalidId};
    HashValue layoutConstraints{0};
    HashValue accuracyHash{0};
    uint32_t profileVersion{0};
    uint32_t compilerVersion{constants::kRealizationCacheFormatVersion};
    HashValue passPipelineHash{0};
    HashValue superoptVersion{0};
    HashValue hardwareFingerprint{0};
    HashValue runtimeConfig{0};

    [[nodiscard]] HashValue totalHash() const noexcept {
        HashValue h = graphHash;
        h = hashCombine(h, propertyFactsHash);
        h = hashCombine(h, hashI64(shapeBucket));
        h = hashCombine(h, accuracyHash);
        h = hashCombine(h, hashU64(profileVersion));
        h = hashCombine(h, hashU64(compilerVersion));
        h = hashCombine(h, passPipelineHash);
        h = hashCombine(h, superoptVersion);
        h = hashCombine(h, hardwareFingerprint);
        return h;
    }

    [[nodiscard]] json::Value toJson() const;
    [[nodiscard]] static Result<CacheKey> fromJson(const json::Value& doc);
};

/// Realization cache entry: strategy, schedule, proof reference, artifact
/// hash, measured result (spec §8/§17: cache realizations, not configs).
struct CacheEntry {
    CacheKey key{};
    SymbolId strategy{kInvalidSymbolId};
    HashValue kernelHash{0};
    json::Value schedule{json::Object{}};
    SymbolId proofRef{kInvalidSymbolId};
    double measuredMs{0.0};
    uint32_t formatVersion{constants::kRealizationCacheFormatVersion};

    [[nodiscard]] json::Value toJson(SymbolTable& symbols) const;
    [[nodiscard]] static Result<CacheEntry> fromJson(const json::Value& doc,
                                                     SymbolTable& symbols);
};

/// In-memory realization cache with optional file persistence. Load
/// validates every entry (Rule 124) and rejects version mismatches
/// (Rule 37/127) with telemetry, never silently.
class RealizationCache {
public:
    explicit RealizationCache(SymbolTable& symbols) : symbols_(symbols) {}

    [[nodiscard]] const CacheEntry* find(const CacheKey& key) const;
    void store(const CacheEntry& entry);

    [[nodiscard]] std::size_t size() const;
    void clear();

    /// Persists all entries to JSON (realization_cache.md schema).
    [[nodiscard]] json::Value toJson() const;

private:
    SymbolTable& symbols_;
    OpenHashMap<HashValue, CacheEntry> entries_{};
};

}  // namespace mlk
