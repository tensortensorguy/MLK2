// MLK+ open-addressing hash map / hash set (Rule 17).
//
// std::unordered_map and std::map are forbidden in the compiler hot path.
// GVN, hash-consing, e-graph memoization, realization cache lookup, and any
// pass requiring a hash table must use this cache-friendly open-addressing
// map with linear probing and power-of-two capacity.
#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

#include "mlk/core/hash.h"
#include "mlk/core/hash_functors.h"

namespace mlk {

namespace detail {
inline constexpr uint32_t kEmptySlot = 0xFFFFFFFFu;
inline constexpr uint32_t kTombstone = 0xFFFFFFFEu;
inline constexpr double kMaxLoadFactor = 0.7;
}  // namespace detail

/// Open-addressing hash map from Key to Value.
/// Requires: Key is trivially comparable (==) and hashable via Hash.
template <typename Key, typename Value, typename Hash = mlk::FnvHash<Key>>
class OpenHashMap {
public:
    OpenHashMap() { rehash(16); }

    explicit OpenHashMap(std::size_t initialBuckets) {
        std::size_t cap = 16;
        while (cap < initialBuckets) cap *= 2;
        rehash(cap);
    }

    /// Inserts or finds. Returns pointer to the stored value and whether a
    /// new entry was created.
    [[nodiscard]] Value* findOrInsert(const Key& key, bool* inserted,
                                      const Value& defaultValue = Value{}) {
        if ((count_ + tombstones_) * 10 >=
            static_cast<std::size_t>(static_cast<double>(slots_.size()) *
                                     detail::kMaxLoadFactor * 10)) {
            rehash(slots_.size() * 2);
        }
        const std::size_t mask = slots_.size() - 1;
        std::size_t idx = static_cast<std::size_t>(Hash{}(key)) & mask;
        std::size_t firstTombstone = kNoIndex;
        for (;;) {
            const uint32_t s = slots_[idx];
            if (s == detail::kEmptySlot) {
                std::size_t target =
                    firstTombstone != kNoIndex ? firstTombstone : idx;
                slots_[target] = static_cast<uint32_t>(entries_.size());
                entries_.push_back({key, defaultValue});
                if (firstTombstone != kNoIndex) --tombstones_;
                ++count_;
                if (inserted) *inserted = true;
                return &entries_.back().value;
            }
            if (s == detail::kTombstone) {
                if (firstTombstone == kNoIndex) firstTombstone = idx;
            } else if (entries_[s].key == key) {
                if (inserted) *inserted = false;
                return &entries_[s].value;
            }
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] Value* find(const Key& key) {
        const std::size_t mask = slots_.size() - 1;
        std::size_t idx = static_cast<std::size_t>(Hash{}(key)) & mask;
        for (;;) {
            const uint32_t s = slots_[idx];
            if (s == detail::kEmptySlot) return nullptr;
            if (s != detail::kTombstone && entries_[s].key == key) {
                return &entries_[s].value;
            }
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] const Value* find(const Key& key) const {
        return const_cast<OpenHashMap*>(this)->find(key);
    }

    void remove(const Key& key) {
        const std::size_t mask = slots_.size() - 1;
        std::size_t idx = static_cast<std::size_t>(Hash{}(key)) & mask;
        for (;;) {
            const uint32_t s = slots_[idx];
            if (s == detail::kEmptySlot) return;
            if (s != detail::kTombstone && entries_[s].key == key) {
                slots_[idx] = detail::kTombstone;
                --count_;
                ++tombstones_;
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] std::size_t size() const { return count_; }

    void clear() {
        entries_.clear();
        slots_.assign(slots_.size(), detail::kEmptySlot);
        count_ = 0;
        tombstones_ = 0;
    }

    template <typename Fn>
    void forEach(Fn&& fn) {
        for (auto& e : entries_) fn(e.key, e.value);
    }

    template <typename Fn>
    void forEach(Fn&& fn) const {
        for (const auto& e : entries_) fn(e.key, e.value);
    }

private:
    struct Slot {
        Key key;
        Value value;
    };

    static constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

    void rehash(std::size_t newCap) {
        std::vector<uint32_t> newSlots(newCap, detail::kEmptySlot);
        const std::size_t mask = newCap - 1;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            std::size_t idx =
                static_cast<std::size_t>(Hash{}(entries_[i].key)) & mask;
            while (newSlots[idx] != detail::kEmptySlot) idx = (idx + 1) & mask;
            newSlots[idx] = static_cast<uint32_t>(i);
        }
        slots_ = std::move(newSlots);
        tombstones_ = 0;
    }

    std::vector<uint32_t> slots_;
    std::vector<Slot> entries_;
    std::size_t count_{0};
    std::size_t tombstones_{0};
};

/// Open-addressing hash set.
template <typename Key, typename Hash = mlk::FnvHash<Key>>
class OpenHashSet {
public:
    OpenHashSet() { rehash(16); }

    /// Returns true if newly inserted.
    [[nodiscard]] bool insert(const Key& key) {
        if ((count_ + tombstones_) * 10 >=
            static_cast<std::size_t>(static_cast<double>(slots_.size()) *
                                     detail::kMaxLoadFactor * 10)) {
            rehash(slots_.size() * 2);
        }
        const std::size_t mask = slots_.size() - 1;
        std::size_t idx = static_cast<std::size_t>(Hash{}(key)) & mask;
        for (;;) {
            const uint32_t s = slots_[idx];
            if (s == detail::kEmptySlot) {
                slots_[idx] = kOccupiedSentinel++;
                ++count_;
                keys_.push_back(key);
                return true;
            }
            if (s != detail::kTombstone && keys_[s] == key) return false;
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] bool contains(const Key& key) const {
        const std::size_t mask = slots_.size() - 1;
        std::size_t idx = static_cast<std::size_t>(Hash{}(key)) & mask;
        for (;;) {
            const uint32_t s = slots_[idx];
            if (s == detail::kEmptySlot) return false;
            if (s != detail::kTombstone && keys_[s] == key) return true;
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] std::size_t size() const { return count_; }
    [[nodiscard]] const std::vector<Key>& keys() const { return keys_; }

private:
    void rehash(std::size_t newCap) {
        std::vector<uint32_t> newSlots(newCap, detail::kEmptySlot);
        const std::size_t mask = newCap - 1;
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            std::size_t idx = static_cast<std::size_t>(Hash{}(keys_[i])) & mask;
            while (newSlots[idx] != detail::kEmptySlot) idx = (idx + 1) & mask;
            newSlots[idx] = static_cast<uint32_t>(i);
        }
        slots_ = std::move(newSlots);
        tombstones_ = 0;
    }

    std::vector<uint32_t> slots_;
    std::vector<Key> keys_;
    std::size_t count_{0};
    std::size_t tombstones_{0};
    uint32_t kOccupiedSentinel{0};
};

}  // namespace mlk
