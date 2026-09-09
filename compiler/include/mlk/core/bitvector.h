// MLK+ BitVector (Rule 18).
//
// Large sparse dataflow sets use a word-backed BitVector. std::vector<bool>
// is forbidden: it is a bit-proxy container with terrible dataflow-analysis
// ergonomics and performance characteristics.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlk {

class BitVector {
public:
    BitVector() = default;
    explicit BitVector(std::size_t bits) { resize(bits); }

    void resize(std::size_t bits) {
        size_ = bits;
        words_.assign((bits + kWordBits - 1) / kWordBits, 0);
    }

    void set(std::size_t i) {
        words_[i / kWordBits] |= (1ULL << (i % kWordBits));
    }

    void reset(std::size_t i) {
        words_[i / kWordBits] &= ~(1ULL << (i % kWordBits));
    }

    void clear() { words_.assign(words_.size(), 0); }

    [[nodiscard]] bool test(std::size_t i) const {
        return (words_[i / kWordBits] >> (i % kWordBits)) & 1ULL;
    }

    [[nodiscard]] std::size_t size() const { return size_; }

    BitVector& operator|=(const BitVector& other) {
        const std::size_t n = words_.size() < other.words_.size()
                                  ? words_.size()
                                  : other.words_.size();
        for (std::size_t i = 0; i < n; ++i) words_[i] |= other.words_[i];
        return *this;
    }

    BitVector& operator&=(const BitVector& other) {
        for (std::size_t i = 0; i < words_.size(); ++i) {
            words_[i] &= i < other.words_.size() ? other.words_[i] : 0;
        }
        return *this;
    }

    [[nodiscard]] bool operator==(const BitVector& other) const {
        return size_ == other.size_ && words_ == other.words_;
    }

    /// Returns true if any bit is set after union with `other` that was not
    /// set before (classic dataflow change detection).
    [[nodiscard]] bool unionWith(const BitVector& other) {
        bool changed = false;
        const std::size_t n = words_.size() < other.words_.size()
                                  ? words_.size()
                                  : other.words_.size();
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t before = words_[i];
            words_[i] |= other.words_[i];
            if (words_[i] != before) changed = true;
        }
        return changed;
    }

    [[nodiscard]] bool any() const {
        for (uint64_t w : words_) {
            if (w != 0) return true;
        }
        return false;
    }

    [[nodiscard]] std::size_t popcount() const {
        std::size_t count = 0;
        for (uint64_t w : words_) {
            count += static_cast<std::size_t>(__builtin_popcountll(w));
        }
        return count;
    }

private:
    static constexpr std::size_t kWordBits = 64;
    std::vector<uint64_t> words_;
    std::size_t size_{0};
};

}  // namespace mlk
