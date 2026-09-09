// MLK+ SparseSet (Rule 18).
//
// Passes tracking sets of NodeIds or ValueIds (liveness, dominance,
// equivalence classes, visited sets, dependency sets) must use SparseSets for
// small dense sets or BitVectors for large sparse sets. std::set,
// std::unordered_set, and std::vector<bool> are forbidden for dataflow data.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlk {

class SparseSet {
public:
    explicit SparseSet(uint32_t universe = 0) { reset(universe); }

    void reset(uint32_t universe) {
        universe_ = universe;
        dense_.clear();
        sparse_.assign(universe, kNotFound);
    }

    void insert(uint32_t v) {
        if (contains(v)) return;
        sparse_[v] = static_cast<uint32_t>(dense_.size());
        dense_.push_back(v);
    }

    void remove(uint32_t v) {
        if (!contains(v)) return;
        uint32_t idx = sparse_[v];
        uint32_t last = dense_.back();
        dense_[idx] = last;
        sparse_[last] = idx;
        dense_.pop_back();
        sparse_[v] = kNotFound;
    }

    [[nodiscard]] bool contains(uint32_t v) const {
        return v < universe_ && sparse_[v] != kNotFound &&
               dense_[sparse_[v]] == v;
    }

    [[nodiscard]] std::size_t size() const { return dense_.size(); }
    [[nodiscard]] bool empty() const { return dense_.empty(); }
    [[nodiscard]] uint32_t operator[](std::size_t i) const { return dense_[i]; }

    [[nodiscard]] const std::vector<uint32_t>& dense() const { return dense_; }

    void clear() {
        for (uint32_t v : dense_) sparse_[v] = kNotFound;
        dense_.clear();
    }

private:
    static constexpr uint32_t kNotFound = 0xFFFFFFFFu;
    std::vector<uint32_t> dense_;
    std::vector<uint32_t> sparse_;
    uint32_t universe_{0};
};

}  // namespace mlk
