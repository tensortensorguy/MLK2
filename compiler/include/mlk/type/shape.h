// MLK+ shapes (realization spec §3: Shape with -1 for dynamic/symbolic).
#pragma once

#include <cstdint>
#include <optional>

#include "mlk/core/constants.h"
#include "mlk/core/small_vector.h"

namespace mlk {

/// Sentinel for a dynamic/symbolic dimension (realization spec §3).
inline constexpr int64_t kDynamicDim = -1;

class Shape {
public:
    Shape() = default;

    static Shape scalar() { return Shape{}; }

    static Shape fromList(std::initializer_list<int64_t> dims) {
        Shape s;
        for (int64_t d : dims) s.dims_.push_back(d);
        return s;
    }

    explicit Shape(std::initializer_list<int64_t> dims) : Shape() {
        for (int64_t d : dims) dims_.push_back(d);
    }

    void addDim(int64_t d) { dims_.push_back(d); }

    [[nodiscard]] std::size_t rank() const { return dims_.size(); }
    [[nodiscard]] bool isScalar() const { return dims_.empty(); }
    [[nodiscard]] bool hasDynamic() const {
        for (int64_t d : dims_) {
            if (d == kDynamicDim) return true;
        }
        return false;
    }
    [[nodiscard]] bool isStatic() const { return !hasDynamic(); }

    /// Number of elements if fully static; nullopt when any dim is dynamic.
    [[nodiscard]] std::optional<int64_t> numel() const {
        int64_t n = 1;
        for (int64_t d : dims_) {
            if (d == kDynamicDim) return std::nullopt;
            n *= d;
        }
        return n;
    }

    [[nodiscard]] int64_t dim(std::size_t i) const { return dims_[i]; }
    [[nodiscard]] int64_t& dim(std::size_t i) { return dims_[i]; }
    [[nodiscard]] const SmallVector<int64_t, constants::kShapeInline>& dims()
        const {
        return dims_;
    }

    void setDims(SmallVector<int64_t, constants::kShapeInline> d) {
        dims_ = std::move(d);
    }

    /// Broadcast-compatible merge (NumPy rules); fails on rank/type mismatch.
    [[nodiscard]] static bool broadcast(const Shape& a, const Shape& b,
                                        Shape& out);

    [[nodiscard]] bool operator==(const Shape& other) const {
        if (dims_.size() != other.dims_.size()) return false;
        for (std::size_t i = 0; i < dims_.size(); ++i) {
            if (dims_[i] != other.dims_[i]) return false;
        }
        return true;
    }

    [[nodiscard]] bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

private:
    SmallVector<int64_t, constants::kShapeInline> dims_{};
};

}  // namespace mlk
