// MLK+ SmallVector (Rule 19).
//
// std::vector is banned for data that usually has 1 to 4 elements (use-def
// chains, node operands, value users, edge lists, block preds/succs).
// SmallVector<T, N> keeps N elements inline and only spills to the heap on
// overflow, avoiding allocation in the common case.
#pragma once

// GCC -Warray-bounds false positive on the inline-buffer placement-new
// pattern (GCC PR 97717 family): the compiler models the element-move loop
// as a memcpy that touches the object tail. The code below is verified by
// ASan/UBSan CI (Rule 153) and the bounds are checked by unit tests
// (tests/unit/unit_core.cpp). Suppressed ONLY in this header.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace mlk {

template <typename T, std::size_t N>
class SmallVector {
    static_assert(N > 0, "SmallVector inline capacity must be nonzero");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    SmallVector() noexcept : size_(0), capacity_(N), heap_(nullptr) {}

    explicit SmallVector(std::size_t count, const T& value = T{}) : SmallVector() {
        reserve(count);
        for (std::size_t i = 0; i < count; ++i) emplace_back(value);
    }

    SmallVector(std::initializer_list<T> init) : SmallVector() {
        reserve(init.size());
        for (const T& v : init) push_back(v);
    }

    SmallVector(const SmallVector& other) : SmallVector() {
        reserve(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i) push_back(other[i]);
    }

    SmallVector(SmallVector&& other) noexcept
        : SmallVector() {
        moveFrom(other);
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            clear();
            reserve(other.size_);
            for (std::size_t i = 0; i < other.size_; ++i) push_back(other[i]);
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            destroyAll();
            freeHeap();
            size_ = 0;
            capacity_ = N;
            heap_ = nullptr;
            moveFrom(other);
        }
        return *this;
    }

    ~SmallVector() {
        destroyAll();
        freeHeap();
    }

    [[nodiscard]] pointer data() noexcept {
        return heap_ ? heap_ : reinterpret_cast<pointer>(inline_);
    }
    [[nodiscard]] const_pointer data() const noexcept {
        return heap_ ? heap_ : reinterpret_cast<const_pointer>(inline_);
    }

    [[nodiscard]] iterator begin() noexcept { return data(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data(); }
    [[nodiscard]] iterator end() noexcept { return data() + size_; }
    [[nodiscard]] const_iterator end() const noexcept { return data() + size_; }

    [[nodiscard]] reference operator[](std::size_t i) noexcept { return data()[i]; }
    [[nodiscard]] const_reference operator[](std::size_t i) const noexcept {
        return data()[i];
    }

    [[nodiscard]] reference front() noexcept { return *data(); }
    [[nodiscard]] const_reference front() const noexcept { return *data(); }
    [[nodiscard]] reference back() noexcept { return data()[size_ - 1]; }
    [[nodiscard]] const_reference back() const noexcept { return data()[size_ - 1]; }

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] size_type capacity() const noexcept { return capacity_; }

    void reserve(std::size_t newCap) {
        if (newCap <= capacity_) return;
        grow(newCap);
    }

    void push_back(const T& v) {
        if (size_ == capacity_) grow(capacity_ * 2);
        ::new (static_cast<void*>(data() + size_)) T(v);
        ++size_;
    }

    void push_back(T&& v) {
        if (size_ == capacity_) grow(capacity_ * 2);
        ::new (static_cast<void*>(data() + size_)) T(std::move(v));
        ++size_;
    }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == capacity_) grow(capacity_ * 2);
        ::new (static_cast<void*>(data() + size_)) T(std::forward<Args>(args)...);
        ++size_;
        return back();
    }

    void pop_back() noexcept {
        --size_;
        data()[size_].~T();
    }

    void clear() noexcept {
        destroyAll();
        size_ = 0;
    }

    void resize(std::size_t count, const T& value = T{}) {
        reserve(count);
        while (size_ < count) emplace_back(value);
        while (size_ > count) pop_back();
    }

    [[nodiscard]] bool contains(const T& v) const {
        for (std::size_t i = 0; i < size_; ++i) {
            if (data()[i] == v) return true;
        }
        return false;
    }

    void eraseAt(std::size_t index) {
        for (std::size_t i = index + 1; i < size_; ++i) {
            data()[i - 1] = std::move(data()[i]);
        }
        pop_back();
    }

    [[nodiscard]] bool operator==(const SmallVector& other) const {
        if (size_ != other.size_) return false;
        for (std::size_t i = 0; i < size_; ++i) {
            if (!(data()[i] == other.data()[i])) return false;
        }
        return true;
    }

    [[nodiscard]] bool operator!=(const SmallVector& other) const {
        return !(*this == other);
    }

private:
    void grow(std::size_t minCap) {
        std::size_t newCap = capacity_ * 2;
        if (newCap < minCap) newCap = minCap;
        auto* newHeap = static_cast<T*>(::operator new(newCap * sizeof(T)));
        // Element relocation is noexcept for all IR payload types (Rule 6:
        // no exceptions anywhere in MLK+ code).
        for (std::size_t i = 0; i < size_; ++i) {
            ::new (static_cast<void*>(newHeap + i)) T(std::move(data()[i]));
        }
        destroyAll();
        freeHeap();
        heap_ = newHeap;
        capacity_ = newCap;
    }

    void moveFrom(SmallVector& other) noexcept {
        if (other.heap_ != nullptr) {
            heap_ = other.heap_;
            capacity_ = other.capacity_;
            size_ = other.size_;
            other.heap_ = nullptr;
            other.size_ = 0;
            other.capacity_ = N;
            return;
        }
        // Inline path: hoisted pointers make element ranges explicit for
        // the optimizer (and avoid re-evaluating data() per element).
        T* const dst = static_cast<T*>(static_cast<void*>(inline_));
        T* const src = static_cast<T*>(static_cast<void*>(other.inline_));
        const std::size_t n = other.size_;
        for (std::size_t i = 0; i < n; ++i) {
            ::new (static_cast<void*>(dst + i)) T(std::move(src[i]));
        }
        size_ = n;
        for (std::size_t i = 0; i < n; ++i) src[i].~T();
        other.size_ = 0;
    }

    void destroyAll() noexcept {
        for (std::size_t i = 0; i < size_; ++i) data()[i].~T();
    }

    void freeHeap() noexcept {
        if (heap_) {
            ::operator delete(heap_);
            heap_ = nullptr;
        }
    }

    std::size_t size_;
    std::size_t capacity_;
    T* heap_;
    alignas(T) unsigned char inline_[sizeof(T) * N];
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

}  // namespace mlk
