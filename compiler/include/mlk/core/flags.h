// MLK+ type-safe flag wrappers (Rule 38).
//
// Any set of independent boolean properties on a hot-path data structure
// (NodeFlags, ValueFlags, EffectTags, PropertyFacts, CapabilityFlags) must be
// represented as a bitmask with type-safe Flags<E> wrappers. Raw integers are
// forbidden for flag-like state.
#pragma once

#include <cstdint>
#include <type_traits>

namespace mlk {

template <typename E>
class Flags {
    static_assert(std::is_enum_v<E>, "Flags<E> requires an enum class E");
    static_assert(sizeof(E) <= sizeof(uint64_t));

public:
    using Underlying = std::uint64_t;  // supports enums wider than 32 bits

    constexpr Flags() noexcept = default;
    constexpr Flags(E e) noexcept
        : bits_(static_cast<Underlying>(1) << static_cast<Underlying>(e)) {}

    [[nodiscard]] static constexpr Flags fromRaw(Underlying raw) noexcept {
        Flags f;
        f.bits_ = raw;
        return f;
    }

    [[nodiscard]] constexpr bool test(E e) const noexcept {
        return (bits_ >> static_cast<Underlying>(e)) & Underlying{1};
    }

    constexpr void set(E e) noexcept {
        bits_ |= Underlying{1} << static_cast<Underlying>(e);
    }

    constexpr void clear(E e) noexcept {
        bits_ &= ~(Underlying{1} << static_cast<Underlying>(e));
    }

    constexpr void set(E e, bool on) noexcept {
        on ? set(e) : clear(e);
    }

    [[nodiscard]] constexpr bool any() const noexcept {
        return bits_ != Underlying{0};
    }

    [[nodiscard]] constexpr bool none() const noexcept {
        return bits_ == Underlying{0};
    }

    [[nodiscard]] constexpr bool anyOf(Flags other) const noexcept {
        return (bits_ & other.bits_) != 0;
    }

    [[nodiscard]] constexpr bool allOf(Flags other) const noexcept {
        return (bits_ & other.bits_) == other.bits_;
    }

    constexpr Flags& operator|=(Flags other) noexcept {
        bits_ |= other.bits_;
        return *this;
    }

    constexpr Flags& operator&=(Flags other) noexcept {
        bits_ &= other.bits_;
        return *this;
    }

    [[nodiscard]] constexpr Flags operator|(Flags other) const noexcept {
        Flags r = *this;
        r |= other;
        return r;
    }

    [[nodiscard]] constexpr Flags operator&(Flags other) const noexcept {
        Flags r = *this;
        r &= other;
        return r;
    }

    [[nodiscard]] constexpr Flags operator~() const noexcept {
        Flags r;
        r.bits_ = static_cast<Underlying>(~bits_);
        return r;
    }

    [[nodiscard]] constexpr bool operator==(Flags other) const noexcept {
        return bits_ == other.bits_;
    }

    [[nodiscard]] constexpr bool operator!=(Flags other) const noexcept {
        return bits_ != other.bits_;
    }

    [[nodiscard]] constexpr Underlying raw() const noexcept { return bits_; }
    constexpr void setRaw(Underlying raw) noexcept { bits_ = raw; }

private:
    Underlying bits_{0};
};

}  // namespace mlk
