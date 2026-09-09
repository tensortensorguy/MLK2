// MLK+ property lattice (Rule 22: properties are first-class and tri-state).
//
// Unknown properties must remain Unknown, never default to True. The lattice
// is the classic three-point lattice: Unknown is top (no information),
// True/False are definite facts, Bottom marks contradictory inference (a
// verifier error).
#pragma once

#include <cstdint>

namespace mlk {

enum class TriState : uint8_t {
    Unknown = 0,
    False = 1,
    True = 2,
};

/// Meet: keep the weaker (less certain) of two facts. True meet False =
/// Bottom is represented by False + verifier diagnostics.
[[nodiscard]] constexpr TriState triStateMeet(TriState a, TriState b) noexcept {
    if (a == TriState::Unknown) return b;
    if (b == TriState::Unknown) return a;
    if (a == b) return a;
    return TriState::False;  // conflict resolved conservatively
}

/// First-class mathematical properties tracked per value/node (Rule 22 list;
/// property_system.md). ids are stable for serialization.
enum class PropertyId : uint16_t {
    Commutative = 0,
    Associative,
    Distributive,
    Idempotent,
    HasIdentity,
    MonotonicIncreasing,
    MonotonicDecreasing,
    Periodic,
    Differentiable,
    Invertible,
    Positive,
    NonNegative,
    Bounded,
    Sparse,
    Symmetric,
    Contiguous,
    Pure,
    IntegerValued,
    kCount,
};

const char* propertyName(PropertyId p) noexcept;
bool propertyByName(const char* name, PropertyId& out) noexcept;

}  // namespace mlk
