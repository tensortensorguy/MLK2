// MLK+ effect kinds (Rule 140: the IR must have an explicit effect model).
//
// Effect classes per spec Part XI; the set is carried per node as a typed
// bitmask (Rule 38). Passes must not reorder effects without proof.
#pragma once

#include "mlk/core/flags.h"

namespace mlk {

enum class EffectKind : uint8_t {
    Pure = 0,
    MemoryRead,
    MemoryWrite,
    DomainStateMutation,
    RandomState,
    SolverState,
    IO,
    FFI,
    Allocation,
    Tracing,
    ApproximationSwitch,
    Exception,
    Guard,
    ReferenceBarrier,
    PhysicalPlacement,
    kCount,
};

using EffectSet = Flags<EffectKind>;

const char* effectKindName(EffectKind k) noexcept;

/// Canonical effect set of an op class (effect_inference consumes this).
[[nodiscard]] constexpr EffectSet pureEffects() noexcept { return EffectSet{}; }

}  // namespace mlk
