// MLK+ Math Domain Profile (Part 0 of the Compiler Laws; math_domain_profiles.md).
//
// A versioned description of a domain's semantics, capabilities, type system,
// algebraic laws, numeric model, calculus model, approximation policy, memory
// model, and effect model. The core compiler stays domain-agnostic (Part 0):
// all domain knowledge enters through profiles, capability flags, property
// classes, and effect classes — never through if (domain == Tensor) in
// generic passes (Rule 28).
#pragma once

#include <cstdint>
#include <string>

#include "mlk/core/flags.h"
#include "mlk/core/hash.h"
#include "mlk/core/symbol_table.h"
#include "mlk/effect/effect_kind.h"
#include "mlk/property/property_lattice.h"
#include "mlk/type/domain.h"
#include "mlk/support/json.h"
#include "mlk/core/result.h"
#include "mlk/type/math_type.h"

namespace mlk {

/// Capability flags (Part 0: Domain Capability Flags). If a capability is
/// absent, the compiler must not assume the feature exists; if present, the
/// compiler must preserve its observable semantics.
enum class Capability : uint32_t {
    HasSymbolicValues = 0,
    HasNumericValues,
    HasExactArithmetic,
    HasFloatingPoint,
    HasComplexNumbers,
    HasArbitraryPrecision,
    HasIntervals,
    HasFunctionValues,
    HasOperators,
    HasTensorDomain,
    HasSparseDomain,
    HasDynamicShapes,
    HasSymbolicShapes,
    HasCalculus,
    HasDerivatives,
    HasIntegrals,
    HasLimits,
    HasGradients,
    HasSolvers,
    HasAlgebraicRewriting,
    HasApproximation,
    HasULPContracts,
    HasFastMath,
    HasErrorBounds,
    HasLayoutTransforms,
    HasKernelFusion,
    HasAutotuning,
    HasSuperoptimization,
    HasRuntimeCodeGeneration,
    HasDynamicGraphMutation,
    HasPhysicalMemoryPlacement,
    HasCPUBackend,
    HasGPUBackend,
    HasAcceleratorBackend,
    HasBenchmarkFeedback,
    HasHardwareCounters,
    HasProofCertificates,
    HasStaticCertification,
    HasTracingHooks,
    HasProfilerHooks,
    HasDebuggerHooks,
    HasFFI,
    HasGlobalRuntimeLock,
    HasFreeThreading,
    HasAsyncKernels,
    HasStreamingEvaluation,
    HasResumableEvaluation,
    kCount,
};

using CapabilityFlags = Flags<Capability>;

const char* capabilityName(Capability c) noexcept;
bool capabilityByName(const char* name, Capability& out) noexcept;

/// Numeric semantics declared by the domain (Rule 90).
struct NumericSemantics {
    bool ieee754{true};
    bool preserveNaN{true};
    bool preserveNegativeZero{true};
    bool preserveInf{true};
    /// FP reassociation is FORBIDDEN unless the contract explicitly allows it
    /// (Rule 33, Rule 90).
    bool allowReassociation{false};
    bool allowFMAContraction{true};
    bool allowApproximateReciprocal{false};
    bool integerOverflowWraps{false};
};

/// Approximation policy (Rule 34: no approximation without a contract).
struct ApproximationPolicy {
    bool allowApproximation{false};
    double defaultMaxUlps{constants::kDefaultMaxUlps};
    double defaultMaxAbsError{constants::kDefaultMaxAbsError};
    double defaultMaxRelError{constants::kDefaultMaxRelError};
};

/// A full Math Domain Profile.
struct MathDomainProfile {
    SymbolId name{kInvalidSymbolId};
    SymbolId version{kInvalidSymbolId};  // domain version string, interned
    uint32_t dialectHash{0};
    uint32_t mathIrVersion{constants::kGraphFileFormatVersion};
    uint32_t bytecodeVersion{constants::kBytecodeFormatVersion};

    CapabilityFlags capabilities{};
    NumericSemantics numeric{};
    ApproximationPolicy approximation{};

    /// Algebraic law declarations: for each property, the profile's stance.
    /// E.g. tensor domains declare MatMul non-commutative; exact integer
    /// domains may declare integer add associative = True.
    TriState lawCommutativeMul{TriState::Unknown};
    TriState lawAssociativeAdd{TriState::Unknown};  // FP: Unknown by default!
    TriState lawDistributiveMulAdd{TriState::Unknown};

    [[nodiscard]] bool has(Capability c) const { return capabilities.test(c); }

    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashU64(static_cast<uint64_t>(capabilities.raw()));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(mathIrVersion)));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(bytecodeVersion)));
        h = hashCombine(h, hashU64(numeric.ieee754));
        h = hashCombine(h, hashU64(numeric.allowReassociation));
        h = hashCombine(h, hashU64(approximation.allowApproximation));
        h = hashCombine(h, hashF64(approximation.defaultMaxUlps));
        return h;
    }
};

/// Loads a Math Domain Profile from profile JSON (schemas/
/// math_domain_profile.schema.json). Untrusted input: validated (Rule 124).
[[nodiscard]] Result<MathDomainProfile> domainProfileFromJson(
    const json::Value& doc, SymbolTable& symbols);
[[nodiscard]] json::Value domainProfileToJson(const MathDomainProfile& p,
                                              SymbolTable& symbols);

}  // namespace mlk
