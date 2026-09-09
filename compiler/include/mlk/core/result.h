// MLK+ core result types.
//
// Rule 6:  Native C++ exceptions are forbidden on compiler/runtime hot paths.
//          All fallible operations return Result<T, Error> (std::expected).
// Rule 26: Zero-cost error propagation via std::expected monadic operations
//          or the MLK_TRY / MLK_TRYV macros, which compile to a single branch.
// Rule 68: Every function returning a Result is [[nodiscard]].
//
// Note on Rules 26 vs 69: Rule 69 bans C-style macros for logic; Rule 26
// explicitly sanctions "a custom TRY() macro that compiles down to a single
// branch". The macros below are that sanctioned mechanism and the only logic
// macros in the codebase (see docs/adr/0002-try-macro-sanction.md).
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace mlk {

/// Stable error codes across the whole system (serialization-safe).
enum class ErrorCode : uint32_t {
    Ok = 0,
    InvalidArgument,
    InvalidGraph,
    InvalidArtifact,
    UnsupportedCapability,
    Unimplemented,
    BudgetExceeded,
    VerificationFailed,
    ProofInvalid,
    AccuracyViolation,
    EffectIllegal,
    CacheInvalid,
    ParseError,
    IoError,
    Cancelled,
    ResourceExhausted,
    Internal,
};

const char* errorCodeName(ErrorCode code) noexcept;

/// A structured error. Rule 67: never opaque. Carries a human-readable
/// message plus the violated rule id when the error corresponds to a spec
/// rule (SymbolId of "rule.NNNN" or 0 when not rule-backed).
struct Error {
    ErrorCode code{ErrorCode::Internal};
    std::string message{};
    uint32_t rule{0};

    Error() = default;
    Error(ErrorCode c, std::string msg, uint32_t ruleId = 0) noexcept
        : code(c), message(std::move(msg)), rule(ruleId) {}
};

/// The universal fallible result type (Rule 6).
template <typename T>
using Result = std::expected<T, Error>;

/// The universal unit-success type for operations that produce no value.
struct Ok final {
    char padding; // avoid zero-sized warnings in some contexts
    explicit Ok() noexcept : padding(0) {}
};

inline Ok ok() noexcept { return Ok{}; }

/// Sugar: represent std::unexpected with automatic Error construction.
inline std::unexpected<Error> err(ErrorCode c, std::string msg,
                                  uint32_t ruleId = 0) noexcept {
    return std::unexpected<Error>{Error{c, std::move(msg), ruleId}};
}

// --- Rule 26 sanctioned propagation macros ---------------------------------
// MLK_TRYV(expr): propagate the error of a Result<Ok> expression.
// MLK_TRY(dst, expr): propagate the error and bind the value to dst.
#define MLK_TRYV(expr)                                                        \
    do {                                                                      \
        auto _mlk_r = (expr);                                                 \
        if (!_mlk_r.has_value()) [[unlikely]] {                             \
            return std::unexpected<Error>(std::move(_mlk_r).error());         \
        }                                                                     \
    } while (false)

#define MLK_TRY(dst, expr)                                                    \
    do {                                                                      \
        auto _mlk_r = (expr);                                                 \
        if (!_mlk_r.has_value()) [[unlikely]] {                             \
            return std::unexpected<Error>(std::move(_mlk_r).error());         \
        }                                                                     \
        dst = std::move(*_mlk_r);                                             \
    } while (false)

}  // namespace mlk
