// MLK+ diagnostics (Rule 67: actionable compiler diagnostics).
//
// The compiler must never output opaque errors. Every Diagnostic includes:
// the exact source/graph location, a clear human-readable message, the
// expected vs actual state, the violated rule or contract, and a suggested
// fix.
//
// Diagnostics are cold-path artifacts: std::string appears here by design and
// nowhere else in the IR (Rule 16 keeps identifiers interned; Rule 9 bans
// strings only from hot IR data structures).
#pragma once

#include <cstdint>
#include <string>

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"

namespace mlk {

enum class Severity : uint8_t {
    Note,
    Warning,
    Error,
    Fatal,
};

const char* severityName(Severity s) noexcept;

/// Fully actionable diagnostic record.
struct Diagnostic {
    Severity severity{Severity::Error};
    /// Graph coordinates.
    uint32_t nodeId{constants::kInvalidId};
    uint32_t valueId{constants::kInvalidId};
    /// Which pass/component produced this diagnostic.
    SymbolId pass{kInvalidSymbolId};
    std::string message{};
    std::string expected{};
    std::string actual{};
    /// Spec rule identifier as text, e.g. "Rule 33" (cold-path convenience).
    std::string rule{};
    std::string suggestedFix{};

    [[nodiscard]] Error toError(ErrorCode code = ErrorCode::VerificationFailed)
        const {
        std::string full = message;
        if (!expected.empty() || !actual.empty()) {
            full += " (expected: " + expected + ", actual: " + actual + ")";
        }
        if (!rule.empty()) {
            full += " [" + rule + "]";
        }
        if (!suggestedFix.empty()) {
            full += " Suggested fix: " + suggestedFix;
        }
        return Error{code, std::move(full)};
    }
};

/// Diagnostic collection + reporting sink. Passed through PassContext;
/// never globally held (Rule 144: no hidden global mutable state).
class DiagnosticEngine {
public:
    void report(Diagnostic diag) {
        if (sink_) sink_(diag, sinkUser_);
        entries_.push_back(std::move(diag));
    }

    void reportError(uint32_t nodeId, std::string message,
                     std::string expected = {}, std::string actual = {},
                     std::string rule = {}, std::string fix = {}) {
        Diagnostic d;
        d.severity = Severity::Error;
        d.nodeId = nodeId;
        d.message = std::move(message);
        d.expected = std::move(expected);
        d.actual = std::move(actual);
        d.rule = std::move(rule);
        d.suggestedFix = std::move(fix);
        report(std::move(d));
    }

    void setSink(void (*fn)(const Diagnostic&, void*), void* user) {
        sink_ = fn;
        sinkUser_ = user;
    }

    [[nodiscard]] bool hasErrors() const {
        for (const auto& d : entries_) {
            if (d.severity == Severity::Error ||
                d.severity == Severity::Fatal) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] const std::vector<Diagnostic>& entries() const {
        return entries_;
    }

    void clear() { entries_.clear(); }

private:
    std::vector<Diagnostic> entries_;
    void (*sink_)(const Diagnostic&, void*){nullptr};
    void* sinkUser_{nullptr};
};

/// Convert a failed Result into a Diagnostic (preserving rule id).
[[nodiscard]] Diagnostic diagnosticFromError(const Error& e,
                                             SymbolId pass = kInvalidSymbolId);

}  // namespace mlk
