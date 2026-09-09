// mlk_core: result/diagnostics/json implementations (cold-path code lives in
// the .cc to keep headers lean; hot-path helpers are header-only).
#include "mlk/core/diagnostics.h"
#include "mlk/core/result.h"

namespace mlk {

const char* errorCodeName(ErrorCode code) noexcept {
    switch (code) {  // exhaustive on closed enum (Rule 78)
        case ErrorCode::Ok: return "Ok";
        case ErrorCode::InvalidArgument: return "InvalidArgument";
        case ErrorCode::InvalidGraph: return "InvalidGraph";
        case ErrorCode::InvalidArtifact: return "InvalidArtifact";
        case ErrorCode::UnsupportedCapability: return "UnsupportedCapability";
        case ErrorCode::Unimplemented: return "Unimplemented";
        case ErrorCode::BudgetExceeded: return "BudgetExceeded";
        case ErrorCode::VerificationFailed: return "VerificationFailed";
        case ErrorCode::ProofInvalid: return "ProofInvalid";
        case ErrorCode::AccuracyViolation: return "AccuracyViolation";
        case ErrorCode::EffectIllegal: return "EffectIllegal";
        case ErrorCode::CacheInvalid: return "CacheInvalid";
        case ErrorCode::ParseError: return "ParseError";
        case ErrorCode::IoError: return "IoError";
        case ErrorCode::Cancelled: return "Cancelled";
        case ErrorCode::ResourceExhausted: return "ResourceExhausted";
        case ErrorCode::Internal: return "Internal";
    }
    return "Unknown";
}

const char* severityName(Severity s) noexcept {
    switch (s) {  // exhaustive on closed enum (Rule 78)
        case Severity::Note: return "note";
        case Severity::Warning: return "warning";
        case Severity::Error: return "error";
        case Severity::Fatal: return "fatal";
    }
    return "unknown";
}

Diagnostic diagnosticFromError(const Error& e, SymbolId pass) {
    Diagnostic d;
    d.severity = Severity::Error;
    d.pass = pass;
    d.message = e.message;
    if (e.rule != 0) {
        d.rule = "Rule " + std::to_string(e.rule);
    }
    return d;
}

}  // namespace mlk
