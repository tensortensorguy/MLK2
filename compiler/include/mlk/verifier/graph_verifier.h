// MLK+ graph verifier (Rule 47: runs in debug builds after every pass;
// Rule 145: checks fallback, proof, and memory metadata too).
#pragma once

#include "mlk/core/diagnostics.h"
#include "mlk/core/result.h"
#include "mlk/ir/math_graph.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

struct VerifyOptions {
    bool checkTypes{true};          // derived types re-validated
    bool checkEffects{true};        // effect chain continuity (Rule 145)
    bool checkGuards{true};         // GraphState on every guard (Rule 5/145)
    bool checkFacts{true};          // property lattice consistency
    bool checkAcyclic{true};        // no cycles where illegal
};

/// Full verification. Diagnostics are always actionable (Rule 67):
/// location, expected vs actual, violated rule, suggested fix.
[[nodiscard]] bool verifyGraph(const MathGraph& graph,
                               const MathDomainProfile* profile,
                               VerifyOptions opts, DiagnosticEngine& diag);

}  // namespace mlk
