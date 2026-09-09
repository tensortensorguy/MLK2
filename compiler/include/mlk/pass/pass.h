// MLK+ pass framework (pass/pass.h, spec §6).
#pragma once

#include "mlk/core/diagnostics.h"
#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/effect/effect_kind.h"
#include "mlk/ir/math_graph.h"
#include "mlk/proof/accuracy_contract.h"
#include "mlk/support/kernel_ir.h"
#include "mlk/type/domain_profile.h"

namespace mlk {

enum class PassKind : uint8_t {
    Analysis,
    Transform,
    Lowering,
    Verify,
    Tune,
    Superopt,
};

const char* passKindName(PassKind k) noexcept;

enum class Tier : uint8_t {
    Tier0 = 0,  // reference interpreter / oracle
    Tier1,      // baseline kernel compiler
    Tier2,      // optimizing JIT / autotuner
    Tier3,      // AOT / supercompiled (compile=INF)
};

const char* tierName(Tier t) noexcept;

/// Rule 10/131: strict, explicit budgets; violations trigger fallback, not
/// evaluation stalls.
struct PassBudget {
    int64_t nodeEdits{constants::kDefaultPassEditBudget};
    uint32_t fixpointIterations{constants::kMaxPassFixpointIterations};
    double seconds{60.0};

    void consume(int64_t n = 1) { nodeEdits -= n; }
    [[nodiscard]] bool exhausted() const { return nodeEdits <= 0; }
};

class CostModel;
class CancellationToken;  // see mlk/core/cancellation.h
class IEventSink;         // see mlk/core/event_sink.h (telemetry face)

/// PassContext (spec §6): everything a pass may see. No hidden globals
/// (Rule 144).
struct PassContext {
    const MathDomainProfile* domainProfile{nullptr};
    const AccuracyContract* accuracy{nullptr};
    CostModel* costModel{nullptr};
    DiagnosticEngine* diag{nullptr};
    IEventSink* telemetry{nullptr};
    CancellationToken* cancel{nullptr};
    SymbolTable* symbols{nullptr};
    PassBudget budget{};
    Tier tier{Tier::Tier1};
    /// Kill switches by pass name (Rule 60/150: every optimization has one).
    const OpenHashMap<SymbolId, bool>* killSwitches{nullptr};
    /// Pass configuration knobs (bucket edges, tile choices, etc.).
    const OpenHashMap<SymbolId, int64_t>* knobs{nullptr};
    /// Lowering output sink: lower.to_kernel_ir / backend.emit_binary write
    /// their artifacts here (spec §12: kernel IR between math and backend).
    KernelModule* kernelOut{nullptr};

    [[nodiscard]] bool killed(SymbolId passName) const {
        if (killSwitches == nullptr) return false;
        const bool* k = killSwitches->find(passName);
        return k != nullptr && *k;
    }
};

/// PassResult (spec §6 + Rule 30): what changed, what got invalidated, and
/// structured telemetry events emitted during the run.
struct PassResult {
    bool changed{false};
    uint32_t nodesBefore{0};
    uint32_t nodesAfter{0};
    SmallVector<SymbolId, 4> invalidatedAnalyses{};
};

/// The pass interface (spec §6). Passes must be idempotent and monotonic
/// (Rule 10) and declare their contract (Rule 142).
class Pass {
public:
    virtual ~Pass() = default;

    /// Stable textual identity (interned per-SymbolTable by the registry;
    /// Rule 16: ids are table-scoped, so passes carry text, not ids).
    [[nodiscard]] virtual const char* nameText() const = 0;
    [[nodiscard]] virtual PassKind kind() const = 0;

    [[nodiscard]] virtual Result<PassResult> run(PassContext& ctx,
                                                 MathGraph& graph) = 0;
};

}  // namespace mlk
