// MLK+ shared pass implementation helpers (internal to compiler/src/passes).
#pragma once

#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/pass.h"
#include "mlk/pass/pass_registry.h"

namespace mlk::passes {

/// Base class providing name/kind plumbing + telemetry + budget discipline
/// (Rule 10: passes must be idempotent/monotonic; budget checked by caller).
class PassBase : public Pass {
public:
    PassBase(SymbolTable& symbols, const char* name, PassKind kind)
        : symbols_(symbols), kind_(kind), nameStr_(name) {}

    [[nodiscard]] const char* nameText() const override {
        return nameStr_.c_str();
    }
    [[nodiscard]] PassKind kind() const override { return kind_; }
    /// Table-scoped id (resolved through the CURRENT context table).
    [[nodiscard]] SymbolId nameId(SymbolTable& symbols) const {
        return symbols.intern(nameStr_);
    }

protected:
    SymbolTable& symbols_;
    PassKind kind_;
    std::string nameStr_;
};

/// Registers a pass with its contract (Rule 142).
void registerPass(SymbolTable& symbols, Pass& pass, PassKind kind,
                  std::initializer_list<const char*> required,
                  std::initializer_list<const char*> produced,
                  std::initializer_list<const char*> invalidated,
                  std::initializer_list<Tier> tiers);

/// Registers every pass in the compiler (deterministic order).
void registerAllPasses(SymbolTable& symbols);

// --- Category registrators (defined in their category .cpp files) ----------
void registerAnalysisPasses(SymbolTable& symbols);
void registerMathPasses(SymbolTable& symbols);
void registerEgraphPasses(SymbolTable& symbols);
void registerCalculusPasses(SymbolTable& symbols);
void registerTensorPasses(SymbolTable& symbols);
void registerApproxPasses(SymbolTable& symbols);
void registerSchedulePasses(SymbolTable& symbols);
void registerPhysicalPasses(SymbolTable& symbols);
void registerLowerPasses(SymbolTable& symbols);
void registerBackendPasses(SymbolTable& symbols);

}  // namespace mlk::passes
