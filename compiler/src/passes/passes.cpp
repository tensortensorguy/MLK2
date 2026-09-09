// Pass registration core.
#include "passes_common.h"

#include <string>

namespace mlk::passes {

void registerPass(SymbolTable& symbols, Pass& pass, PassKind kind,
                  std::initializer_list<const char*> required,
                  std::initializer_list<const char*> produced,
                  std::initializer_list<const char*> invalidated,
                  std::initializer_list<Tier> tiers) {
    PassRegistrar reg(symbols, pass.nameText(), kind,
                      required, produced, invalidated, tiers, &pass);
}

void registerAllPasses(SymbolTable& symbols) {
    registerAnalysisPasses(symbols);
    registerMathPasses(symbols);
    registerEgraphPasses(symbols);
    registerCalculusPasses(symbols);
    registerTensorPasses(symbols);
    registerApproxPasses(symbols);
    registerSchedulePasses(symbols);
    registerPhysicalPasses(symbols);
    registerLowerPasses(symbols);
    registerBackendPasses(symbols);
}

}  // namespace mlk::passes
