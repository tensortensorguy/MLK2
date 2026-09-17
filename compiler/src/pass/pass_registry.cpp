// Pass registry implementation.
#include "mlk/pass/pass_registry.h"

#include <algorithm>
#include <cstring>

namespace mlk {

const char* passKindName(PassKind k) noexcept {
    switch (k) {  // Rule 78: exhaustive
        case PassKind::Analysis: return "analysis";
        case PassKind::Transform: return "transform";
        case PassKind::Lowering: return "lowering";
        case PassKind::Verify: return "verify";
        case PassKind::Tune: return "tune";
        case PassKind::Superopt: return "superopt";
    }
    return "?";
}

const char* tierName(Tier t) noexcept {
    switch (t) {
        case Tier::Tier0: return "tier0";
        case Tier::Tier1: return "tier1";
        case Tier::Tier2: return "tier2";
        case Tier::Tier3: return "tier3";
    }
    return "?";
}

PassRegistry& PassRegistry::instance() {
    static PassRegistry registry;
    return registry;
}

void PassRegistry::add(PassContract contract, Pass* pass) {
    Entry e;
    e.contract = std::move(contract);
    e.pass = pass;
    // Textual identity is captured at registration (Rule 16: passes
    // carry text). The contract's SymbolId is table-scoped and is NOT
    // used for resolution — only for deterministic iteration order.
    e.nameText = pass != nullptr ? pass->nameText() : nullptr;
    entries_.push_back(std::move(e));
    // Deterministic ordering by name id (interning is deterministic per
    // table; Rule 143).
    std::sort(entries_.begin(), entries_.end(),
              [](const Entry& a, const Entry& b) {
                  return a.contract.name < b.contract.name;
              });
}

Pass* PassRegistry::byName(const SymbolTable& symbols, SymbolId name) {
    if (name == kInvalidSymbolId) return nullptr;
    const std::string text = symbols.text(name);
    for (auto& e : entries_) {
        if (e.nameText != nullptr && std::strcmp(e.nameText, text.c_str()) == 0) {
            return e.pass;
        }
    }
    return nullptr;
}

const PassContract* PassRegistry::contractByName(
    const SymbolTable& symbols, SymbolId name) const {
    if (name == kInvalidSymbolId) return nullptr;
    const std::string text = symbols.text(name);
    for (const auto& e : entries_) {
        if (e.nameText != nullptr && std::strcmp(e.nameText, text.c_str()) == 0) {
            return &e.contract;
        }
    }
    return nullptr;
}

}  // namespace mlk
