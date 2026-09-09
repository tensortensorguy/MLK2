// MLK+ node attributes.
//
// Rule 16: identifiers are interned SymbolIds — attrs are keyed by SymbolId,
// never by string. Values are tagged: i64 / f64 / bool / SymbolId.
#pragma once

#include <cstdint>
#include <variant>

#include "mlk/core/hash.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"

namespace mlk {

struct AttrValue {
    std::variant<int64_t, double, bool, SymbolId> v;

    [[nodiscard]] bool operator==(const AttrValue& o) const {
        return v == o.v;
    }
    [[nodiscard]] HashValue hash() const noexcept {
        if (std::holds_alternative<int64_t>(v)) {
            return hashI64(std::get<int64_t>(v));
        }
        if (std::holds_alternative<double>(v)) {
            return hashF64(std::get<double>(v));
        }
        if (std::holds_alternative<bool>(v)) {
            return hashU64(std::get<bool>(v) ? 1 : 0);
        }
        return hashU64(std::get<SymbolId>(v));
    }
};

inline constexpr uint32_t kMaxAttrsPerNode = 8;

struct Attr {
    SymbolId name{kInvalidSymbolId};
    AttrValue value{};
};

/// Attribute list: small, inline, ordered, deterministic iteration.
using AttrList = SmallVector<Attr, constants::kOperandInline>;

[[nodiscard]] inline const AttrValue* findAttr(const AttrList& attrs,
                                               SymbolId name) {
    for (const auto& a : attrs) {
        if (a.name == name) return &a.value;
    }
    return nullptr;
}

[[nodiscard]] inline const int64_t* findAttrI(const AttrList& attrs,
                                              SymbolId name) {
    const AttrValue* v = findAttr(attrs, name);
    return v && std::holds_alternative<int64_t>(v->v)
               ? &std::get<int64_t>(v->v)
               : nullptr;
}

[[nodiscard]] inline const double* findAttrF(const AttrList& attrs,
                                             SymbolId name) {
    const AttrValue* v = findAttr(attrs, name);
    return v && std::holds_alternative<double>(v->v)
               ? &std::get<double>(v->v)
               : nullptr;
}

}  // namespace mlk
