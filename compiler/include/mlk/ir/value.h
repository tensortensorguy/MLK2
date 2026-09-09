// MLK+ Value (spec: the fundamental thing is a Value, not a TensorValue).
//
//   Value
//   ├── Type (MathType)
//   ├── Domain
//   ├── Shape / dimensionality (inside MathType)
//   ├── Representation (kind + TensorDescriptor when applicable)
//   ├── Properties (FactSet)
//   └── Version
//
// Scalars, tensors, functions, derivatives, integrals, constants — all
// participate as Values in the same graph.
#pragma once

#include "mlk/core/flags.h"
#include "mlk/core/hash.h"
#include "mlk/core/small_vector.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/attrs.h"
#include "mlk/ir/node_id.h"
#include "mlk/ir/value_id.h"
#include "mlk/property/fact.h"
#include "mlk/type/math_type.h"

namespace mlk {

enum class ValueKind : uint8_t {
    Constant = 0,   // mathematical constant (2, pi, e)
    Variable,       // named input/parameter
    Placeholder,    // graph input slot
    NodeResult,     // produced by a node
    Symbol,         // symbolic atom (unevaluated)
};

const char* valueKindName(ValueKind k) noexcept;

enum class ValueFlag : uint8_t {
    Dead,          // marked dead by DCE (kept for provenance until purge)
    Canonical,     // canonical representative of its equivalence class
    kCount,
};
using ValueFlags = Flags<ValueFlag>;

/// Payload of a constant value (Rule 16: no strings; Rule 24: hashable).
struct ConstantPayload {
    bool isInt{false};
    double f64{0.0};
    int64_t i64{0};

    [[nodiscard]] bool operator==(const ConstantPayload& o) const {
        return isInt == o.isInt && f64 == o.f64 && i64 == o.i64;
    }
    [[nodiscard]] HashValue hash() const noexcept {
        return isInt ? hashI64(i64) : hashF64(f64);
    }
};

struct Value {
    ValueId id{kInvalidValueId};
    ValueKind kind{ValueKind::Variable};
    ConstantPayload constant{};
    MathType type{};
    FactSet facts{};
    SymbolId name{kInvalidSymbolId};
    NodeId producer{kInvalidNodeId};
    /// Bumped on documented lowering decisions (Rule 89: versioned deps).
    uint32_t version{0};
    ValueFlags flags{};
    /// Structural hash cache; maintained by MathGraph (Rule 24).
    mutable HashValue hash{0};
    mutable bool hashValid{false};

    [[nodiscard]] bool isConstant() const { return kind == ValueKind::Constant; }
    [[nodiscard]] bool isVariable() const { return kind == ValueKind::Variable; }
    [[nodiscard]] bool isPlaceholder() const {
        return kind == ValueKind::Placeholder;
    }
    [[nodiscard]] bool isSymbol() const { return kind == ValueKind::Symbol; }
};

}  // namespace mlk
