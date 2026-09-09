// Names for domains, dtypes, algebra classes, value kinds, properties.
// Cold-path textual layer (printers, diagnostics, tools). The IR itself
// stores only ids (Rule 16).
#include "mlk/type/domain.h"
#include "mlk/ir/value.h"
#include "mlk/property/property_lattice.h"

namespace mlk {

const char* domainName(Domain d) noexcept {
    switch (d) {  // Rule 78: exhaustive
        case Domain::Unknown: return "unknown";
        case Domain::Symbolic: return "symbolic";
        case Domain::Bool: return "bool";
        case Domain::Int: return "int";
        case Domain::Float: return "float";
        case Domain::Complex: return "complex";
        case Domain::Real: return "real";
        case Domain::Function: return "function";
        case Domain::Operator: return "operator";
        case Domain::Set: return "set";
        case Domain::Sequence: return "sequence";
        case Domain::Distribution: return "distribution";
    }
    return "?";
}

int dtypeBytes(Dtype dt) noexcept {
    switch (dt) {
        case Dtype::None: return 0;
        case Dtype::Bool1: return 1;
        case Dtype::I8: case Dtype::U8: return 1;
        case Dtype::I16: case Dtype::U16: case Dtype::F16: case Dtype::BF16:
            return 2;
        case Dtype::I32: case Dtype::U32: case Dtype::F32: case Dtype::C64:
            return 4;
        case Dtype::I64: case Dtype::U64: case Dtype::F64: case Dtype::C128:
            return 8;
    }
    return 0;
}

const char* dtypeName(Dtype dt) noexcept {
    switch (dt) {
        case Dtype::None: return "none";
        case Dtype::Bool1: return "bool";
        case Dtype::I8: return "i8";
        case Dtype::I16: return "i16";
        case Dtype::I32: return "i32";
        case Dtype::I64: return "i64";
        case Dtype::U8: return "u8";
        case Dtype::U16: return "u16";
        case Dtype::U32: return "u32";
        case Dtype::U64: return "u64";
        case Dtype::F16: return "f16";
        case Dtype::BF16: return "bf16";
        case Dtype::F32: return "f32";
        case Dtype::F64: return "f64";
        case Dtype::C64: return "c64";
        case Dtype::C128: return "c128";
    }
    return "?";
}

const char* algebraicClassName(AlgebraicClass a) noexcept {
    switch (a) {
        case AlgebraicClass::Unknown: return "unknown";
        case AlgebraicClass::ScalarField: return "scalar_field";
        case AlgebraicClass::VectorSpace: return "vector_space";
        case AlgebraicClass::MatrixAlgebra: return "matrix_algebra";
        case AlgebraicClass::TensorAlgebra: return "tensor_algebra";
        case AlgebraicClass::PolynomialRing: return "polynomial_ring";
        case AlgebraicClass::AbstractRing: return "abstract_ring";
        case AlgebraicClass::AbstractGroup: return "abstract_group";
    }
    return "?";
}

const char* differentiabilityName(Differentiability df) noexcept {
    switch (df) {
        case Differentiability::Unknown: return "unknown";
        case Differentiability::NonDifferentiable: return "non_differentiable";
        case Differentiability::PiecewiseDifferentiable:
            return "piecewise_differentiable";
        case Differentiability::Differentiable: return "differentiable";
        case Differentiability::Smooth: return "smooth";
        case Differentiability::Analytic: return "analytic";
    }
    return "?";
}

const char* valueKindName(ValueKind k) noexcept {
    switch (k) {
        case ValueKind::Constant: return "constant";
        case ValueKind::Variable: return "variable";
        case ValueKind::Placeholder: return "placeholder";
        case ValueKind::NodeResult: return "node_result";
        case ValueKind::Symbol: return "symbol";
    }
    return "?";
}

const char* propertyName(PropertyId p) noexcept {
    switch (p) {
        case PropertyId::Commutative: return "commutative";
        case PropertyId::Associative: return "associative";
        case PropertyId::Distributive: return "distributive";
        case PropertyId::Idempotent: return "idempotent";
        case PropertyId::HasIdentity: return "has_identity";
        case PropertyId::MonotonicIncreasing: return "monotonic_increasing";
        case PropertyId::MonotonicDecreasing: return "monotonic_decreasing";
        case PropertyId::Periodic: return "periodic";
        case PropertyId::Differentiable: return "differentiable";
        case PropertyId::Invertible: return "invertible";
        case PropertyId::Positive: return "positive";
        case PropertyId::NonNegative: return "non_negative";
        case PropertyId::Bounded: return "bounded";
        case PropertyId::Sparse: return "sparse";
        case PropertyId::Symmetric: return "symmetric";
        case PropertyId::Contiguous: return "contiguous";
        case PropertyId::Pure: return "pure";
        case PropertyId::IntegerValued: return "integer_valued";
        case PropertyId::kCount: return "?";
    }
    return "?";
}

bool propertyByName(const char* name, PropertyId& out) noexcept {
    static constexpr PropertyId kAll[] = {
        PropertyId::Commutative, PropertyId::Associative,
        PropertyId::Distributive, PropertyId::Idempotent,
        PropertyId::HasIdentity, PropertyId::MonotonicIncreasing,
        PropertyId::MonotonicDecreasing, PropertyId::Periodic,
        PropertyId::Differentiable, PropertyId::Invertible,
        PropertyId::Positive, PropertyId::NonNegative, PropertyId::Bounded,
        PropertyId::Sparse, PropertyId::Symmetric, PropertyId::Contiguous,
        PropertyId::Pure, PropertyId::IntegerValued};
    for (const PropertyId p : kAll) {
        if (__builtin_strcmp(propertyName(p), name) == 0) {
            out = p;
            return true;
        }
    }
    return false;
}

}  // namespace mlk
