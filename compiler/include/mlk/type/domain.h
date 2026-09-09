// MLK+ mathematical domains.
//
// "Tensor is one domain built on top of a general mathematical graph, not
// the identity of the IR." The domain lattice below treats scalars, vectors,
// matrices, tensors, functions, operators, sets, sequences, and
// distributions as peers inside one type system (spec: Multi-Domain
// Mathematical Contract; realization spec §3 core object model).
#pragma once

#include <cstdint>

namespace mlk {

/// Base mathematical domain of a Value.
enum class Domain : uint8_t {
    Unknown = 0,
    Symbolic,    // unevaluated mathematical object
    Bool,
    Int,
    Float,       // IEEE-style floating point (dtype carries precision)
    Complex,
    Real,        // exact/abstract real (symbolic-real profiles)
    Function,    // function value f : R -> R etc.
    Operator,    // linear/abstract operator value
    Set,
    Sequence,
    Distribution,
};

/// Algebraic structure declared/inferred for a value (realization spec §3).
enum class AlgebraicClass : uint8_t {
    Unknown = 0,
    ScalarField,
    VectorSpace,
    MatrixAlgebra,
    TensorAlgebra,
    PolynomialRing,
    AbstractRing,
    AbstractGroup,
};

/// Differentiability class (realization spec §3).
enum class Differentiability : uint8_t {
    Unknown = 0,
    NonDifferentiable,
    PiecewiseDifferentiable,
    Differentiable,
    Smooth,
    Analytic,
};

/// Element dtypes for numeric domains (floats/ints/complex).
enum class Dtype : uint8_t {
    None = 0,
    Bool1,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, BF16, F32, F64,
    C64, C128,   // complex<float>, complex<double>
};

/// Byte size of one element of `dt`; 0 for None.
[[nodiscard]] int dtypeBytes(Dtype dt) noexcept;

/// Human-readable dtype name (cold path: printing, tools).
[[nodiscard]] const char* dtypeName(Dtype dt) noexcept;

const char* domainName(Domain d) noexcept;
const char* algebraicClassName(AlgebraicClass a) noexcept;
const char* differentiabilityName(Differentiability df) noexcept;

}  // namespace mlk
