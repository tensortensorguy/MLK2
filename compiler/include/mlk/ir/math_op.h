// MLK+ mathematical operations (spec: "nodes become mathematical
// transformations").
//
// One op table covers arithmetic, math functions, linear algebra/tensor ops,
// calculus (first-class: Derivative/Integral/Gradient/Limit are nodes), and
// abstract/symbolic operations. Explicit conversion nodes exist because Rule
// 39 forbids implicit conversions in the IR.
//
// Rule 15/77: op tables are data — the metadata table below is the single
// source of truth consumed by the verifier, type inference, property
// inference, effect inference, the printer, and the cost model.
#pragma once

#include <cstdint>

namespace mlk {

enum class MathOp : uint16_t {
    // --- Arithmetic ----------------------------------------------------------
    Add = 0,
    Sub,
    Mul,
    Div,
    Neg,
    Pow,

    // --- Math functions ------------------------------------------------------
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sqrt,
    Rsqrt,
    Erf,
    Gelu,

    // --- Linear algebra / tensor (one domain of many) ------------------------
    MatMul,
    Dot,
    Transpose,
    Reshape,
    Broadcast,
    ReduceSum,
    ReduceMax,
    ReduceMean,
    Softmax,
    Einsum,
    Conv,

    // --- Calculus (first-class nodes) ----------------------------------------
    Derivative,
    Integral,
    Gradient,
    Limit,

    // --- Abstract / symbolic ---------------------------------------------------
    Solve,
    Apply,
    Lambda,

    // --- Explicit conversions (Rule 39: no implicit conversions in IR) -------
    IntToFloat,
    FloatToInt,
    RealToComplex,
    ScalarToTensor,
    TensorToScalar,
    LayoutTransform,
    Reinterpret,
    BitCast,
    Box,
    Unbox,
    NativeToMathRef,
    MathToNativeRef,

    // --- Sentinel ------------------------------------------------------------
    kCount,
};

/// Operand arity: exact or variable.
struct OpArity {
    uint8_t min;
    uint8_t max;  // 255 = variadic
};

/// Static per-op metadata table (data-driven; Rule 77).
struct OpInfo {
    MathOp op;
    const char* name;  // canonical textual name (used by frontend/printer)
    OpArity arity;
};

[[nodiscard]] const OpInfo& opInfo(MathOp op) noexcept;
[[nodiscard]] const char* opName(MathOp op) noexcept;
/// Parses an op by canonical name; nullopt when unknown (tools/frontends).
[[nodiscard]] bool opByName(const char* name, MathOp& out) noexcept;
[[nodiscard]] bool isVariadic(MathOp op) noexcept;
[[nodiscard]] bool arityAccepts(MathOp op, uint32_t n) noexcept;

/// Ops that are legal on tensors/elementwise arrays (used by tensor passes
/// to classify work without hardcoding domain checks in generic passes —
/// Rule 28: knowledge lives in tables and profiles, not if-chains).
[[nodiscard]] bool isElementwiseMath(MathOp op) noexcept;
[[nodiscard]] bool isReduction(MathOp op) noexcept;
[[nodiscard]] bool isTensorOp(MathOp op) noexcept;
[[nodiscard]] bool isCalculusOp(MathOp op) noexcept;
[[nodiscard]] bool isConversionOp(MathOp op) noexcept;

}  // namespace mlk
