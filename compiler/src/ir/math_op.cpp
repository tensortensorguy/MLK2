// MLK+ op metadata + property/effect tables (single source of truth).
//
// Rule 77: op tables must be declarative data, not scattered switch logic.
// The verifier, type inference, property inference, effect inference, the
// printer, and the cost model all consume these tables.
#include "mlk/ir/math_op.h"

#include "mlk/core/constants.h"

namespace mlk {

#define ARITY1 OpArity{1, 1}
#define ARITY2 OpArity{2, 2}
#define ARITYV OpArity{1, 255}

// clang-format off
static constexpr OpInfo kOpTable[] = {
    // Arithmetic
    {MathOp::Add,    "add",    ARITYV},
    {MathOp::Sub,    "sub",    ARITY2},
    {MathOp::Mul,    "mul",    ARITYV},
    {MathOp::Div,    "div",    ARITY2},
    {MathOp::Neg,    "neg",    ARITY1},
    {MathOp::Pow,    "pow",    ARITY2},
    // Math functions
    {MathOp::Exp,    "exp",    ARITY1},
    {MathOp::Log,    "log",    ARITY1},
    {MathOp::Sin,    "sin",    ARITY1},
    {MathOp::Cos,    "cos",    ARITY1},
    {MathOp::Tan,    "tan",    ARITY1},
    {MathOp::Tanh,   "tanh",   ARITY1},
    {MathOp::Sqrt,   "sqrt",   ARITY1},
    {MathOp::Rsqrt,  "rsqrt",  ARITY1},
    {MathOp::Erf,    "erf",    ARITY1},
    {MathOp::Gelu,   "gelu",   ARITY1},
    // Linear algebra / tensor
    {MathOp::MatMul,    "matmul",    ARITY2},
    {MathOp::Dot,       "dot",       ARITY2},
    {MathOp::Transpose, "transpose", ARITY1},
    {MathOp::Reshape,   "reshape",   ARITY1},
    {MathOp::Broadcast, "broadcast", ARITY1},
    {MathOp::ReduceSum, "reduce_sum", ARITY1},
    {MathOp::ReduceMax, "reduce_max", ARITY1},
    {MathOp::ReduceMean,"reduce_mean",ARITY1},
    {MathOp::Softmax,   "softmax",   ARITY1},
    {MathOp::Einsum,    "einsum",    ARITY2},
    {MathOp::Conv,      "conv",      ARITY2},
    // Calculus
    {MathOp::Derivative, "derivative", ARITY2},
    {MathOp::Integral,   "integral",   ARITY1},
    {MathOp::Gradient,   "gradient",   ARITY2},
    {MathOp::Limit,      "limit",      ARITY2},
    // Abstract / symbolic
    {MathOp::Solve,   "solve",   ARITY2},
    {MathOp::Apply,   "apply",   ARITYV},
    {MathOp::Lambda,  "lambda",  ARITYV},
    // Conversions (Rule 39)
    {MathOp::IntToFloat,     "int_to_float",      ARITY1},
    {MathOp::FloatToInt,     "float_to_int",      ARITY1},
    {MathOp::RealToComplex,  "real_to_complex",   ARITY1},
    {MathOp::ScalarToTensor, "scalar_to_tensor",  ARITY1},
    {MathOp::TensorToScalar, "tensor_to_scalar",  ARITY1},
    {MathOp::LayoutTransform,"layout_transform",  ARITY1},
    {MathOp::Reinterpret,    "reinterpret",       ARITY1},
    {MathOp::BitCast,        "bitcast",           ARITY1},
    {MathOp::Box,            "box",               ARITY1},
    {MathOp::Unbox,          "unbox",             ARITY1},
    {MathOp::NativeToMathRef,"native_to_math_ref",ARITY1},
    {MathOp::MathToNativeRef,"math_to_native_ref",ARITY1},
};
// clang-format on

static_assert(sizeof(kOpTable) / sizeof(kOpTable[0]) ==
                  static_cast<std::size_t>(MathOp::kCount),
              "op table must cover every MathOp (Rule 78: exhaustive)");

const OpInfo& opInfo(MathOp op) noexcept {
    const auto idx = static_cast<uint32_t>(op);
    if (idx < static_cast<uint32_t>(MathOp::kCount)) [[likely]] {
        return kOpTable[idx];
    }
    return kOpTable[0];
}

const char* opName(MathOp op) noexcept { return opInfo(op).name; }

bool opByName(const char* name, MathOp& out) noexcept {
    for (const auto& info : kOpTable) {
        if (__builtin_strcmp(info.name, name) == 0) {
            out = info.op;
            return true;
        }
    }
    return false;
}

bool isVariadic(MathOp op) noexcept { return opInfo(op).arity.max >= 255; }

bool arityAccepts(MathOp op, uint32_t n) noexcept {
    const OpArity& a = opInfo(op).arity;
    return n >= a.min && (a.max >= 255 || n <= a.max);
}

bool isElementwiseMath(MathOp op) noexcept {
    switch (op) {  // Rule 78: exhaustive switch on closed enum
        case MathOp::Add: case MathOp::Sub: case MathOp::Mul:
        case MathOp::Div: case MathOp::Neg: case MathOp::Pow:
        case MathOp::Exp: case MathOp::Log: case MathOp::Sin:
        case MathOp::Cos: case MathOp::Tan: case MathOp::Tanh:
        case MathOp::Sqrt: case MathOp::Rsqrt: case MathOp::Erf:
        case MathOp::Gelu:
            return true;
        default:
            return false;
    }
}

bool isReduction(MathOp op) noexcept {
    switch (op) {
        case MathOp::ReduceSum: case MathOp::ReduceMax:
        case MathOp::ReduceMean: case MathOp::Dot:
            return true;
        default:
            return false;
    }
}

bool isTensorOp(MathOp op) noexcept {
    switch (op) {
        case MathOp::MatMul: case MathOp::Transpose: case MathOp::Reshape:
        case MathOp::Broadcast: case MathOp::Softmax: case MathOp::Einsum:
        case MathOp::Conv: case MathOp::LayoutTransform:
            return true;
        default:
            return isReduction(op);
    }
}

bool isCalculusOp(MathOp op) noexcept {
    switch (op) {
        case MathOp::Derivative: case MathOp::Integral:
        case MathOp::Gradient: case MathOp::Limit:
            return true;
        default:
            return false;
    }
}

bool isConversionOp(MathOp op) noexcept {
    switch (op) {
        case MathOp::IntToFloat: case MathOp::FloatToInt:
        case MathOp::RealToComplex: case MathOp::ScalarToTensor:
        case MathOp::TensorToScalar: case MathOp::LayoutTransform:
        case MathOp::Reinterpret: case MathOp::BitCast: case MathOp::Box:
        case MathOp::Unbox: case MathOp::NativeToMathRef:
        case MathOp::MathToNativeRef:
            return true;
        default:
            return false;
    }
}

}  // namespace mlk
