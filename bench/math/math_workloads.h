// MLK+ math benchmark workloads (bench/math/; Rule 56: seeded, realistic
// workloads — golden-ratio low-discrepancy fills, never all-zero buffers).
//
// Each workload is the SPEC's flagship mathematical graph (math-IR
// realization spec §9): elementwise math, fused epilogues, and the
// calculus-first derivative workload d/dx[x²·sin(x)].
#pragma once

#include "mlk/core/result.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_builder.h"
#include "mlk/type/math_type.h"

#include <cmath>

namespace mlk::bench {

/// Golden-ratio low-discrepancy fill over [lo, hi] (deterministic seed).
inline void fillLowDiscrepancy(double* x, int64_t n, double lo, double hi) {
    constexpr double kGolden = 0.61803398874989484820;
    for (int64_t i = 0; i < n; ++i) {
        double f = std::fmod(static_cast<double>(i) * kGolden, 1.0);
        x[i] = lo + (hi - lo) * f;
    }
}

/// Rule 39 helper: explicit scalar->tensor lift with an annotated
/// broadcast-target type (the frontend inserts conversions; the IR never
/// coerces implicitly).
[[nodiscard]] Result<ValueId> liftScalar(GraphBuilder& b, ValueId scalar,
                                         const MathType& tensorType) {
    MLK_TRY_VAR(lifted, b.op(MathOp::ScalarToTensor, {scalar}));
    b.graph().value(lifted).type = tensorType;
    return lifted;
}

/// f64 tensor of the given shape (row-major, contiguous).
[[nodiscard]] inline MathType tensorF64(
    std::initializer_list<int64_t> dims) {
    return MathType::tensorValue(Dtype::F64, dims);
}

/// The six benchmark workloads. Each returns a freshly built MathGraph.
struct Workloads {
    /// y = sin(x² + 3x)  (spec §9 flagship example)
    [[nodiscard]] static Result<MathGraph> sinSq3x(SymbolTable& symbols,
                                                   int64_t n) {
        GraphBuilder b(symbols);
        const MathType t = tensorF64({n});
        const ValueId x = b.placeholder("x", t);
        MLK_TRY_VAR(x2, b.op(MathOp::Mul, {x, x}));
        const ValueId three =
            b.constant(3.0, MathType::scalar(Domain::Float, Dtype::F64));
        MLK_TRY_VAR(threeT, liftScalar(b, three, t));
        MLK_TRY_VAR(x3, b.op(MathOp::Mul, {threeT, x}));
        MLK_TRY_VAR(sum, b.op(MathOp::Add, {x2, x3}));
        MLK_TRY_VAR(s, b.op(MathOp::Sin, {sum}));
        b.output(s);
        return std::move(b.graph());
    }

    /// y = sin(x² + 3x) * a + b  (fused epilogue; spec §9 candidate 5)
    [[nodiscard]] static Result<MathGraph> sinSq3xFused(SymbolTable& symbols,
                                                        int64_t n) {
        GraphBuilder b(symbols);
        const MathType t = tensorF64({n});
        const MathType sc = MathType::scalar(Domain::Float, Dtype::F64);
        const ValueId x = b.placeholder("x", t);
        MLK_TRY_VAR(x2, b.op(MathOp::Mul, {x, x}));
        const ValueId three = b.constant(3.0, sc);
        MLK_TRY_VAR(threeT, liftScalar(b, three, t));
        MLK_TRY_VAR(x3, b.op(MathOp::Mul, {threeT, x}));
        MLK_TRY_VAR(sum, b.op(MathOp::Add, {x2, x3}));
        MLK_TRY_VAR(s, b.op(MathOp::Sin, {sum}));
        const ValueId a = b.constant(1.5, sc);
        MLK_TRY_VAR(aT, liftScalar(b, a, t));
        MLK_TRY_VAR(sa, b.op(MathOp::Mul, {s, aT}));
        const ValueId bb = b.constant(0.25, sc);
        MLK_TRY_VAR(bT, liftScalar(b, bb, t));
        MLK_TRY_VAR(y, b.op(MathOp::Add, {sa, bT}));
        b.output(y);
        return std::move(b.graph());
    }

    /// y = tanh(x)
    [[nodiscard]] static Result<MathGraph> tanhWl(SymbolTable& symbols,
                                                  int64_t n) {
        GraphBuilder b(symbols);
        const ValueId x = b.placeholder("x", tensorF64({n}));
        MLK_TRY_VAR(y, b.op(MathOp::Tanh, {x}));
        b.output(y);
        return std::move(b.graph());
    }

    /// y = exp(x)
    [[nodiscard]] static Result<MathGraph> expWl(SymbolTable& symbols,
                                                 int64_t n) {
        GraphBuilder b(symbols);
        const ValueId x = b.placeholder("x", tensorF64({n}));
        MLK_TRY_VAR(y, b.op(MathOp::Exp, {x}));
        b.output(y);
        return std::move(b.graph());
    }

    /// y = d/dx [x² · sin(x)]  — the calculus-first workload: the graph
    /// carries the Derivative NODE; the compiler lowers it symbolically
    /// (calculus.derivative_symbolic) and fuses the result.
    [[nodiscard]] static Result<MathGraph> derivativeX2SinX(
        SymbolTable& symbols, int64_t n) {
        GraphBuilder b(symbols);
        const MathType t = tensorF64({n});
        const ValueId x = b.placeholder("x", t);
        MLK_TRY_VAR(x2, b.op(MathOp::Mul, {x, x}));
        MLK_TRY_VAR(s, b.op(MathOp::Sin, {x}));
        MLK_TRY_VAR(f, b.op(MathOp::Mul, {x2, s}));
        MLK_TRY_VAR(df, b.op(MathOp::Derivative, {f, x}));
        b.output(df);
        return std::move(b.graph());
    }

    /// C = A · B, A=[M,K], B=[K,N] f64 row-major
    [[nodiscard]] static Result<MathGraph> matMul(SymbolTable& symbols,
                                                  int64_t m, int64_t k,
                                                  int64_t nn) {
        GraphBuilder b(symbols);
        const ValueId a = b.placeholder("A", tensorF64({m, k}));
        const ValueId bb = b.placeholder("B", tensorF64({k, nn}));
        MLK_TRY_VAR(c, b.op(MathOp::MatMul, {a, bb}));
        b.output(c);
        return std::move(b.graph());
    }
};

}  // namespace mlk::bench
