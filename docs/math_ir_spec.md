# MLK+ Math IR Specification

## Values

A Value carries: `id` (ValueId, uint32 — Rule 15), `kind`
(Constant/Variable/Placeholder/NodeResult/Symbol), `MathType`, `FactSet`,
`name` (SymbolId — Rule 16), `producer`, `version`, cached stable hash.

`MathType` = {Domain, AlgebraicClass, Differentiability, Dtype,
optional<Shape>, optional<TensorDescriptor>, optional<FunctionSig>}.
Domains: Symbolic, Bool, Int, Float, Complex, Real, Function, Operator, Set,
Sequence, Distribution. Tensor is a structured object recognized by type
inference — not a base primitive.

## Nodes

A Node carries: `op` (MathOp), `inputs` (SmallVector<ValueId, 4> — Rule 19),
`results`, `attrs` (interned — Rule 16), `effects` (Rule 140), `flags`,
`facts`, and speculation metadata (Rules 5, 141).

MathOp groups (see `mlk/ir/math_op.h` for the full table):
- arithmetic: add, sub, mul, div, neg, pow
- math functions: exp, log, sin, cos, tan, tanh, sqrt, rsqrt, erf, gelu
- tensor: matmul, dot, transpose, reshape, broadcast, reduce_*, softmax,
  einsum, conv
- calculus (first-class): derivative, integral, gradient, limit
- abstract: solve, apply, lambda
- explicit conversions (Rule 39): int_to_float, float_to_int, real_to_complex,
  scalar_to_tensor, tensor_to_scalar, layout_transform, reinterpret, bitcast,
  box, unbox, native_to_math_ref, math_to_native_ref

## Graph laws

1. References are 32-bit indices; a value's producer always has a lower node
   id than its consumers (topological by construction; verifier-enforced).
2. Mutations bump `version()`; specializations may depend on it (Rule 89).
3. Equivalence is recorded (`recordEquivalent`), never silently destroyed;
   purges happen only through versioned, documented lowering decisions
   (Rule 21).
4. Every graph has a stable content hash (Rule 24) computed bottom-up from
   op + attrs + operand hashes.
5. The effect chain (Rule 140) is maintained on mutation and validated by
   `ir.verify`.

## Serialization

`.mlk` files are JSON with `format: "mlk-graph"`, a version field, dense
value ids in file order, node_result entries in node creation order, nodes in
topological order, and an `outputs` array. The loader is a trust boundary:
malformed input is rejected with actionable diagnostics (Rules 67, 124).
