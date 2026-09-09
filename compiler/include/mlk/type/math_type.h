// MLK+ abstract mathematical type (spec: "types become much more abstract").
//
// A Type describes: base domain, dimensions, algebraic structure,
// differentiability, and constraints. "Tensor" is a particular structured
// mathematical object (TensorDescriptor attached to a MathType) rather than
// a special primitive — Rule 23 keeps its physical attributes (strides,
// layout, memory space) in the physical layer, not the math type.
#pragma once

#include <cstdint>
#include <optional>

#include "mlk/core/hash.h"
#include "mlk/type/domain.h"
#include "mlk/type/shape.h"

namespace mlk {

/// Physical-reality attributes of a tensor-typed value. Note (Rule 23): the
/// *memory space / allocation* side lives in the physical layer; this
/// descriptor only carries the mathematical/representational facts the graph
/// itself reasons about.
struct TensorDescriptor {
    Dtype element{Dtype::F32};
    Shape shape{};
    SmallVector<int64_t, constants::kShapeInline> strides{};  // elements
    bool contiguous{true};

    [[nodiscard]] bool operator==(const TensorDescriptor& o) const {
        return element == o.element && shape == o.shape &&
               strides == o.strides && contiguous == o.contiguous;
    }
    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashU64(static_cast<uint64_t>(element));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(shape.rank())));
        for (std::size_t i = 0; i < shape.rank(); ++i) {
            h = hashCombine(h, hashI64(shape.dim(i)));
        }
        for (std::size_t i = 0; i < strides.size(); ++i) {
            h = hashCombine(h, hashI64(strides[i]));
        }
        return h;
    }
};

/// Signature of a function-typed value: (params) -> result.
struct FunctionSig {
    SmallVector<Domain, 4> params{};
    Domain result{Domain::Unknown};

    [[nodiscard]] bool operator==(const FunctionSig& o) const {
        if (params.size() != o.params.size() || result != o.result) {
            return false;
        }
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (params[i] != o.params[i]) return false;
        }
        return true;
    }
    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashU64(static_cast<uint64_t>(result));
        for (std::size_t i = 0; i < params.size(); ++i) {
            h = hashCombine(h, hashU64(static_cast<uint64_t>(params[i])));
        }
        return h;
    }
};

/// The abstract type of a Value.
struct MathType {
    Domain domain{Domain::Unknown};
    AlgebraicClass algebra{AlgebraicClass::Unknown};
    Differentiability diff{Differentiability::Unknown};
    Dtype dtype{Dtype::None};

    /// Present for shaped domains (Tensor/Vector/Matrix/Sequence/Set...).
    std::optional<Shape> shape;
    /// Present iff domain == Float/Int/Bool elementwise tensor object.
    std::optional<TensorDescriptor> tensor;
    /// Present iff domain == Function.
    std::optional<FunctionSig> func;

    [[nodiscard]] bool isScalarLike() const {
        return shape == std::nullopt && tensor == std::nullopt &&
               func == std::nullopt;
    }
    [[nodiscard]] bool isTensor() const { return tensor.has_value(); }
    [[nodiscard]] bool isFunction() const {
        return domain == Domain::Function;
    }

    [[nodiscard]] bool operator==(const MathType& o) const {
        return domain == o.domain && algebra == o.algebra &&
               diff == o.diff && dtype == o.dtype && shape == o.shape &&
               tensor == o.tensor && func == o.func;
    }
    [[nodiscard]] bool operator!=(const MathType& o) const {
        return !(*this == o);
    }

    [[nodiscard]] HashValue hash() const noexcept {
        HashValue h = hashU64(static_cast<uint64_t>(domain));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(algebra)));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(diff)));
        h = hashCombine(h, hashU64(static_cast<uint64_t>(dtype)));
        if (shape) {
            h = hashCombine(h, hashU64(static_cast<uint64_t>(shape->rank())));
            for (std::size_t i = 0; i < shape->rank(); ++i) {
                h = hashCombine(h, hashI64(shape->dim(i)));
            }
        }
        if (tensor) h = hashCombine(h, tensor->hash());
        if (func) h = hashCombine(h, func->hash());
        return h;
    }

    /// Convenience constructors used by frontends.
    [[nodiscard]] static MathType scalar(Domain d, Dtype dt) {
        MathType t;
        t.domain = d;
        t.dtype = dt;
        return t;
    }
    [[nodiscard]] static MathType tensorValue(Dtype element,
                                              std::initializer_list<int64_t> dims) {
        MathType t;
        t.domain = Domain::Float;
        t.dtype = element;
        Shape s{dims};
        t.shape = s;
        TensorDescriptor td;
        td.element = element;
        td.shape = s;
        t.tensor = td;
        return t;
    }
    [[nodiscard]] static MathType functionValue(FunctionSig sig) {
        MathType t;
        t.domain = Domain::Function;
        t.func = std::move(sig);
        return t;
    }
};

}  // namespace mlk
