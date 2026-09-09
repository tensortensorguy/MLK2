// Dtype/Shape implementation for mlk_type (names live in mlk_ir names.cpp
// to keep the mlk_ir layer free of upward dependencies).
#include "mlk/type/domain.h"

#include "mlk/core/constants.h"
#include "mlk/type/shape.h"

namespace mlk {

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

bool Shape::broadcast(const Shape& a, const Shape& b, Shape& out) {
    const std::size_t ra = a.rank();
    const std::size_t rb = b.rank();
    const std::size_t r = ra > rb ? ra : rb;
    if (r > constants::kMaxRank) return false;
    SmallVector<int64_t, constants::kShapeInline> dims;
    for (std::size_t i = 0; i < r; ++i) {
        const int64_t da = i < r - ra ? 1 : a.dim(i - (r - ra));
        const int64_t db = i < r - rb ? 1 : b.dim(i - (r - rb));
        if (da == db) {
            dims.push_back(da);
        } else if (da == 1) {
            dims.push_back(db);
        } else if (db == 1) {
            dims.push_back(da);
        } else if (da == kDynamicDim) {
            dims.push_back(db);
        } else if (db == kDynamicDim) {
            dims.push_back(da);
        } else {
            return false;  // incompatible: diagnosed by caller (Rule 67)
        }
    }
    out.setDims(std::move(dims));
    return true;
}

}  // namespace mlk
