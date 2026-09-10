# mlk_poly — hermetic polyhedral engine (layout spec: new component).
#
# Dependency flow (strict, one-directional, Rule 70):
#   mlk_core + mlk_ir -> mlk_poly -> mlk_passes -> mlk_pipeline
#
# The engine is pure integer/rational arithmetic over index-based data
# structures; it has no external dependencies (Rule 160: hermetic build) and
# never throws (Rule 6: Result<T> everywhere).
mlk_add_library(mlk_poly
    INCLUDE_DIR include
    SOURCES
        src/rational.cpp
        src/affine_expr.cpp
        src/int_set.cpp
        src/affine_map.cpp
    PUBLIC_DEPS mlk_core mlk_ir
)

set(MLK_POLY_LIB mlk_poly PARENT_SCOPE)
