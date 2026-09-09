# MLK+ superoptimizer plugins.
mlk_add_library(mlk_superopt
    INCLUDE_DIR include
    SOURCES
        src/superopt.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_proof mlk_cost
)
