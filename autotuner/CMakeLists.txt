# MLK+ autotuner.
mlk_add_library(mlk_autotune
    INCLUDE_DIR include
    SOURCES
        src/autotuner.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_proof mlk_cost mlk_passes
        mlk_pipeline mlk_runtime
)
