# MLK+ runtime: GraphState, fallback, execution, telemetry, safepoints, cache.
mlk_add_library(mlk_runtime
    INCLUDE_DIR include
    SOURCES
        src/runtime_core.cpp
        src/execution/execution.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_proof mlk_pass_core mlk_passes
        mlk_pipeline mlk_verifier
)
