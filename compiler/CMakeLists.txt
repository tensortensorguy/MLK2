# MLK+ compiler libraries (layout spec §3).
#
# Dependency flow is strict and one-directional:
#   mlk_core -> mlk_ir -> mlk_type/property/effect -> proof/verifier
#             -> pass_core -> passes -> pipeline
#
# Each target only sees its direct dependencies; backends never reach into
# random compiler internals — they consume lowered IR + target description.

mlk_add_library(mlk_core
    INCLUDE_DIR include
    SOURCES
        src/core/diagnostics.cpp
        src/core/json.cpp
)

mlk_add_library(mlk_ir
    INCLUDE_DIR include
    SOURCES
        src/ir/math_op.cpp
        src/ir/math_graph.cpp
        src/ir/graph_printer.cpp
        src/ir/graph_json.cpp
    PUBLIC_DEPS mlk_core
)

mlk_add_library(mlk_type
    INCLUDE_DIR include
    SOURCES
        src/type/domain.cpp
        src/type/domain_profile.cpp
        src/type/type_inference.cpp
    PUBLIC_DEPS mlk_core mlk_ir
)

mlk_add_library(mlk_property
    INCLUDE_DIR include
    SOURCES
        src/property/property_inference.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type
)

mlk_add_library(mlk_effect
    INCLUDE_DIR include
    SOURCES
        src/effect/effect_inference.cpp
    PUBLIC_DEPS mlk_core mlk_ir
)

mlk_add_library(mlk_proof
    INCLUDE_DIR include
    SOURCES
        src/proof/accuracy_contract.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type
)

mlk_add_library(mlk_verifier
    INCLUDE_DIR include
    SOURCES
        src/verifier/graph_verifier.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_property mlk_effect
)

mlk_add_library(mlk_pass_core
    INCLUDE_DIR include
    SOURCES
        src/pass/pass_registry.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_property mlk_effect mlk_proof
)

mlk_add_library(mlk_cost
    INCLUDE_DIR include
    SOURCES
        src/cost/cost_model.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type
)

mlk_add_library(mlk_passes
    INCLUDE_DIR include
    SOURCES
        src/passes/passes.cpp
        src/passes/analysis/analysis_passes.cpp
        src/passes/math/math_passes.cpp
        src/passes/egraph/egraph.cpp
        src/passes/egraph/egraph_passes.cpp
        src/passes/calculus/calculus_passes.cpp
        src/passes/tensor/tensor_passes.cpp
        src/passes/approx/approx_passes.cpp
        src/passes/schedule/schedule_passes.cpp
        src/passes/physical/physical_passes.cpp
        src/passes/lower/lower_passes.cpp
        src/passes/backend/backend_passes.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_property mlk_effect mlk_proof
        mlk_verifier mlk_pass_core mlk_cost
)

set(MLK_COMPILER_LIBS mlk_core mlk_ir mlk_type mlk_property mlk_effect
    mlk_proof mlk_verifier mlk_pass_core mlk_cost mlk_passes PARENT_SCOPE)
