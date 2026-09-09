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

# Additional libraries are appended below as they are implemented:
# mlk_type, mlk_property, mlk_effect, mlk_proof, mlk_verifier,
# mlk_pass_core, mlk_passes, mlk_pipeline, mlk_cost.
set(MLK_COMPILER_LIBS mlk_core mlk_ir PARENT_SCOPE)
