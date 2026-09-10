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
        src/ir/names.cpp
        src/ir/math_graph.cpp
        src/ir/graph_printer.cpp
        src/ir/graph_json.cpp
        src/support/kernel_ir.cpp
        # Rule 140: the effect model is part of the IR core. It lives here
        # to keep the dependency flow acyclic (effect inference is an
        # op-table concern, not an upstream layer).
        src/effect/effect_inference.cpp
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

# mlk_effect: the effect system headers + implementation now live in
# mlk_ir (Rule 140: effects are IR core). Kept as an interface target so the
# CMake target list matches the layout spec exactly.
add_library(mlk_effect INTERFACE)
target_link_libraries(mlk_effect INTERFACE mlk_ir)

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
    # Rule 26 (one component, one concern): every pass lives in its own
    # translation unit named exactly after the pass. Adding a pass = adding
    # one file + one declaration in passes_common.h + one call in
    # passes.cpp. The e-graph engine (egraph.{h,cpp}) is shared support,
    # not a pass.
    SOURCES
        src/passes/passes.cpp
        # Analysis (spec §8.1)
        src/passes/analysis/ir.verify.cpp
        src/passes/analysis/type.infer.cpp
        src/passes/analysis/shape.infer.cpp
        src/passes/analysis/property.infer.cpp
        src/passes/analysis/effect.infer.cpp
        src/passes/analysis/accuracy.analyze.cpp
        src/passes/analysis/cost.roofline.cpp
        src/passes/analysis/workload.bucket.cpp
        src/passes/analysis/alias.infer.cpp
        # Math canonicalization (spec §8.2)
        src/passes/math/math.canonicalize.cpp
        src/passes/math/math.normalize_ops.cpp
        src/passes/math/math.associative_flatten.cpp
        src/passes/math/math.commutative_sort.cpp
        src/passes/math/math.constant_fold.cpp
        src/passes/math/math.identity_elim.cpp
        src/passes/math/math.strength_reduce.cpp
        src/passes/math/math.cse.cpp
        src/passes/math/math.dce.cpp
        src/passes/math/math.algebraic_simplify.cpp
        src/passes/math/math.expression_balance.cpp
        # E-graph (spec §8.3); egraph.cpp = shared engine
        src/passes/egraph/egraph.cpp
        src/passes/egraph/egraph.build.cpp
        src/passes/egraph/egraph.saturate.cpp
        src/passes/egraph/egraph.extract.cpp
        # Calculus (spec §8.4)
        src/passes/calculus/calculus.derivative_symbolic.cpp
        src/passes/calculus/calculus.derivative_simplify.cpp
        src/passes/calculus/calculus.derivative_numeric.cpp
        src/passes/calculus/calculus.integral_quadrature.cpp
        # Tensor (spec §8.5)
        src/passes/tensor/tensor.layout_infer.cpp
        src/passes/tensor/tensor.transpose_elim.cpp
        src/passes/tensor/tensor.einsum_lower.cpp
        src/passes/tensor/tensor.contraction_path.cpp
        src/passes/tensor/tensor.fusion_find.cpp
        src/passes/tensor/tensor.matmul_algorithm_select.cpp
        # Approximation (spec §8.6)
        src/passes/approx/approx.policy_gate.cpp
        src/passes/approx/approx.function_lower.cpp
        src/passes/approx/approx.ulp_verify.cpp
        # Schedule (spec §8.7)
        src/passes/schedule/schedule.region_extract.cpp
        src/passes/schedule/schedule.fuse.cpp
        src/passes/schedule/schedule.tile.cpp
        src/passes/schedule/schedule.vectorize.cpp
        src/passes/schedule/schedule.parallelize.cpp
        # Physical/memory (spec §8.8)
        src/passes/physical/memory.liveness.cpp
        src/passes/physical/memory.buffer_plan.cpp
        src/passes/physical/memory.place.cpp
        src/passes/physical/memory.layout_select.cpp
        # Lowering (spec §8.11)
        src/passes/lower/lower.to_kernel_ir.cpp
        # Polyhedral (spec §8.13 — docs/polyhedral_spec.md; 1:1 pass->file)
        src/passes/poly/poly.scop_detect.cpp
        src/passes/poly/poly.dependence.cpp
        # Backend (spec §8.12)
        src/passes/backend/backend.emit_binary.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_property mlk_effect mlk_proof
        mlk_verifier mlk_pass_core mlk_cost mlk_poly
)

mlk_add_library(mlk_pipeline
    INCLUDE_DIR include
    SOURCES
        src/pipeline/pipeline_runner.cpp
    PUBLIC_DEPS mlk_core mlk_ir mlk_type mlk_proof mlk_verifier
        mlk_pass_core mlk_passes mlk_cost
)

set(MLK_COMPILER_LIBS mlk_core mlk_ir mlk_type mlk_property mlk_effect
    mlk_proof mlk_verifier mlk_pass_core mlk_cost mlk_passes mlk_pipeline
    PARENT_SCOPE)
