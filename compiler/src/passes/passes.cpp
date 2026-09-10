// Pass registration core. Each pass is defined and registered in its own
// translation unit (see passes_common.h: 1:1 pass->file mapping); this file
// only fixes the deterministic registration order (Rule 143 — the registry
// itself additionally sorts by name id, so order here is belt-and-braces).
#include "passes_common.h"

namespace mlk::passes {

void registerPass(SymbolTable& symbols, Pass& pass, PassKind kind,
                  std::initializer_list<const char*> required,
                  std::initializer_list<const char*> produced,
                  std::initializer_list<const char*> invalidated,
                  std::initializer_list<Tier> tiers) {
    PassRegistrar reg(symbols, pass.nameText(), kind,
                      required, produced, invalidated, tiers, &pass);
}

void registerAllPasses(SymbolTable& symbols) {
    // Analysis (spec §8.1)
    register_ir_verify_pass(symbols);
    register_type_infer_pass(symbols);
    register_shape_infer_pass(symbols);
    register_property_infer_pass(symbols);
    register_effect_infer_pass(symbols);
    register_accuracy_analyze_pass(symbols);
    register_cost_roofline_pass(symbols);
    register_workload_bucket_pass(symbols);
    register_alias_infer_pass(symbols);
    // Math canonicalization (spec §8.2)
    register_math_canonicalize_pass(symbols);
    register_math_normalize_ops_pass(symbols);
    register_math_associative_flatten_pass(symbols);
    register_math_commutative_sort_pass(symbols);
    register_math_constant_fold_pass(symbols);
    register_math_identity_elim_pass(symbols);
    register_math_strength_reduce_pass(symbols);
    register_math_cse_pass(symbols);
    register_math_dce_pass(symbols);
    register_math_algebraic_simplify_pass(symbols);
    register_math_expression_balance_pass(symbols);
    // E-graph (spec §8.3)
    register_egraph_build_pass(symbols);
    register_egraph_saturate_pass(symbols);
    register_egraph_extract_pass(symbols);
    // Calculus (spec §8.4)
    register_calculus_derivative_symbolic_pass(symbols);
    register_calculus_derivative_simplify_pass(symbols);
    register_calculus_derivative_numeric_pass(symbols);
    register_calculus_integral_quadrature_pass(symbols);
    // Tensor (spec §8.5)
    register_tensor_layout_infer_pass(symbols);
    register_tensor_transpose_elim_pass(symbols);
    register_tensor_einsum_lower_pass(symbols);
    register_tensor_contraction_path_pass(symbols);
    register_tensor_fusion_find_pass(symbols);
    register_tensor_matmul_algorithm_select_pass(symbols);
    // Approximation (spec §8.6)
    register_approx_policy_gate_pass(symbols);
    register_approx_function_lower_pass(symbols);
    register_approx_ulp_verify_pass(symbols);
    // Schedule (spec §8.7)
    register_schedule_region_extract_pass(symbols);
    register_schedule_fuse_pass(symbols);
    register_schedule_tile_pass(symbols);
    register_schedule_vectorize_pass(symbols);
    register_schedule_parallelize_pass(symbols);
    // Physical/memory (spec §8.8)
    register_memory_liveness_pass(symbols);
    register_memory_buffer_plan_pass(symbols);
    register_memory_place_pass(symbols);
    register_memory_layout_select_pass(symbols);
    // Lowering (spec §8.11)
    register_lower_to_kernel_ir_pass(symbols);
    // Polyhedral (spec §8.13 — docs/polyhedral_spec.md)
    register_poly_scop_detect_pass(symbols);
    register_poly_dependence_pass(symbols);
    register_poly_schedule_pass(symbols);
    // Backend (spec §8.12)
    register_backend_emit_binary_pass(symbols);
}

}  // namespace mlk::passes
