// MLK+ shared pass implementation helpers (internal to compiler/src/passes).
//
// File layout (Rule 26: one component, one concern): EVERY pass lives in
// its own translation unit named exactly after the pass — math.constant_fold
// lives in math/math.constant_fold.cpp, egraph.build in
// egraph/egraph.build.cpp, and so on. With 40+ passes, category files would
// become unmaintainable; a 1:1 pass->file mapping keeps review, kill-switch
// audits, and pass isolation (Rule 60) tractable.
//
// Shared rewrite helpers that several passes need live in
// math/math_rewrite_utils.h; the e-graph engine (not a pass) lives in
// egraph/egraph.{h,cpp}.
#pragma once

#include "mlk/core/diagnostics.h"
#include "mlk/core/symbol_table.h"
#include "mlk/ir/graph_printer.h"
#include "mlk/pass/pass.h"
#include "mlk/pass/pass_registry.h"

namespace mlk::passes {

/// Base class providing name/kind plumbing + telemetry + budget discipline
/// (Rule 10: passes must be idempotent/monotonic; budget checked by caller).
class PassBase : public Pass {
public:
    PassBase(SymbolTable& symbols, const char* name, PassKind kind)
        : symbols_(symbols), kind_(kind), nameStr_(name) {}

    [[nodiscard]] const char* nameText() const override {
        return nameStr_.c_str();
    }
    [[nodiscard]] PassKind kind() const override { return kind_; }
    /// Table-scoped id (resolved through the CURRENT context table).
    [[nodiscard]] SymbolId nameId(SymbolTable& symbols) const {
        return symbols.intern(nameStr_);
    }

protected:
    SymbolTable& symbols_;
    PassKind kind_;
    std::string nameStr_;
};

/// Registers a pass with its contract (Rule 142).
void registerPass(SymbolTable& symbols, Pass& pass, PassKind kind,
                  std::initializer_list<const char*> required,
                  std::initializer_list<const char*> produced,
                  std::initializer_list<const char*> invalidated,
                  std::initializer_list<Tier> tiers);

/// Registers every pass in the compiler (deterministic order).
void registerAllPasses(SymbolTable& symbols);

// --- Per-pass registration (one definition per pass file) -------------------
// Analysis (analysis/)
void register_ir_verify_pass(SymbolTable& symbols);
void register_type_infer_pass(SymbolTable& symbols);
void register_shape_infer_pass(SymbolTable& symbols);
void register_property_infer_pass(SymbolTable& symbols);
void register_effect_infer_pass(SymbolTable& symbols);
void register_accuracy_analyze_pass(SymbolTable& symbols);
void register_cost_roofline_pass(SymbolTable& symbols);
void register_workload_bucket_pass(SymbolTable& symbols);
void register_alias_infer_pass(SymbolTable& symbols);
// Math (math/)
void register_math_canonicalize_pass(SymbolTable& symbols);
void register_math_normalize_ops_pass(SymbolTable& symbols);
void register_math_associative_flatten_pass(SymbolTable& symbols);
void register_math_commutative_sort_pass(SymbolTable& symbols);
void register_math_constant_fold_pass(SymbolTable& symbols);
void register_math_identity_elim_pass(SymbolTable& symbols);
void register_math_strength_reduce_pass(SymbolTable& symbols);
void register_math_cse_pass(SymbolTable& symbols);
void register_math_dce_pass(SymbolTable& symbols);
void register_math_algebraic_simplify_pass(SymbolTable& symbols);
void register_math_expression_balance_pass(SymbolTable& symbols);
// E-graph (egraph/)
void register_egraph_build_pass(SymbolTable& symbols);
void register_egraph_saturate_pass(SymbolTable& symbols);
void register_egraph_extract_pass(SymbolTable& symbols);
// Calculus (calculus/)
void register_calculus_derivative_symbolic_pass(SymbolTable& symbols);
void register_calculus_derivative_simplify_pass(SymbolTable& symbols);
void register_calculus_derivative_numeric_pass(SymbolTable& symbols);
void register_calculus_integral_quadrature_pass(SymbolTable& symbols);
// Tensor (tensor/)
void register_tensor_layout_infer_pass(SymbolTable& symbols);
void register_tensor_transpose_elim_pass(SymbolTable& symbols);
void register_tensor_einsum_lower_pass(SymbolTable& symbols);
void register_tensor_contraction_path_pass(SymbolTable& symbols);
void register_tensor_fusion_find_pass(SymbolTable& symbols);
void register_tensor_matmul_algorithm_select_pass(SymbolTable& symbols);
// Approximation (approx/)
void register_approx_policy_gate_pass(SymbolTable& symbols);
void register_approx_function_lower_pass(SymbolTable& symbols);
void register_approx_ulp_verify_pass(SymbolTable& symbols);
// Schedule (schedule/)
void register_schedule_region_extract_pass(SymbolTable& symbols);
void register_schedule_fuse_pass(SymbolTable& symbols);
void register_schedule_tile_pass(SymbolTable& symbols);
void register_schedule_vectorize_pass(SymbolTable& symbols);
void register_schedule_parallelize_pass(SymbolTable& symbols);
// Physical/memory (physical/)
void register_memory_liveness_pass(SymbolTable& symbols);
void register_memory_buffer_plan_pass(SymbolTable& symbols);
void register_memory_place_pass(SymbolTable& symbols);
void register_memory_layout_select_pass(SymbolTable& symbols);
// Lowering (lower/)
void register_lower_to_kernel_ir_pass(SymbolTable& symbols);
// Polyhedral (poly/; spec §8.13 — docs/polyhedral_spec.md)
void register_poly_scop_detect_pass(SymbolTable& symbols);
void register_poly_dependence_pass(SymbolTable& symbols);
// Backend (backend/)
void register_backend_emit_binary_pass(SymbolTable& symbols);

}  // namespace mlk::passes
