# MLK+ benchmark harness (mlk_bench driver; Rule 49 protocol).
mlk_add_executable(mlk_bench_runner
    SOURCES bench_runner.cpp
    DEPS mlk_runtime mlk_autotune
)

# Math benchmark suite: MLK+ Tier-2 realized kernels vs baselines
# (torch.compile / JAX / C drivers live in bench/math; Rule 32: measured
#  wins only, Rule 49: statistically valid protocol).
mlk_add_executable(mlk_math_bench
    SOURCES math/math_bench.cpp
    DEPS mlk_runtime mlk_autotune
)

# Compile-time benchmark: per-tier pipeline wall time + polyhedral phase
# split via kill switches (docs/benchmarks.md section 2; Rule 49 protocol).
mlk_add_executable(mlk_compile_bench
    SOURCES compile/compile_bench.cpp
    DEPS mlk_runtime mlk_autotune mlk_pipeline mlk_passes
)
