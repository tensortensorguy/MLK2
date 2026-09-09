# MLK+ benchmark harness (mlk_bench driver; Rule 49 protocol).
mlk_add_executable(mlk_bench_runner
    SOURCES bench_runner.cpp
    DEPS mlk_runtime mlk_autotune
)
