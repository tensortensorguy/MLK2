# MLK+ driver tools (layout spec §13).
mlk_add_executable(mlkc
    SOURCES mlkc/mlkc.cpp
    DEPS mlk_runtime mlk_autotune mlk_superopt mlk_backend_cpu
)
mlk_add_executable(mlk-verify
    SOURCES mlk-verify/mlk_verify.cpp
    DEPS mlk_runtime mlk_verifier
)
mlk_add_executable(mlk-tune
    SOURCES mlk-tune/mlk_tune.cpp
    DEPS mlk_runtime mlk_autotune
)
mlk_add_executable(mlk-bench
    SOURCES mlk-bench/mlk_bench.cpp
    DEPS mlk_runtime mlk_autotune
)
mlk_add_executable(mlk-profile
    SOURCES mlk-profile/mlk_profile.cpp
    DEPS mlk_runtime
)
mlk_add_executable(mlk-replay
    SOURCES mlk-replay/mlk_replay.cpp
    DEPS mlk_runtime
)
mlk_add_executable(mlk-poly
    SOURCES mlk-poly/mlk_poly.cpp
    DEPS mlk_runtime mlk_backend_cpu mlk_poly
)
mlk_add_executable(mlk-generate-profiles
    SOURCES mlk-generate-profiles/mlk_generate_profiles.cpp
    DEPS mlk_type
)
