# CPU backend: KernelModule execution + C++ source emission for AOT
# artifacts (spec §12: Math IR -> Kernel IR -> C++ emitter -> clang/gcc).
mlk_add_library(mlk_backend_cpu
    INCLUDE_DIR include
    SOURCES
        src/cpp_emitter.cpp
    PUBLIC_DEPS mlk_runtime
)
