# CPU backend: KernelModule execution + AOT artifact emission (C++ and
# x86-64 assembly) + the out-of-process driver (spec §12: Math IR ->
# Kernel IR -> emitter -> system toolchain; ADR-0003/0006: no in-process
# machine codegen anywhere — the driver builds a .so with the external
# compiler and loads it, so no emitter-owned W^X pages exist).
mlk_add_library(mlk_backend_cpu
    INCLUDE_DIR include
    SOURCES
        src/cpp_emitter.cpp
        src/asm_emitter.cpp
        src/cuda_emitter.cpp
        src/slab_plan.cpp
        src/backend_driver.cpp
    PUBLIC_DEPS mlk_runtime
)
# dlopen/dlsym for the loaded artifacts (empty on platforms where dl is
# part of libc; the driver itself is POSIX-guarded).
target_link_libraries(mlk_backend_cpu PUBLIC ${CMAKE_DL_LIBS})
