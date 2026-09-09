# Backends consume a lowered IR + target description — they never reach into
# random compiler internals (layout spec §3).
#
# mlk_backend_interpreter: Tier 0 reference execution (Part I tier 0).
# mlk_backend_cpu: CPU KernelModule execution + C++ source emission
#                  (spec §12 first backend path).
add_subdirectory(interpreter)
add_subdirectory(cpu)

# Optional backends are wired via MLK_USE_LLVM / MLK_USE_CUDA / etc.
# (cmake/dependencies.cmake); they stay OFF by default for hermetic builds.
if(NOT MLK_USE_LLVM)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/llvm_stub.cpp "")
endif()
