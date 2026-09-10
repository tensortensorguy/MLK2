# MLK+ root build script.
#
# Component monorepo (see docs/architecture.md). Dependency flow is strict and
# one-directional (Rule 70, layout spec §3):
#
#   mlk_core -> mlk_ir -> {mlk_type,mlk_property,mlk_effect}
#            -> {mlk_proof,mlk_verifier} -> mlk_pass_core -> mlk_passes
#            -> mlk_pipeline -> {mlk_runtime,mlk_autotune,mlk_superopt,backends}
#
# The compiler/runtime/autotuner/superopt trees are compiled with
# -fno-exceptions -fno-rtti (Rules 6 and 8). There are no exceptions to that.
cmake_minimum_required(VERSION 3.24)

project(mlkplus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(compiler_flags)
include(sanitizers)
include(dependencies)
include(tools)
include(mlk_library)

option(MLK_BUILD_TOOLS   "Build MLK+ driver tools"                ON)
option(MLK_BUILD_TESTS   "Build MLK+ test suites"                 ON)
option(MLK_BUILD_BENCH   "Build MLK+ benchmark harness"           ON)
option(MLK_BUILD_EXAMPLES "Build MLK+ examples"                   ON)
option(MLK_WARNINGS_AS_ERRORS "Treat warnings as errors in MLK+ code" ON)

set(MLK_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Runtime output directory")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${MLK_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${MLK_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${MLK_OUTPUT_DIR}/lib")

mlk_apply_global_flags()

# ---------------------------------------------------------------------------
# Core compiler libraries (strict dependency order, layout spec §3)
# ---------------------------------------------------------------------------
add_subdirectory(compiler)

# ---------------------------------------------------------------------------
# Polyhedral engine (mlk_poly): hermetic Presburger/affine toolkit consumed
# by mlk_passes; sits directly above mlk_ir in the dependency flow.
# ---------------------------------------------------------------------------
add_subdirectory(polyhedral)

# ---------------------------------------------------------------------------
# Runtime, autotuner, superoptimizer, backends
# ---------------------------------------------------------------------------
add_subdirectory(runtime)
add_subdirectory(autotuner)
add_subdirectory(superopt)
add_subdirectory(backends)

# ---------------------------------------------------------------------------
# Tools, tests, bench, examples
# ---------------------------------------------------------------------------
if(MLK_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
if(MLK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(MLK_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(MLK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

message(STATUS "MLK+ configured: C++${CMAKE_CXX_STANDARD}, warnings-as-errors=${MLK_WARNINGS_AS_ERRORS}")
