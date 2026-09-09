# Compiler/linker flags for all MLK+ targets.
#
# Rules enforced here (to the letter):
#   Rule 6  - NO EXCEPTIONS on the hot path: -fno-exceptions for compiler,
#             runtime, autotuner, superopt, backends, tools, tests.
#   Rule 8  - NO RTTI: -fno-rtti everywhere in MLK+ code.
#   Rule 25 - C++26 hints are used in source ([[likely]], [[unlikely]],
#             [[assume]]); the -std=c++26 requirement lives in the root file.
#   Rule 81 - Release builds optimize hard (-O3), no exceptions machinery.

add_library(mlk_cxx_flags INTERFACE)

target_compile_features(mlk_cxx_flags INTERFACE cxx_std_26)

if(MSVC)
    target_compile_options(mlk_cxx_flags INTERFACE
        /W4
        /Zc:preprocessor
        /EHs-c-
        /GR-
    )
    target_compile_definitions(mlk_cxx_flags INTERFACE _HAS_EXCEPTIONS=0)
else()
    target_compile_options(mlk_cxx_flags INTERFACE
        -fno-exceptions        # Rule 6
        -fno-rtti              # Rule 8
        -Wall
        -Wextra
        -Wpedantic
        -Wnon-virtual-dtor
        -Wcast-align
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # Known GCC false positives with std::variant<..., std::string, ...>
        # member copies under -O2/-O3 (GCC PR 108157 family). Narrow
        # suppression; -Wall -Wextra remain fully active.
        target_compile_options(mlk_cxx_flags INTERFACE -Wno-maybe-uninitialized)
    endif()
    if(MLK_WARNINGS_AS_ERRORS)
        target_compile_options(mlk_cxx_flags INTERFACE -Werror)
    endif()
endif()

# Rule 81: release builds are optimized hard.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mlk_cxx_flags INTERFACE $<$<CONFIG:Release>:-O3>)
endif()

function(mlk_apply_flags target)
    target_link_libraries(${target} PUBLIC mlk_cxx_flags)
endfunction()

function(mlk_apply_global_flags)
    # Hook for flag injection order; per-target linking happens via
    # mlk_apply_flags. Kept so the root script reads declaratively.
endfunction()
