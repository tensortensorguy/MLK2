# Optional third-party dependencies (layout spec §third_party).
#
# Rule 160 requires hermetic builds and pinned dependencies. Every dependency
# here is OFF by default so that the default build is fully hermetic and
# requires no network access. When an optional dependency is enabled it must
# be pinned via FETCHCONTENT_SOURCEDIR_<NAME> or a vendored copy.

option(MLK_USE_GOOGLETEST "Fetch GoogleTest (off = hermetic test harness)" OFF)
option(MLK_USE_JSON       "Fetch nlohmann_json (off = built-in minimal JSON)" OFF)
option(MLK_USE_LLVM       "Use system LLVM for the optional llvm backend" OFF)
option(MLK_USE_Z3         "Use system Z3 for formal equivalence checking" OFF)

set(MLK_THIRD_PARTY_TARGETS "" CACHE INTERNAL "")

if(MLK_USE_GOOGLETEST)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        URL_HASH SHA256=7e497931d8aace6eaf07dfe34a7a5d51e31f7037d2f6e9db36531db9b91e6b80)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    list(APPEND MLK_THIRD_PARTY_TARGETS GTest::gtest GTest::gtest_main)
endif()

if(MLK_USE_JSON)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
    FetchContent_MakeAvailable(nlohmann_json)
    list(APPEND MLK_THIRD_PARTY_TARGETS nlohmann_json::nlohmann_json)
endif()

if(MLK_USE_LLVM)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "MLK+ LLVM backend enabled: ${LLVM_PACKAGE_VERSION}")
endif()

if(MLK_USE_Z3)
    find_package(Z3 REQUIRED)
    message(STATUS "MLK+ Z3 proof backend enabled")
endif()

set(MLK_THIRD_PARTY_TARGETS "${MLK_THIRD_PARTY_TARGETS}" CACHE INTERNAL "")
