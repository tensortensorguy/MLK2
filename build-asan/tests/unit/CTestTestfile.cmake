# CMake generated Testfile for 
# Source directory: /home/z/my-project/mlk2-repo/tests/unit
# Build directory: /home/z/my-project/mlk2-repo/build-asan/tests/unit
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("mlk_unit_core" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_core")
set_tests_properties("mlk_unit_core" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_ir" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_ir")
set_tests_properties("mlk_unit_ir" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_passes_math" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_passes_math")
set_tests_properties("mlk_unit_passes_math" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_passes_analysis" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_passes_analysis")
set_tests_properties("mlk_unit_passes_analysis" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_egraph" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_egraph")
set_tests_properties("mlk_unit_egraph" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_calculus" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_calculus")
set_tests_properties("mlk_unit_calculus" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_tensor" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_tensor")
set_tests_properties("mlk_unit_tensor" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_runtime" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_runtime")
set_tests_properties("mlk_unit_runtime" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_autotune" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_autotune")
set_tests_properties("mlk_unit_autotune" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_superopt" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_superopt")
set_tests_properties("mlk_unit_superopt" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
add_test("mlk_unit_golden" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_unit_golden")
set_tests_properties("mlk_unit_golden" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;6;mlk_add_test;/home/z/my-project/mlk2-repo/tests/unit/CMakeLists.txt;0;")
