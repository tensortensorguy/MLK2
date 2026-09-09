# CMake generated Testfile for 
# Source directory: /home/z/my-project/mlk2-repo/tests/fallback
# Build directory: /home/z/my-project/mlk2-repo/build-asan/tests/fallback
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("mlk_fallback" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_fallback")
set_tests_properties("mlk_fallback" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests/fallback" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/fallback/CMakeLists.txt;1;mlk_add_test;/home/z/my-project/mlk2-repo/tests/fallback/CMakeLists.txt;0;")
