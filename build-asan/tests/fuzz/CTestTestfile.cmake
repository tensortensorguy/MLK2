# CMake generated Testfile for 
# Source directory: /home/z/my-project/mlk2-repo/tests/fuzz
# Build directory: /home/z/my-project/mlk2-repo/build-asan/tests/fuzz
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("mlk_fuzz" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_fuzz")
set_tests_properties("mlk_fuzz" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests/fuzz" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/fuzz/CMakeLists.txt;1;mlk_add_test;/home/z/my-project/mlk2-repo/tests/fuzz/CMakeLists.txt;0;")
