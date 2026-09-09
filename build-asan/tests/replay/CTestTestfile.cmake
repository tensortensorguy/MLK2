# CMake generated Testfile for 
# Source directory: /home/z/my-project/mlk2-repo/tests/replay
# Build directory: /home/z/my-project/mlk2-repo/build-asan/tests/replay
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("mlk_replay" "/home/z/my-project/mlk2-repo/build-asan/bin/mlk_replay")
set_tests_properties("mlk_replay" PROPERTIES  ENVIRONMENT "MLK_TEST_DIR=/home/z/my-project/mlk2-repo/tests" _BACKTRACE_TRIPLES "/home/z/my-project/mlk2-repo/tests/CMakeLists.txt;10;add_test;/home/z/my-project/mlk2-repo/tests/replay/CMakeLists.txt;1;mlk_add_test;/home/z/my-project/mlk2-repo/tests/replay/CMakeLists.txt;0;")
