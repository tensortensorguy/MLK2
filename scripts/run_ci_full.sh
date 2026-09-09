#!/bin/bash
# Full CI (Rules 42, 43, 153): build matrix + sanitizers + differential +
# fuzz + golden + tool smoke tests.
set -e
cd "$(dirname "$0")/.."
for sanitizer in "" address undefined thread; do
    cfg="build-ci-full${sanitizer:+-$sanitizer}"
    args="-DCMAKE_BUILD_TYPE=Release -DMLK_BUILD_TOOLS=ON"
    if [ -n "$sanitizer" ]; then args="$args -DMLK_SANITIZER=$sanitizer"; fi
    echo "=== configuring $cfg ($args) ==="
    cmake -S . -B "$cfg" -G Ninja $args
    cmake --build "$cfg"
    ctest --test-dir "$cfg" --output-on-failure
done
echo "=== tool smoke tests ==="
./build-ci-full/bin/mlkc list-passes > /dev/null
./build-ci-full/bin/mlkc compile examples/scalar/sin_graph.mlk --tier=1 --emit=ir > /dev/null
./build-ci-full/bin/mlk-verify examples/scalar/sin_graph.mlk
./build-ci-full/bin/mlk-generate-profiles /tmp/mlk-profiles-smoke > /dev/null
echo "CI full: green"
