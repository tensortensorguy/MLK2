#!/bin/bash
# Quick CI: configure + build + unit tests.
set -e
cd "$(dirname "$0")/.."
cmake -S . -B build-ci -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-ci
ctest --test-dir build-ci --output-on-failure
