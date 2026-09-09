#!/bin/bash
# MLK+ anti-slop lint (Rule 84). Failures block merge.
set -e
cd "$(dirname "$0")/.."
fail=0
# Banned containers in hot IR code (Rules 9, 17, 18)
if grep -rn "std::unordered_map\|std::unordered_set" compiler/include/mlk compiler/src --include='*.h' --include='*.cpp' | grep -v '^\s*//' | grep -v ': *//' | grep -v symbol_table.h | grep -v hash_map.h; then
    echo "LINT FAIL: banned hash containers in IR/passes (Rule 17)"; fail=1
fi
if grep -rn "std::vector<bool>" compiler/ runtime/ | grep -v ': *//'; then
    echo "LINT FAIL: std::vector<bool> banned (Rule 18)"; fail=1
fi
if grep -rn "dynamic_cast\|typeid" compiler/ runtime/ autotuner/ superopt/ backends/; then
    echo "LINT FAIL: RTTI constructs banned (Rule 8)"; fail=1
fi
if grep -rn "throw " compiler/ runtime/ autotuner/ superopt/ backends/ --include='*.cpp' --include='*.h' | grep -v "Test\|test"; then
    echo "LINT FAIL: throw banned (Rule 6)"; fail=1
fi
# std::string in IR data structures (Rule 16: interned symbols only)
if grep -rn "std::string" compiler/include/mlk/ir/ | grep -v "// cold path" | grep -v graph_printer | grep -v graph_json; then
    echo "LINT WARN: std::string inside ir/ headers — verify cold-path only" >&2
fi
# Hidden globals: getenv in hot paths (Rule 84 checklist)
if grep -rn "getenv" compiler/ runtime/ autotuner/ superopt/ backends/; then
    echo "LINT FAIL: getenv banned in compiler/runtime code"; fail=1
fi
if [ "$fail" -ne 0 ]; then exit 1; fi
echo "lint: clean"
