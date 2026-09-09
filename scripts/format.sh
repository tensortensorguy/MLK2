#!/bin/bash
# MLK+ formatting (clang-format when available; Rule 66).
set -e
cd "$(dirname "$0")/.."
if command -v clang-format >/dev/null 2>&1; then
    git ls-files '*.h' '*.hpp' '*.cpp' | xargs clang-format -i
    echo "formatted"
else
    echo "clang-format not found; skipping (non-fatal)" >&2
fi
