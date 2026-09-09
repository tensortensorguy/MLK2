#!/bin/bash
# Local pre-commit hook (Rule 66: must complete in < 2 seconds).
set -e
START=$(date +%s%N)
bash "$(dirname "$0")/lint.sh"
git diff --check || exit 1
END=$(date +%s%N)
MS=$(( (END - START) / 1000000 ))
echo "pre-commit: ${MS}ms (budget 2000ms — Rule 66)"
[ "$MS" -lt 2000 ]
