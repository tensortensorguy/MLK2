#!/usr/bin/env python3
"""Doc staleness check (Rule 160: stale documentation is a defect).
Verifies every registered pass has a docs/passes/<name>.md entry."""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    pat = re.compile(r'\w+\(\s*symbols,\s*"([a-z_.0-9]+)"')
    missing = []
    seen = set()
    for dirpath, _, files in os.walk(os.path.join(ROOT, "compiler", "src", "passes")):
        for f in files:
            if not f.endswith(".cpp"):
                continue
            text = open(os.path.join(dirpath, f)).read()
            for m in pat.finditer(text):
                seen.add(m.group(1))
    docs_dir = os.path.join(ROOT, "docs", "passes")
    for name in sorted(seen):
        path = os.path.join(docs_dir, name + ".md")
        if not os.path.exists(path):
            missing.append(name)
    if missing:
        print("Missing pass docs:", ", ".join(missing))
        return 1
    print(f"pass docs: complete ({len(seen)} passes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
