#!/usr/bin/env python3
"""
Fail if any Python source contains a PEP 604-style `| None` union in type annotations.
This helps keep the codebase Python 3.9–compatible (use Optional[...] instead).
"""
from __future__ import annotations

import os
import re
import sys
from typing import List

# naive but effective: look for annotation contexts with a pipe to None
PATTERNS = [
    re.compile(r":\s*[^#\n]*\|\s*None\b"),   # e.g. def foo(x: Optional[int])
    re.compile(r"->\s*[^#\n]*\|\s*None\b"),  # e.g. def foo(...) -> Optional[int]
]

def scan_file(path: str) -> List[str]:
    hits: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                # skip comments quickly
                if line.lstrip().startswith("#"):
                    continue
                for pat in PATTERNS:
                    if pat.search(line):
                        hits.append(f"{path}:{i}: {line.rstrip()}")
    except Exception as exc:
        hits.append(f"{path}: error reading file: {exc}")
    return hits


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    failures: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        # skip common virtualenv/ci/build folders
        if any(skip in dirpath for skip in (".git", ".venv", "venv", "build", "dist", ".mypy_cache", ".pytest_cache")):
            continue
        for name in filenames:
            if not name.endswith(".py"):
                continue
            failures.extend(scan_file(os.path.join(dirpath, name)))
    if failures:
        print("Found PEP 604 `| None` unions in type annotations. Please use Optional[T] for Python 3.9 compatibility.\n", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
