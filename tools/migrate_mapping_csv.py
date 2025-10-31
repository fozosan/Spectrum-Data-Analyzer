#!/usr/bin/env python3
"""Upgrade legacy mapping CSV files to the new 'ordering' column."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


CANDIDATE_COLUMNS = [
    "ordering",
    "order",
    "x_mapping",
    "x",
    "X Mapping",
    "X",
]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: migrate_mapping_csv.py <input.csv> [output.csv]")
        return 1

    src = Path(argv[1])
    if not src.exists():
        print(f"Input file not found: {src}")
        return 1

    dst = Path(argv[2]) if len(argv) > 2 else src

    df = pd.read_csv(src)
    for cand in CANDIDATE_COLUMNS:
        if cand in df.columns:
            src_col = cand
            break
    else:
        print("No ordering/x/x_mapping column found")
        return 1

    df = df.rename(columns={src_col: "ordering"})
    df.to_csv(dst, index=False)
    print(f"Wrote with 'ordering' column: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
