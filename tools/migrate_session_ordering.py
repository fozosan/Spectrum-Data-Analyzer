#!/usr/bin/env python3
"""Upgrade saved session files to use the 'ordering' map."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: migrate_session_ordering.py <input.json> <output.json>")
        return 1

    src = Path(argv[1])
    if not src.exists():
        print(f"Input file not found: {src}")
        return 1

    dst = Path(argv[2])

    with src.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    legacy = data.pop("x_mapping", None)
    ordering = data.get("ordering")

    if ordering is None and legacy is None:
        print("No 'ordering' or legacy 'x_mapping' data found; nothing to migrate.")
        return 1

    if ordering is not None:
        ordering_map = {str(k): float(v) for k, v in dict(ordering).items()}
    else:
        ordering_map = {str(k): float(v) for k, v in dict(legacy).items()}
        data["ordering"] = ordering_map

    with dst.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if ordering is not None and legacy is None:
        print(f"Re-saved session with existing ordering to {dst}.")
    elif ordering is not None:
        print(f"Migrated session with {len(ordering_map)} ordering entries (overwriting legacy x_mapping) to {dst}.")
    else:
        print(f"Migrated legacy x_mapping → ordering with {len(ordering_map)} entries to {dst}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
