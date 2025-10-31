#!/usr/bin/env python3
"""Upgrade saved session files to use the 'ordering' map."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: migrate_session_ordering.py <input.json> [output.json]")
        return 1

    src = Path(argv[1])
    if not src.exists():
        print(f"Input file not found: {src}")
        return 1

    dst = Path(argv[2]) if len(argv) > 2 else src

    with src.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "ordering" not in data and "x_mapping" not in data:
        print("Session does not contain 'ordering' or legacy 'x_mapping' entries; nothing to migrate.")
        return 1

    legacy = data.pop("x_mapping", None)
    if "ordering" in data and data["ordering"]:
        # Ensure ordering keys are strings and values are floats
        ordering = {str(k): float(v) for k, v in dict(data["ordering"]).items()}
    elif legacy:
        ordering = {str(k): float(v) for k, v in dict(legacy).items()}
        data["ordering"] = ordering
    else:
        data["ordering"] = {}
        ordering = {}

    if dst == src:
        # writing back to same file; ensure atomic by writing temp? For simplicity, overwrite
        with dst.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        with dst.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Migrated session saved to {dst} with {len(ordering)} ordering entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
