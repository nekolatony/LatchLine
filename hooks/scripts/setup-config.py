#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    plugin_root = Path(os.environ.get("CLAUDE_PLUGIN_ROOT", ""))
    if not plugin_root.is_dir():
        plugin_root = Path(__file__).resolve().parents[2]

    src = plugin_root / "config" / "reviewer.conf"
    if not src.is_file():
        print(f"Missing template config: {src}", file=sys.stderr)
        return 1

    dest_dir = Path.home() / ".latchline"
    dest = dest_dir / "settings.conf"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Config already exists: {dest}")
        return 0

    shutil.copyfile(src, dest)
    print(f"Wrote config: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
