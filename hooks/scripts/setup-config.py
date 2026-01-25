#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install or update LatchLine config at ~/.latchline/settings.conf"
    )
    parser.add_argument("--backend", choices=["codex", "claude", "both"])
    parser.add_argument("--block", choices=["0", "1", "2"])
    parser.add_argument("--log-dir")
    return parser.parse_args()


def update_config_lines(lines: list[str], overrides: dict[str, str]) -> list[str]:
    seen = set()
    updated = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in overrides:
            updated.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            updated.append(line)
    for key, value in overrides.items():
        if key not in seen:
            updated.append(f"{key}={value}")
    return updated


def main() -> int:
    if any(flag in sys.argv for flag in ("-h", "--help")):
        parse_args()
        return 0

    args = parse_args()

    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    plugin_root = Path(env_root).expanduser() if env_root else None
    if plugin_root is None or not plugin_root.is_dir():
        plugin_root = Path(__file__).resolve().parents[2]

    src = plugin_root / "config" / "reviewer.conf"
    if not src.is_file():
        print(f"Missing template config: {src}", file=sys.stderr)
        return 1

    dest_dir = Path.home() / ".latchline"
    dest = dest_dir / "settings.conf"
    dest_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.backend:
        overrides["REVIEWER_BACKEND"] = args.backend
    if args.block:
        overrides["REVIEWER_BLOCK"] = args.block
    if args.log_dir:
        overrides["LATCHLINE_LOG_DIR"] = args.log_dir

    template_lines = src.read_text(encoding="utf-8").splitlines()
    final_lines = update_config_lines(template_lines, overrides)
    dest.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print(f"Wrote config: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
