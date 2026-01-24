# LatchLine

LatchLine is a global Claude Code/Codex review hook that gates changes per prompt, logs diffs, and asks for explicit approval before applying feedback.

## Contents
- `hooks/ai-review.sh`: The review hook script.
- `config/reviewer.conf`: Configuration options for backend and blocking modes.

## Quick start
1. Copy `hooks/ai-review.sh` into your Claude hooks directory.
2. Copy `config/reviewer.conf` to `~/.claude/reviewer.conf` (or adjust path in the hook).
3. Ensure the hook is registered in `~/.claude/settings.json`.

## Configuration
See `config/reviewer.conf` for defaults and supported values.
