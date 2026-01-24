# LatchLine

LatchLine is a global Claude Code/Codex review hook that gates changes per prompt, logs diffs, and asks for explicit approval before applying feedback.

## Contents
- `src/latchline/cli.py`: The review hook entrypoint.
- `hooks/ai-review.py`: Thin wrapper script (optional).
- `config/reviewer.conf`: Configuration options for backend and blocking modes.

## Quick start
1. Install uv if you don't already have it.
2. Copy `config/reviewer.conf` to `~/.latchline/settings.conf` (global) or to `.latchline/settings.conf` in your project root/current directory.
3. Register the hook command in `~/.claude/settings.json` (see below).

## Hook setup (Claude Code)
LatchLine is a Claude Code hook command. It relies on the same hook events as your current `~/.claude/hooks/ai-review.sh` setup:
- `UserPromptSubmit` to start a new review session.
- `PreToolUse` for `Edit|Write` to snapshot files before changes.
- `PostToolUse` for `AskUserQuestion` to record gate answers.
- `Stop` to compute the diff and run the reviewer.

Why each hook is needed:

| Hook event | Why it matters |
| --- | --- |
| `UserPromptSubmit` | Starts a new prompt session so later edits are grouped correctly. |
| `PreToolUse` (`Edit|Write`) | Captures a “before” snapshot of files so we can diff after edits. |
| `PostToolUse` (`AskUserQuestion`) | Captures your response to the review gate (apply/skip) so the run log is complete. |
| `Stop` | Finalizes the diff and runs the reviewer when the prompt finishes. |

Replace the existing command in `~/.claude/settings.json` with a uv command that points at this repo:
```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "uv run --project /path/to/LatchLine latchline" }] }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "uv run --project /path/to/LatchLine latchline" }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "AskUserQuestion",
        "hooks": [{ "type": "command", "command": "uv run --project /path/to/LatchLine latchline" }]
      }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "uv run --project /path/to/LatchLine latchline" }] }
    ]
  }
}
```
Optional: you can still point the hook at `hooks/ai-review.py`, but `uv run ... latchline` is the intended path.

## Configuration
See `config/reviewer.conf` for defaults and supported values.
`LATCHLINE_LOG_DIR` controls where runs/logs are written (default: `/tmp`).

## Development
Run tests with `uv run --group dev pytest` or `just test`.
Format with `just format` and lint with `just lint`.
Run the full Python version matrix with `just test-all`.
