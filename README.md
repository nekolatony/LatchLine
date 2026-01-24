# LatchLine

<img src="assets/latchline-logo.svg" width="120" alt="LatchLine logo" />

LatchLine is a global Claude Code/Codex review hook that gates changes per prompt, logs diffs, and asks for explicit approval before applying feedback.

## Why “LatchLine”
LatchLine is named for a latch: it holds changes at the end of a prompt, shows the review, and only proceeds when you explicitly release it (apply/skip).

## Prerequisites
- Python 3.10+
- uv
- Claude Code with hook support enabled

## How to set up
1. Clone this repo and note the absolute path.
2. Copy `config/reviewer.conf` to `~/.latchline/settings.conf` (global) or to `.latchline/settings.conf` in your project root/current directory.
3. Register the hook command in `~/.claude/settings.json` using the repo path (example below).
4. Restart Claude Code so the hook config is reloaded.

## Hook setup (Claude Code)
LatchLine runs as a Claude Code hook command and must be registered for these events:
- `UserPromptSubmit`
- `PreToolUse` (with `Edit|Write`)
- `PostToolUse` (with `AskUserQuestion`)
- `Stop`

Why each hook is needed:

| Hook event | When it fires | Why it matters |
| --- | --- | --- |
| `UserPromptSubmit` | At the start of a prompt | Starts a new review session so edits are grouped correctly. |
| `PreToolUse` (`Edit|Write`) | Right before edits | Captures a “before” snapshot so we can diff after edits. |
| `PostToolUse` (`AskUserQuestion`) | After the review gate prompt | Records your apply/skip response so the run log is complete. |
| `Stop` | When the prompt finishes | Finalizes the diff and runs the reviewer. |

Example `~/.claude/settings.json`:
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

## Configuration
See `config/reviewer.conf` for defaults and supported values.

| Config | Default | Values | Explanation |
| --- | --- | --- | --- |
| `REVIEWER_BACKEND` | `codex` | `codex`, `claude`, `both` | Which reviewer to run. |
| `REVIEWER_BLOCK` | `0` | `0`, `1`, `2` | How hard to gate the session (`0` = log only, `1` = block on blockers, `2` = always block and ask for apply/skip). |
| `LATCHLINE_LOG_DIR` | `/tmp` | Any path | Base directory for logs/state/runs (created if missing, falls back to `/tmp` on failure). |

## Development
Run tests with `uv run --group dev pytest` or `just test`.
Format with `just format` and lint with `just lint`.
Run the full Python version matrix with `just test-all`.
