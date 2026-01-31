# LatchLine

LatchLine is a global Claude Code/Codex review hook that gates changes per prompt, logs diffs, and asks for explicit approval before applying feedback.

## Why “LatchLine”
LatchLine is named for a latch: it holds changes at the end of a prompt, shows the review, and only proceeds when you explicitly release it (apply/skip).

## Prerequisites
- Python 3.10+
- uv
- Claude Code with hook support enabled
- Optional: Codex CLI (for `REVIEWER_BACKEND=codex`)

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
| `REVIEWER_CONTEXT` | `1` | `0`, `1` | Whether to build a context bundle for diff-first reviews. |
| `REVIEWER_CONTEXT_MAX_BYTES` | `500000` | Any integer | Max size of the context bundle in bytes (raise to include more context). |
| `REVIEWER_CONTEXT_DEPTH` | `2` | Any integer | Dependency expansion depth for local imports. |
| `REVIEWER_CONFIDENCE_THRESHOLD` | `70` | `0-100` | Minimum confidence score to block on a finding. Findings below threshold are logged but won't block. |
| `REVIEWER_MULTI_PASS` | `1` | `0`, `1` | Enable two-pass review: smoke check (fast, obvious issues) then semantic pass (deep analysis). |
| `LATCHLINE_LOG_DIR` | `/tmp` | Any path | Base directory for logs/state/runs (created if missing, falls back to `/tmp` on failure). |

## Output Files

Each review run creates files in `$LATCHLINE_LOG_DIR/runs/{session_id}/prompt-{id}/`:

| File | Description |
| --- | --- |
| `diff.patch` | Unified diff of all changes |
| `review.structured.json` | Structured findings with confidence scores, line mappings |
| `review.diff.md` | Annotated diff with findings mapped to specific lines |
| `review.log.md` | Human-readable review log |
| `context.txt` | Context bundle sent to reviewer |

## Features

### Diff Highlighting
Findings are automatically mapped to the specific diff lines that triggered them. The annotated diff shows:
- Color-coded severity (critical=red bg, high=red, medium=yellow, low=blue)
- Inline annotations with finding ID and title
- Markdown format in `review.diff.md` for easy reading

### Multi-Pass Review
When `REVIEWER_MULTI_PASS=1` (default), reviews run in two passes:
1. **Smoke pass**: Fast check for obvious issues (syntax errors, clear bugs)
2. **Semantic pass**: Deep analysis (data flow, edge cases, test coverage)

If the smoke pass finds high-confidence critical issues, the semantic pass is skipped.

### Confidence Scoring
Each finding has a confidence score (0-100). Only findings with `confidence >= REVIEWER_CONFIDENCE_THRESHOLD` will block. This reduces false positive friction.

## Development
Run tests with `uv run --group dev pytest` or `just test`.
Format with `just format` and lint with `just lint`.
Run the full Python version matrix with `just test-all`.
