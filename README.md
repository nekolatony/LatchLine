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
| `REVIEWER_IMPACT_ANALYSIS` | `0` | `0`, `1` | Enable cross-file impact analysis: builds reverse dependency graph, detects breaking changes. |
| `REVIEWER_IMPACT_MAX_DEPENDENTS` | `20` | Any integer | Maximum number of dependent files to report. |
| `REVIEWER_IMPACT_DEPTH` | `1` | `0+` | Transitive dependency depth (`0` = direct only, `1+` = include transitive). |
| `REVIEWER_INCLUDE_DEPENDENTS` | `0` | `0`, `1` | Add dependent files to context bundle for reviewer. |
| `REVIEWER_CUSTOM_RULES` | `1` | `0`, `1` | Enable custom rules from `rules.md` files. |
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
| `impact.json` | Cross-file impact report (when enabled) |
| `impact.md` | Human-readable impact summary (when enabled) |

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

### Cross-File Impact Analysis
When `REVIEWER_IMPACT_ANALYSIS=1`, LatchLine analyzes how changes affect other files:

- **Reverse dependency graph**: Builds a cached graph of which files import what, so it can quickly find files that depend on your changes.
- **Breaking change detection**: Compares before/after snapshots to detect:
  - Removed functions or classes
  - Reduced function arguments (breaks callers)
  - Added required arguments
- **Dependent file tracking**: Reports which files import the changed code and may need updates.

Impact findings are added to the review with category `impact` and include:
- High-severity alerts for breaking changes
- Medium-severity warnings when many files depend on changed code

Output files:
- `impact.json`: Structured report with changed symbols, dependents, and breaking changes
- `impact.md`: Human-readable summary

The dependency graph is cached in `$LATCHLINE_LOG_DIR/cache/` and incrementally updated based on file modification times.

### Custom Rules
Define custom review rules via `rules.md` files. Rules are free-form markdown instructions passed directly to the AI reviewer.

**File locations** (checked in order, all applicable rules are included):
1. `~/.latchline/rules.md` - Global rules (apply everywhere)
2. `{project_root}/.latchline/rules.md` - Project rules
3. `{dir}/.latchline/rules.md` - Directory rules (for each directory from project root toward changed files)

Rules cascade with more specific rules taking priority. The AI naturally understands that directory-level rules override project-level rules which override global rules.

**Example `~/.latchline/rules.md`:**
```markdown
# My Global Rules

- Always use type hints in Python
- Prefer composition over inheritance
- No print statements, use logging
```

**Example `project/.latchline/rules.md`:**
```markdown
# Project Rules

This is a FastAPI project:
- Use Pydantic models for request/response
- All endpoints need OpenAPI descriptions
```

**Example `project/services/payments/.latchline/rules.md`:**
```markdown
# Payment Service Rules

PCI compliance required:
- Never log card numbers or CVV
- Use parameterized queries only
```

All applicable rules are concatenated with section headers and injected into the review prompt.

## Development
Run tests with `uv run --group dev pytest` or `just test`.
Format with `just format` and lint with `just lint`.
Run the full Python version matrix with `just test-all`.
