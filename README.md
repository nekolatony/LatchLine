# LatchLine

LatchLine is a Claude Code review hook that gates changes, logs diffs, and requires explicit approval before proceeding.

Named for a latch: it holds changes at the end of a prompt, shows the review, and only releases when you explicitly approve (apply/skip).

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- Claude Code with hook support
- Optional: Codex CLI (for `REVIEWER_BACKEND=codex`)

## Quick Start

1. Clone and note the path:
   ```bash
   git clone https://github.com/nekolatony/LatchLine.git
   ```

2. Copy config:
   ```bash
   mkdir -p ~/.latchline
   cp config/reviewer.conf ~/.latchline/settings.conf
   ```

3. Add hooks to `~/.claude/settings.json`:
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

4. Restart Claude Code.

## Usage

After any edit, LatchLine shows a review. Respond with:
- `review:apply` - Accept and continue
- `review:skip` - Skip reviews for this session
- `review:enable` - Re-enable after skip

## Documentation

- [Configuration](docs/configuration.md) - All settings explained
- [Custom Rules](docs/custom-rules.md) - Writing project-specific rules
- [Features](docs/features.md) - Multi-pass review, impact analysis, confidence scoring

## Key Features

- **Multi-pass review**: Fast smoke check, then deep semantic analysis
- **Confidence scoring**: Findings rated 0-100, configurable threshold for blocking
- **Context bundling**: Includes surrounding code for better review accuracy
- **Custom rules**: Define rules via `rules.md` at global, project, or directory level
- **Cross-file impact**: Detects breaking changes and tracks dependents
- **Diff highlighting**: Findings mapped to specific lines

## Development

```bash
uv run --group dev pytest    # Run tests
just test                    # Run tests (alternative)
just format                  # Format code
just lint                    # Lint code
just test-all                # Full Python version matrix
```
