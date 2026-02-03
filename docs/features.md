# Features

## Multi-Pass Review

LatchLine uses a two-pass review strategy to balance speed and thoroughness.

### Pass 1: Smoke Check

A fast scan for obvious issues:
- Syntax errors
- Clear bugs (null dereference, off-by-one)
- Missing imports
- Obvious security issues

If smoke finds high-confidence critical issues, the semantic pass is skipped to save time.

### Pass 2: Semantic Analysis

Deep analysis when smoke passes:
- Data flow tracing
- Invariant verification
- Edge case detection
- Error handling completeness
- Test coverage assessment

### Configuration

```
REVIEWER_MULTI_PASS=1  # Enable (default)
REVIEWER_MULTI_PASS=0  # Single pass only
```

## Confidence Scoring

Every finding includes a confidence score (0-100) indicating how certain the reviewer is.

### Score Ranges

| Range | Meaning | Example |
|-------|---------|---------|
| 90-100 | Certain | Verified via docs, obvious from code |
| 70-89 | High | Strong evidence in diff/context |
| 50-69 | Medium | Likely but needs verification |
| 30-49 | Low | Possible issue, uncertain |
| 0-29 | Speculative | Mentioned for awareness |

### Blocking Threshold

Only findings with confidence ≥ threshold cause blocking (when `REVIEWER_BLOCK=1`):

```
REVIEWER_CONFIDENCE_THRESHOLD=70  # Default
```

Lower the threshold to be more strict, raise it to reduce false positives.

## Diff Highlighting

Findings are mapped to specific lines in the diff for easy navigation.

### Output Format

The `review.diff.md` file shows an annotated diff:

```diff
+def process(data):
+    result = data["key"]  # [F001] Missing null check
+    return result
```

### Severity Colors

In terminal output:
- **Critical**: Red background
- **High**: Red text
- **Medium**: Yellow text
- **Low**: Blue text
- **Info**: Gray text

## Context Bundling

LatchLine builds a context bundle to help the reviewer understand changes in context.

### What's Included

1. **Changed files** (full content)
2. **Imported modules** (based on `REVIEWER_CONTEXT_DEPTH`)
3. **Related config files** (pyproject.toml, package.json, etc.)
4. **Type definitions** referenced in changes

### Controlling Context Size

```
REVIEWER_CONTEXT_MAX_BYTES=500000  # Max bundle size
REVIEWER_CONTEXT_DEPTH=2          # Import depth
```

Increase for complex changes, decrease for faster reviews.

## Cross-File Impact Analysis

Detects how changes affect other parts of the codebase.

### Dependency Graph

LatchLine builds and caches a reverse dependency graph:
- Which files import what
- Updated incrementally based on file modification times
- Cached in `$LATCHLINE_LOG_DIR/cache/`

### Breaking Change Detection

Compares before/after snapshots to detect:

| Change | Severity | Example |
|--------|----------|---------|
| Removed function | High | `def helper()` deleted |
| Removed class | High | `class User` deleted |
| Reduced arguments | High | `def foo(a, b)` → `def foo(a)` |
| Added required argument | Medium | `def foo(a)` → `def foo(a, b)` |

### Dependent Tracking

Reports files that may need updates:

```
impact.md:
  Changed: src/utils.py
  Dependents (12 files):
    - src/api/users.py (imports: helper, process)
    - src/api/orders.py (imports: helper)
    ...
```

### Configuration

```
REVIEWER_IMPACT_ANALYSIS=1       # Enable
REVIEWER_IMPACT_MAX_DEPENDENTS=20  # Limit reported files
REVIEWER_IMPACT_DEPTH=1          # Transitive depth
REVIEWER_INCLUDE_DEPENDENTS=1    # Add to context bundle
```

## Review Gate

The review gate controls when Claude can proceed after reviews.

### Gate Modes

| Mode | Behavior |
|------|----------|
| `REVIEWER_BLOCK=0` | Log only, never block |
| `REVIEWER_BLOCK=1` | Block if blockers found |
| `REVIEWER_BLOCK=2` | Always block, require explicit action |

### User Actions

When blocked, respond with:
- `review:apply` - Accept feedback and continue
- `review:skip` - Skip reviews for rest of session
- `review:enable` / `review:resume` - Re-enable after skip

### Session Persistence

Skip state persists for the Claude Code session. Starting a new session resets to default behavior.

## Output Files

Each review creates files in `$LATCHLINE_LOG_DIR/latchline/runs/{session}/prompt-{n}/`:

| File | Content |
|------|---------|
| `diff.patch` | Unified diff of changes |
| `review.structured.json` | Machine-readable findings |
| `review.diff.md` | Annotated diff with findings |
| `review.log.md` | Human-readable summary |
| `context.txt` | Context bundle sent to reviewer |
| `impact.json` | Impact analysis (if enabled) |
| `impact.md` | Human-readable impact (if enabled) |

### Structured Output Schema

```json
{
  "findings": [
    {
      "id": "F001",
      "severity": "medium",
      "category": "correctness",
      "confidence": 85,
      "title": "Missing null check",
      "description": "data['key'] may raise KeyError",
      "file_path": "src/handler.py",
      "line_start": 42,
      "line_end": 42,
      "suggested_fix": "Use data.get('key') or check key exists"
    }
  ],
  "blockers_summary": "1 medium finding",
  "notes_summary": "Consider adding type hints"
}
```

### Finding Categories

- `correctness` - Logic errors, bugs
- `security` - Vulnerabilities, unsafe patterns
- `regression` - Breaking changes
- `missing_test` - Untested code paths
- `style` - Convention violations
- `performance` - Inefficient code
- `impact` - Cross-file effects (from impact analysis)
