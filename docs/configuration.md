# Configuration

LatchLine is configured via `settings.conf` files using simple `KEY=value` syntax.

## Config File Locations

Settings are loaded from the first file found (in order):

1. `{cwd}/.latchline/settings.conf` - Current directory
2. `{project_root}/.latchline/settings.conf` - Git root
3. `~/.latchline/settings.conf` - Global fallback

This allows project-specific overrides while maintaining global defaults.

## Settings Reference

### Core Settings

#### REVIEWER_BACKEND

Which AI to use for reviews.

| Value | Description |
|-------|-------------|
| `codex` | Use OpenAI Codex CLI (default) |
| `claude` | Use Claude Code CLI |
| `both` | Run both and merge findings |

```
REVIEWER_BACKEND=claude
```

#### REVIEWER_BLOCK

How aggressively to gate changes.

| Value | Behavior |
|-------|----------|
| `0` | Log only - reviews run but never block |
| `1` | Block on blockers - halt if critical/high findings with confidence ≥ threshold |
| `2` | Always block - show review and wait for `review:apply` or `review:skip` |

```
REVIEWER_BLOCK=2
```

#### LATCHLINE_LOG_DIR

Base directory for logs, runs, and cache.

```
LATCHLINE_LOG_DIR=/tmp
```

Output structure:
```
$LATCHLINE_LOG_DIR/
├── latchline/
│   ├── runs/{session_id}/prompt-{n}/   # Per-prompt outputs
│   ├── cache/                           # Dependency graph cache
│   └── sessions/                        # Session state
```

### Context Bundle Settings

The context bundle provides surrounding code to help the reviewer understand changes.

#### REVIEWER_CONTEXT

Enable/disable context bundling.

| Value | Behavior |
|-------|----------|
| `0` | Diff-only review (faster, less context) |
| `1` | Include context bundle (default) |

```
REVIEWER_CONTEXT=1
```

#### REVIEWER_CONTEXT_MAX_BYTES

Maximum size of the context bundle in bytes. Increase for large changes or complex dependencies.

```
REVIEWER_CONTEXT_MAX_BYTES=500000
```

#### REVIEWER_CONTEXT_DEPTH

How many levels of imports to follow when building context.

| Value | Behavior |
|-------|----------|
| `0` | Only include directly changed files |
| `1` | Include files imported by changed files |
| `2` | Include two levels of imports (default) |
| `n` | Include n levels |

```
REVIEWER_CONTEXT_DEPTH=2
```

### Review Behavior

#### REVIEWER_CONFIDENCE_THRESHOLD

Minimum confidence score (0-100) for a finding to block.

Findings below this threshold are logged but won't trigger blocking (when `REVIEWER_BLOCK=1`).

```
REVIEWER_CONFIDENCE_THRESHOLD=70
```

Confidence guidelines used by reviewers:
- **90-100**: Certain - verified via docs or obvious from code
- **70-89**: High - strong evidence in diff/context
- **50-69**: Medium - likely but needs verification
- **30-49**: Low - possible issue, uncertain
- **0-29**: Speculative - mentioned for awareness

#### REVIEWER_MULTI_PASS

Enable two-pass review strategy.

| Value | Behavior |
|-------|----------|
| `0` | Single pass (legacy) |
| `1` | Two-pass: smoke then semantic (default) |

```
REVIEWER_MULTI_PASS=1
```

**Pass 1 (Smoke)**: Fast check for obvious issues - syntax errors, clear bugs, missing null checks.

**Pass 2 (Semantic)**: Deep analysis - data flow, invariants, edge cases, test coverage.

If smoke finds high-confidence critical issues, semantic is skipped.

#### REVIEWER_CUSTOM_RULES

Enable custom rules from `rules.md` files.

| Value | Behavior |
|-------|----------|
| `0` | Ignore rules.md files |
| `1` | Load and apply rules (default) |

```
REVIEWER_CUSTOM_RULES=1
```

See [Custom Rules](custom-rules.md) for writing rules.

### Impact Analysis

Cross-file impact analysis detects breaking changes and tracks dependencies.

#### REVIEWER_IMPACT_ANALYSIS

Enable/disable impact analysis.

| Value | Behavior |
|-------|----------|
| `0` | Disabled (default) |
| `1` | Build dependency graph, detect breaking changes |

```
REVIEWER_IMPACT_ANALYSIS=1
```

#### REVIEWER_IMPACT_MAX_DEPENDENTS

Maximum dependent files to report. Prevents noise in heavily-imported modules.

```
REVIEWER_IMPACT_MAX_DEPENDENTS=20
```

#### REVIEWER_IMPACT_DEPTH

Transitive dependency depth.

| Value | Behavior |
|-------|----------|
| `0` | Direct dependents only |
| `1` | Include files that import dependents (default) |
| `n` | n levels of transitive dependents |

```
REVIEWER_IMPACT_DEPTH=1
```

#### REVIEWER_INCLUDE_DEPENDENTS

Add dependent files to the context bundle.

| Value | Behavior |
|-------|----------|
| `0` | Don't include dependents in context (default) |
| `1` | Include files that import changed code |

```
REVIEWER_INCLUDE_DEPENDENTS=1
```

Useful when you want the reviewer to see how callers use the changed code.

## Environment Variables

All settings can also be set via environment variables with the same names:

```bash
export REVIEWER_BACKEND=claude
export REVIEWER_BLOCK=2
```

Environment variables take precedence over config files.

## Example Configurations

### Minimal (log only)

```
REVIEWER_BACKEND=claude
REVIEWER_BLOCK=0
```

### Strict review gate

```
REVIEWER_BACKEND=claude
REVIEWER_BLOCK=2
REVIEWER_CONFIDENCE_THRESHOLD=60
REVIEWER_MULTI_PASS=1
```

### Full analysis

```
REVIEWER_BACKEND=both
REVIEWER_BLOCK=2
REVIEWER_CONTEXT=1
REVIEWER_CONTEXT_DEPTH=3
REVIEWER_IMPACT_ANALYSIS=1
REVIEWER_INCLUDE_DEPENDENTS=1
```
