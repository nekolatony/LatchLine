# Custom Rules

Custom rules let you enforce project-specific coding standards, security policies, and style guidelines. Rules are written in Markdown and passed directly to the AI reviewer.

## How Rules Work

1. LatchLine collects all applicable `rules.md` files
2. Rules are concatenated with section headers
3. The combined rules are injected into the review prompt
4. The reviewer applies rules when analyzing the diff
5. Violations are cited with the rule name in findings

## File Locations

Rules are loaded from these locations (all applicable files are combined):

| Location | Scope | Example |
|----------|-------|---------|
| `~/.latchline/rules.md` | Global - all projects | Personal coding standards |
| `{project}/.latchline/rules.md` | Project-wide | Team conventions |
| `{dir}/.latchline/rules.md` | Directory-specific | Module-specific rules |

### Lookup Example

For a change to `myproject/services/payments/handler.py`:

```
~/.latchline/rules.md                           → Global
myproject/.latchline/rules.md                   → Project
myproject/services/.latchline/rules.md          → Services module
myproject/services/payments/.latchline/rules.md → Payments module
```

All found files are included. More specific rules naturally take priority when they conflict with broader rules.

## Writing Rules

Rules are free-form Markdown. The AI interprets them as review instructions.

### Basic Structure

```markdown
# Project Name Rules

## Category Name

Description of what to check.

**Correct:**
```python
# good example
```

**Incorrect:**
```python
# bad example
```
```

### Tips for Effective Rules

**Be specific**: Vague rules produce inconsistent results.

```markdown
# Bad
- Use good variable names

# Good
- Variable names must be descriptive (min 3 chars, no single letters except loop indices)
- Boolean variables must be prefixed with is_, has_, should_, or can_
```

**Include examples**: Show correct and incorrect patterns.

```markdown
## Error Handling

All API calls must have explicit error handling.

**Correct:**
```python
try:
    response = client.fetch(id)
except ClientError as e:
    logger.error("fetch failed", error=str(e), id=id)
    raise
```

**Incorrect:**
```python
response = client.fetch(id)  # No error handling
```
```

**Explain the why**: Helps the reviewer make judgment calls.

```markdown
## No f-strings in logs

Never use f-strings in structlog messages. Use keyword arguments to enable
proper log aggregation, querying, and lazy evaluation.
```

## Example Rules

### Python Style

```markdown
## Python Style

### Type Hints
- All public functions must have type hints
- Use `X | None` not `Optional[X]`
- Use built-in generics: `list[str]` not `List[str]`

### Imports
- No wildcard imports (`from x import *`)
- Group imports: stdlib, third-party, local (with blank lines)
- No inline imports inside functions
```

### Security Rules

```markdown
## Security

### Sensitive Data
- Never log passwords, tokens, API keys, or PII
- Use parameterized queries, never string interpolation for SQL
- Sanitize all user input before use

### Authentication
- All endpoints except /health must require authentication
- Token validation must happen before any business logic
```

### API Conventions

```markdown
## REST API

### Response Format
All responses must follow this structure:
```json
{
  "data": { ... },
  "error": null | { "code": "...", "message": "..." }
}
```

### Status Codes
- 200: Success with data
- 201: Created
- 400: Client error (validation, bad request)
- 401: Not authenticated
- 403: Not authorized
- 404: Resource not found
- 500: Server error (never expose internals)
```

### Framework-Specific

```markdown
## Flask Conventions

- Return `(data, status_code)` tuples, not `jsonify()`
- Use `abort()` with `http.HTTPStatus` for errors
- All routes must have OpenAPI docstrings
```

### Module-Specific

For `payments/.latchline/rules.md`:

```markdown
## Payment Processing

PCI DSS compliance required in this module:

- Never log card numbers, CVV, or full account numbers
- Mask card numbers in any output (show only last 4 digits)
- All database queries must use parameterized statements
- Encryption required for card data at rest and in transit
- Audit log all payment operations with timestamp and user ID
```

## Disabling Rules

To disable custom rules entirely:

```
# In settings.conf
REVIEWER_CUSTOM_RULES=0
```

To disable rules for a specific directory, create an empty `rules.md`:

```bash
touch myproject/legacy/.latchline/rules.md
```

## Debugging Rules

Check which rules are being loaded by looking at the review prompt in the run output:

```
$LATCHLINE_LOG_DIR/latchline/runs/{session}/prompt-{n}/
```

The context file shows the full prompt including rules sections:

```
=== Global Rules ===
...

=== Project Rules ===
...
```
