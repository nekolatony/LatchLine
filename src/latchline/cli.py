from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Finding:
    id: str
    severity: str  # critical|high|medium|low|info
    category: str  # correctness|security|regression|missing_test|style|performance
    confidence: int  # 0-100
    title: str
    description: str
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    suggested_fix: str | None = None
    pass_number: int = 1


@dataclass
class ReviewResult:
    version: str = "1.0"
    timestamp: str = ""
    backend: str = ""
    pass_number: int = 1
    pass_type: str = "smoke"  # smoke|semantic
    findings: list[Finding] = field(default_factory=list)
    blockers_summary: str = ""
    notes_summary: str = ""


def read_json_stdin() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def json_get(data: dict[str, Any], path: str) -> str:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, "")
        else:
            cur = ""
            break
    if isinstance(cur, (dict, list)):
        return ""
    if cur is None:
        return ""
    return str(cur)


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    except FileNotFoundError:
        return []


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def append_text(path: str, content: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_config(path: str) -> dict[str, str]:
    config: dict[str, str] = {}
    if not os.path.isfile(path):
        return config
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            if key:
                config[key] = value
    return config


def ensure_directory(path: str) -> str | None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path if os.path.isdir(path) else None


def resolve_log_dir(config: dict[str, str]) -> str:
    log_dir = config.get("LATCHLINE_LOG_DIR") or os.environ.get("LATCHLINE_LOG_DIR") or "/tmp"
    resolved = ensure_directory(log_dir)
    if resolved:
        return resolved
    fallback = ensure_directory("/tmp")
    return fallback or "/tmp"


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_path(path: str, cwd_path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(cwd_path, expanded))


def is_within_root(path: str, root: str) -> bool:
    if not root:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), root]) == root
    except ValueError:
        return False


def display_path(path: str, project_root: str) -> str:
    if project_root and is_within_root(path, project_root):
        return os.path.relpath(path, project_root)
    return path


def is_text_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"\x00" not in chunk
    except OSError:
        return False


def read_text_limited(path: str, max_bytes: int) -> tuple[str, bool]:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
    except OSError:
        return "", False
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        git_path = os.path.join(cur, ".git")
        if os.path.isdir(git_path) or os.path.isfile(git_path):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return ""
        cur = parent


def resolve_config_path(cwd: str) -> str:
    config_name = os.path.join(".latchline", "settings.conf")
    candidates: list[str] = []
    if cwd:
        candidates.append(os.path.join(cwd, config_name))
    project_root = find_project_root(cwd) if cwd else ""
    if project_root and project_root != cwd:
        candidates.append(os.path.join(project_root, config_name))
    candidates.append(os.path.expanduser(os.path.join("~", config_name)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[-1]


ROOT_ONLY_PATTERNS = [
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    ".gitlab-ci.yml",
    ".circleci/config.yml",
    "azure-pipelines.yml",
]

COMMON_CONTEXT_PATTERNS = [
    "Dockerfile",
    "Dockerfile.*",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "compose*.yml",
    "compose*.yaml",
    "Makefile",
    "Justfile",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
    "Gemfile",
    "Gemfile.lock",
    "pyproject.toml",
    "requirements*.txt",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "setup.cfg",
    "setup.py",
    ".env",
    ".env.*",
    ".env.example",
    ".env.sample",
    ".env.template",
    ".tool-versions",
    ".python-version",
    ".nvmrc",
    ".ruby-version",
]

INFRA_DIRS = ("infra", "terraform", "k8s", "kubernetes", "helm", "charts")


def list_infra_files(project_root: str) -> list[str]:
    if not project_root:
        return []
    patterns = ROOT_ONLY_PATTERNS + COMMON_CONTEXT_PATTERNS
    matches: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(project_root, pattern)):
            if os.path.isfile(path):
                matches.add(os.path.abspath(path))

    for dirname in INFRA_DIRS:
        base = os.path.join(project_root, dirname)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                if name.endswith((".yml", ".yaml", ".json", ".tf", ".tfvars")):
                    matches.add(os.path.join(root, name))

    return sorted(matches)


def iter_dirs_to_root(start_dir: str, root: str) -> list[str]:
    paths: list[str] = []
    cur = os.path.abspath(start_dir)
    root_path = os.path.abspath(root)
    while True:
        if not is_within_root(cur, root_path):
            break
        paths.append(cur)
        if cur == root_path:
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return paths


def list_adjacent_context_files(
    changed_files: list[str], project_root: str
) -> list[str]:
    if not project_root:
        return []
    matches: set[str] = set()
    for path in changed_files:
        if not path:
            continue
        file_dir = os.path.dirname(path)
        for directory in iter_dirs_to_root(file_dir, project_root):
            for pattern in COMMON_CONTEXT_PATTERNS:
                for match in glob.glob(os.path.join(directory, pattern)):
                    if os.path.isfile(match):
                        matches.add(os.path.abspath(match))
    return sorted(matches)


def list_test_files(project_root: str, changed_files: list[str]) -> list[str]:
    if not project_root:
        return []
    base_names = {
        os.path.splitext(os.path.basename(path))[0].lower()
        for path in changed_files
        if path
    }
    if not base_names:
        return []
    test_roots = [os.path.join(project_root, "tests"), os.path.join(project_root, "test")]
    test_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
    matches: set[str] = set()
    for test_root in test_roots:
        if not os.path.isdir(test_root):
            continue
        for root, _, files in os.walk(test_root):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in test_exts:
                    continue
                stem = os.path.splitext(name)[0].lower()
                if any(base in stem for base in base_names):
                    matches.add(os.path.join(root, name))
    return sorted(matches)


def resolve_python_module(
    module: str, level: int, file_dir: str, project_root: str
) -> list[str]:
    if level > 0:
        base_dir = file_dir
        for _ in range(max(level - 1, 0)):
            base_dir = os.path.dirname(base_dir)
    else:
        base_dir = project_root
    module_path = module.replace(".", "/") if module else ""
    candidates: list[str] = []
    if module_path:
        base = os.path.join(base_dir, module_path)
        candidates.append(f"{base}.py")
        candidates.append(os.path.join(base, "__init__.py"))
    else:
        candidates.append(os.path.join(base_dir, "__init__.py"))
    return [c for c in candidates if os.path.isfile(c) and is_within_root(c, project_root)]


def resolve_python_imports(path: str, project_root: str) -> list[str]:
    if not project_root:
        return []
    text = read_text(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    file_dir = os.path.dirname(path)
    results: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.update(
                    resolve_python_module(alias.name, 0, file_dir, project_root)
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            results.update(resolve_python_module(module, level, file_dir, project_root))
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                results.update(resolve_python_module(name, level, file_dir, project_root))
    return sorted(results)


JS_IMPORT_RE = re.compile(
    r"(?:import\s+(?:.+?\s+from\s+)?|export\s+(?:.+?\s+from\s+)?|import\s*\(|require\s*\()"
    r"\s*['\"]([^'\"]+)['\"]"
)


def resolve_js_imports(path: str, project_root: str) -> list[str]:
    if not project_root:
        return []
    text = read_text(path)
    file_dir = os.path.dirname(path)
    results: set[str] = set()
    exts = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json"]
    for match in JS_IMPORT_RE.findall(text):
        if not match.startswith((".", "/")):
            continue
        base = (
            os.path.join(project_root, match.lstrip("/"))
            if match.startswith("/")
            else os.path.join(file_dir, match)
        )
        candidates: list[str] = []
        if os.path.splitext(base)[1]:
            candidates.append(base)
        else:
            for ext in exts:
                candidates.append(base + ext)
            for ext in exts:
                candidates.append(os.path.join(base, f"index{ext}"))
        for candidate in candidates:
            if os.path.isfile(candidate) and is_within_root(candidate, project_root):
                results.add(candidate)
    return sorted(results)


def expand_dependencies(
    seed_files: list[str], project_root: str, depth: int
) -> list[str]:
    if depth <= 0:
        return seed_files
    resolved: set[str] = set()
    frontier = [path for path in seed_files if os.path.isfile(path)]
    for path in frontier:
        resolved.add(path)
    for _ in range(depth):
        next_frontier: list[str] = []
        for path in frontier:
            ext = os.path.splitext(path)[1].lower()
            deps: list[str] = []
            if ext == ".py":
                deps = resolve_python_imports(path, project_root)
            elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                deps = resolve_js_imports(path, project_root)
            for dep in deps:
                if dep not in resolved:
                    resolved.add(dep)
                    next_frontier.append(dep)
        if not next_frontier:
            break
        frontier = next_frontier
    return sorted(resolved)


def build_context_bundle(
    *,
    run_dir: str,
    project_root: str,
    cwd_path: str,
    changed_files: list[str],
    config: dict[str, str],
) -> tuple[str, str]:
    enabled = parse_bool(
        config.get("REVIEWER_CONTEXT") or os.environ.get("REVIEWER_CONTEXT"),
        True,
    )
    if not enabled:
        return "", ""
    max_bytes = parse_int(
        config.get("REVIEWER_CONTEXT_MAX_BYTES")
        or os.environ.get("REVIEWER_CONTEXT_MAX_BYTES"),
        500_000,
    )
    depth = parse_int(
        config.get("REVIEWER_CONTEXT_DEPTH")
        or os.environ.get("REVIEWER_CONTEXT_DEPTH"),
        2,
    )
    if max_bytes <= 0:
        return "", ""

    root = os.path.abspath(project_root or cwd_path)
    abs_changed: list[str] = []
    deleted_entries: list[tuple[str, str]] = []
    for raw in changed_files:
        if not raw:
            continue
        abs_path = normalize_path(raw, cwd_path)
        if os.path.isfile(abs_path):
            abs_changed.append(abs_path)
        else:
            snap_path = os.path.join(run_dir, "before", abs_path.lstrip("/"))
            if os.path.isfile(snap_path):
                deleted_entries.append((snap_path, f"{abs_path} (before deleted)"))

    adjacent_files = list_adjacent_context_files(abs_changed, root)
    infra_files = list_infra_files(root)
    test_files = list_test_files(root, abs_changed)
    deps = expand_dependencies(abs_changed, root, depth)

    ordered: list[str] = []
    seen: set[str] = set()

    def add_paths(paths: list[str]) -> None:
        for path in paths:
            if not path or path in seen or not os.path.isfile(path):
                continue
            ordered.append(path)
            seen.add(path)

    add_paths(abs_changed)
    add_paths(adjacent_files)
    add_paths(test_files)
    add_paths(infra_files)
    add_paths([path for path in deps if path not in seen])

    entries: list[tuple[str, str]] = [
        (path, display_path(path, root)) for path in ordered
    ]
    for snap_path, label in deleted_entries:
        if snap_path not in seen:
            entries.append((snap_path, label))
            seen.add(snap_path)

    context_file = os.path.join(run_dir, "context.txt")
    context_files_file = os.path.join(run_dir, "context.files")
    included_labels: list[str] = []
    total = 0
    truncated = False
    with open(context_file, "w", encoding="utf-8") as out:
        for path, label in entries:
            if not is_text_file(path):
                continue
            header = f"FILE: {label}\n"
            remaining = max_bytes - total
            if remaining <= len(header):
                truncated = True
                break
            out.write(header)
            total += len(header)
            content, clipped = read_text_limited(path, remaining - len(header))
            out.write(content)
            out.write("\n\n")
            total += len(content) + 2
            included_labels.append(label)
            if clipped:
                truncated = True
                out.write("[context truncated to fit budget]\n")
                break
        if truncated:
            out.write("[context truncated: increase REVIEWER_CONTEXT_MAX_BYTES to include more]\n")

    if included_labels:
        write_text(context_files_file, "\n".join(included_labels) + "\n")
    else:
        write_text(context_files_file, "")
    return context_file, context_files_file


def extract_review_field(path: str, label: str) -> str:
    if not os.path.isfile(path):
        return ""
    pattern = re.compile(rf"^{re.escape(label)}:\s*", re.IGNORECASE)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if pattern.match(line):
                return pattern.sub("", line).strip()
    return ""


def build_review_question(backend: str, codex_out: str, claude_out: str) -> str:
    if backend == "both":
        codex_blockers = extract_review_field(codex_out, "BLOCKERS") or "none"
        codex_notes = extract_review_field(codex_out, "NOTES") or "none"
        claude_blockers = extract_review_field(claude_out, "BLOCKERS") or "none"
        claude_notes = extract_review_field(claude_out, "NOTES") or "none"
        summary = (
            "Review complete. "
            f"CODEX blockers: {codex_blockers}; notes: {codex_notes}. "
            f"CLAUDE blockers: {claude_blockers}; notes: {claude_notes}."
        )
        question = (
            f"{summary} I should apply the review feedback now and continue. Do you "
            "agree? Reply with review:apply to proceed, or review:skip to skip reviews "
            "for the rest of this session (you can re-enable later with review:enable "
            "or review:resume)."
        )
    else:
        review_file = codex_out if backend != "claude" else claude_out
        blockers = extract_review_field(review_file, "BLOCKERS") or "none"
        notes = extract_review_field(review_file, "NOTES") or "none"
        summary = f"Review complete. Blockers: {blockers}. Notes: {notes}."
        question = (
            f"{summary} I should apply the review feedback now and continue. Do you "
            "agree? Reply with review:apply to proceed, or review:skip to skip reviews "
            "for the rest of this session (you can re-enable later with review:enable "
            "or review:resume)."
        )
    question = re.sub(r"\s+", " ", question).strip()
    return question


SMOKE_FOCUS = (
    "Focus on obvious issues: syntax errors, clear security flaws, obvious logic bugs, "
    "missing null checks. Be fast and surface-level."
)

SEMANTIC_FOCUS = (
    "Perform deep semantic analysis: trace data flow, verify invariants, check edge cases, "
    "validate error handling, assess test coverage."
)

STRUCTURED_REVIEW_TEMPLATE = """You are a strict code reviewer. {pass_focus}

The diff is the source of truth; use context files only to interpret the diff.
IMPORTANT: Verify every claim. Use web search if needed.

Respond with ONLY valid JSON in this exact format:
{{
  "findings": [
    {{
      "id": "F001",
      "severity": "critical|high|medium|low|info",
      "category": "correctness|security|regression|missing_test|style|performance",
      "confidence": 0-100,
      "title": "Short issue title",
      "description": "Detailed explanation",
      "file_path": "path/to/file.py or null",
      "line_start": 42 or null,
      "line_end": 45 or null,
      "suggested_fix": "Code or guidance, or null"
    }}
  ],
  "blockers_summary": "List of blocking issues or 'none'",
  "notes_summary": "List of non-blocking improvements"
}}

Confidence guidelines:
- 90-100: Certain - verified via docs/search or obvious from code
- 70-89: High - strong evidence in diff/context
- 50-69: Medium - likely but needs verification
- 30-49: Low - possible issue, uncertain
- 0-29: Speculative - mention for awareness only"""


def build_structured_review_prompt(pass_type: str) -> str:
    """Build prompt requesting JSON output with findings schema."""
    if pass_type == "smoke":
        return STRUCTURED_REVIEW_TEMPLATE.format(pass_focus=SMOKE_FOCUS)
    return STRUCTURED_REVIEW_TEMPLATE.format(pass_focus=SEMANTIC_FOCUS)


def parse_structured_output(raw_output: str, pass_number: int, backend: str) -> ReviewResult:
    """Parse JSON from reviewer output, fallback to legacy parsing."""
    result = ReviewResult(
        timestamp=utc_now(),
        backend=backend,
        pass_number=pass_number,
        pass_type="smoke" if pass_number == 1 else "semantic",
    )
    text = raw_output.strip()
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return parse_legacy_output(raw_output, pass_number, backend)
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return parse_legacy_output(raw_output, pass_number, backend)
    findings_data = data.get("findings", [])
    if not isinstance(findings_data, list):
        return parse_legacy_output(raw_output, pass_number, backend)
    findings: list[Finding] = []
    for i, f in enumerate(findings_data):
        if not isinstance(f, dict):
            continue
        raw_severity = str(f.get("severity", "info")).lower()
        valid_severities = {"critical", "high", "medium", "low", "info"}
        severity = raw_severity if raw_severity in valid_severities else "info"
        raw_category = str(f.get("category", "correctness")).lower()
        valid_categories = {
            "correctness", "security", "regression",
            "missing_test", "style", "performance"
        }
        category = raw_category if raw_category in valid_categories else "correctness"
        finding = Finding(
            id=str(f.get("id", f"F{i+1:03d}")),
            severity=severity,
            category=category,
            confidence=max(0, min(100, (
                int(f.get("confidence", 50))
                if isinstance(f.get("confidence"), (int, float)) else 50
            ))),
            title=str(f.get("title", "")),
            description=str(f.get("description", "")),
            file_path=f.get("file_path") if f.get("file_path") else None,
            line_start=int(f["line_start"]) if f.get("line_start") else None,
            line_end=int(f["line_end"]) if f.get("line_end") else None,
            suggested_fix=f.get("suggested_fix") if f.get("suggested_fix") else None,
            pass_number=pass_number,
        )
        findings.append(finding)
    result.findings = findings
    result.blockers_summary = str(data.get("blockers_summary", ""))
    result.notes_summary = str(data.get("notes_summary", ""))
    return result


def parse_legacy_output(raw_output: str, pass_number: int, backend: str) -> ReviewResult:
    """Parse BLOCKERS:/NOTES: format into ReviewResult."""
    result = ReviewResult(
        timestamp=utc_now(),
        backend=backend,
        pass_number=pass_number,
        pass_type="smoke" if pass_number == 1 else "semantic",
    )
    re_flags = re.MULTILINE | re.DOTALL | re.IGNORECASE
    blockers_match = re.search(r'^BLOCKERS:\s*(.+?)(?=^NOTES:|$)', raw_output, re_flags)
    notes_match = re.search(r'^NOTES:\s*(.+?)$', raw_output, re_flags)
    blockers_text = blockers_match.group(1).strip() if blockers_match else ""
    notes_text = notes_match.group(1).strip() if notes_match else ""
    result.blockers_summary = blockers_text or "none"
    result.notes_summary = notes_text or "none"
    if blockers_text and blockers_text.lower() != "none":
        finding = Finding(
            id="F001",
            severity="high",
            category="correctness",
            confidence=70,
            title="Legacy blocker",
            description=blockers_text,
            pass_number=pass_number,
        )
        result.findings.append(finding)
    return result


def write_structured_json(run_dir: str, results: list[ReviewResult]) -> str:
    """Write review.structured.json with all findings."""
    output_path = os.path.join(run_dir, "review.structured.json")
    all_findings: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            continue
        pass_data = {
            "backend": result.backend,
            "pass_type": result.pass_type,
            "pass_number": result.pass_number,
            "blockers_summary": result.blockers_summary,
            "notes_summary": result.notes_summary,
            "findings_count": len(result.findings),
        }
        passes.append(pass_data)
        for finding in result.findings:
            all_findings.append(asdict(finding))
    output = {
        "version": "1.0",
        "generated_at": utc_now(),
        "passes": passes,
        "all_findings": all_findings,
    }
    write_text(output_path, json.dumps(output, indent=2) + "\n")
    return output_path


def format_finding_for_display(finding: Finding) -> str:
    """Human-readable finding format for stderr output."""
    location = ""
    if finding.file_path:
        location = f" [{finding.file_path}"
        if finding.line_start:
            location += f":{finding.line_start}"
            if finding.line_end and finding.line_end != finding.line_start:
                location += f"-{finding.line_end}"
        location += "]"
    return (
        f"[{finding.severity.upper()}] ({finding.confidence}%) {finding.title}{location}\n"
        f"  {finding.description}"
    )


def has_blocking_findings(result: ReviewResult, threshold: int) -> bool:
    """True if any critical/high finding has confidence >= threshold."""
    for finding in result.findings:
        if finding.severity in ("critical", "high") and finding.confidence >= threshold:
            return True
    return False


def get_blocking_findings(result: ReviewResult, threshold: int) -> list[Finding]:
    """Get findings that meet blocking criteria."""
    return [
        f for f in result.findings
        if f.severity in ("critical", "high") and f.confidence >= threshold
    ]


def build_review_question_structured(results: list[ReviewResult], threshold: int) -> str:
    """Build gate question showing confidence scores."""
    blocking: list[Finding] = []
    notes: list[Finding] = []
    for result in results:
        if result is None:
            continue
        for finding in result.findings:
            if finding.severity in ("critical", "high") and finding.confidence >= threshold:
                blocking.append(finding)
            else:
                notes.append(finding)
    if blocking:
        blockers_str = "; ".join(
            f"{f.title} ({f.confidence}% confidence)" for f in blocking[:3]
        )
        if len(blocking) > 3:
            blockers_str += f" +{len(blocking) - 3} more"
    else:
        blockers_str = "none"
    if notes:
        notes_str = f"{len(notes)} non-blocking items"
    else:
        notes_str = "none"
    summary = (
        f"Review complete. Blockers (>={threshold}% confidence): "
        f"{blockers_str}. Notes: {notes_str}."
    )
    question = (
        f"{summary} I should apply the review feedback now and continue. Do you "
        "agree? Reply with review:apply to proceed, or review:skip to skip reviews "
        "for the rest of this session (you can re-enable later with review:enable "
        "or review:resume)."
    )
    return re.sub(r"\s+", " ", question).strip()


def review_has_blockers(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if re.search(r"^BLOCKERS:\s*none\b", content, re.IGNORECASE | re.MULTILINE):
        return False
    return True


def first_blockers_line(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if re.match(r"^BLOCKERS:\s*", line, re.IGNORECASE):
                return line.rstrip("\n")
    return ""


def parse_gate_prompt(prompt_text: str) -> tuple[str, bool]:
    action = ""
    if re.search(r"\breview:apply\b", prompt_text, re.IGNORECASE):
        action = "apply"
    elif re.search(r"\breview:skip\b", prompt_text, re.IGNORECASE):
        action = "skip"
    enable = bool(re.search(r"\breview:(enable|resume)\b", prompt_text, re.IGNORECASE))
    return action, enable


def parse_gate_tool_response(data: dict[str, Any]) -> dict[str, str] | None:
    if data.get("tool_name") != "AskUserQuestion":
        return None
    resp = data.get("tool_response") or {}
    answers: list[str] = []
    questions: list[str] = []

    if isinstance(resp, dict):
        answers_obj = resp.get("answers")
        if isinstance(answers_obj, dict):
            answers.extend([str(v) for v in answers_obj.values()])
        questions_obj = resp.get("questions")
        if isinstance(questions_obj, list):
            for q in questions_obj:
                if isinstance(q, dict):
                    questions.append(str(q.get("question", "")))

    review_question = any(
        (
            "review:apply" in q.lower()
            or "review:skip" in q.lower()
            or "review complete" in q.lower()
        )
        for q in questions
    )
    if not review_question:
        return None

    answers_text = " | ".join([a.replace("\\n", " ").strip() for a in answers if a.strip()])
    questions_text = " | ".join([q.replace("\\n", " ").strip() for q in questions if q.strip()])

    action = ""
    enable = False
    for text in answers:
        if re.search(r"\breview:skip\b", text, re.IGNORECASE):
            action = "skip"
            break
    for text in answers:
        if re.search(r"\breview:apply\b", text, re.IGNORECASE):
            action = "apply"
            break
    for text in answers:
        if re.search(r"\breview:(enable|resume)\b", text, re.IGNORECASE):
            action = "apply"
            enable = True
            break

    if not action:
        for text in answers:
            if re.search(r"\bapply\b", text, re.IGNORECASE):
                action = "apply"
                break
            if re.search(r"\bskip\b", text, re.IGNORECASE):
                action = "skip"
                break

    return {
        "action": action,
        "enable": "1" if enable else "0",
        "answer": answers_text,
        "question": questions_text,
    }


def append_diff(
    diff_file: str,
    before_path: str,
    after_path: str,
    before_label: str,
    after_label: str,
) -> None:
    cmd = [
        "diff",
        "-u",
        "--label",
        before_label,
        "--label",
        after_label,
        before_path,
        after_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        append_text(diff_file, result.stdout)


def run_codex(
    review_prompt: str,
    diff: str,
    codex_out: str,
    base_dir: str,
    context_file: str,
    project_root: str,
) -> None:
    if not shutil.which("codex"):
        return
    context_text = read_text(context_file) if context_file else ""
    input_text = f"{review_prompt}\n\nDIFF:\n{diff}\n"
    if context_text:
        input_text += f"\nCONTEXT:\n{context_text}\n"
    cmd = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        codex_out,
        "-",
    ]
    if project_root and os.path.isdir(project_root):
        cmd[2:2] = ["-C", project_root]
    subprocess.run(cmd, input=input_text, text=True, check=False)
    if os.path.isfile(codex_out):
        shutil.copyfile(codex_out, os.path.join(base_dir, "latest.codex.md"))


def run_claude(
    review_prompt: str,
    diff_file: str,
    run_dir: str,
    claude_out: str,
    base_dir: str,
    context_file: str,
    project_root: str,
) -> None:
    if not shutil.which("claude"):
        return
    prompt = f"Review the diff at {diff_file}."
    if context_file:
        prompt += f" Additional context is available at {context_file}."
    prompt += f" Use the following format:\n{review_prompt}"
    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "Read,Glob,WebSearch,WebFetch",
        "--add-dir",
        run_dir,
    ]
    if project_root and os.path.isdir(project_root):
        cmd.extend(["--add-dir", project_root])
    with open(claude_out, "w", encoding="utf-8") as out:
        subprocess.run(cmd, input=prompt, stdout=out, text=True, check=False)
    if os.path.isfile(claude_out):
        shutil.copyfile(claude_out, os.path.join(base_dir, "latest.claude.md"))


def run_codex_structured(
    pass_type: str,
    diff: str,
    output_file: str,
    base_dir: str,
    context_file: str,
    project_root: str,
) -> ReviewResult | None:
    """Run Codex with structured JSON prompt and parse result."""
    if not shutil.which("codex"):
        return None
    prompt = build_structured_review_prompt(pass_type)
    context_text = read_text(context_file) if context_file else ""
    input_text = f"{prompt}\n\nDIFF:\n{diff}\n"
    if context_text:
        input_text += f"\nCONTEXT:\n{context_text}\n"
    cmd = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        output_file,
        "-",
    ]
    if project_root and os.path.isdir(project_root):
        cmd[2:2] = ["-C", project_root]
    subprocess.run(cmd, input=input_text, text=True, check=False)
    if os.path.isfile(output_file):
        shutil.copyfile(output_file, os.path.join(base_dir, f"latest.codex.{pass_type}.md"))
        raw_output = read_text(output_file)
        return parse_structured_output(raw_output, 1 if pass_type == "smoke" else 2, "codex")
    return None


def run_claude_structured(
    pass_type: str,
    diff_file: str,
    run_dir: str,
    output_file: str,
    base_dir: str,
    context_file: str,
    project_root: str,
) -> ReviewResult | None:
    """Run Claude with structured JSON prompt and parse result."""
    if not shutil.which("claude"):
        return None
    review_prompt = build_structured_review_prompt(pass_type)
    prompt = f"Review the diff at {diff_file}."
    if context_file:
        prompt += f" Additional context is available at {context_file}."
    prompt += f" Use the following format:\n{review_prompt}"
    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "Read,Glob,WebSearch,WebFetch",
        "--add-dir",
        run_dir,
    ]
    if project_root and os.path.isdir(project_root):
        cmd.extend(["--add-dir", project_root])
    with open(output_file, "w", encoding="utf-8") as out:
        subprocess.run(cmd, input=prompt, stdout=out, text=True, check=False)
    if os.path.isfile(output_file):
        shutil.copyfile(output_file, os.path.join(base_dir, f"latest.claude.{pass_type}.md"))
        raw_output = read_text(output_file)
        return parse_structured_output(raw_output, 1 if pass_type == "smoke" else 2, "claude")
    return None


def run_multi_pass_review(
    backend: str,
    diff: str,
    diff_file: str,
    run_dir: str,
    base_dir: str,
    context_file: str,
    project_root: str,
    config: dict[str, str],
) -> tuple[ReviewResult | None, ReviewResult | None]:
    """Run two-pass review: smoke check first, then semantic if needed."""
    confidence_threshold = parse_int(
        config.get("REVIEWER_CONFIDENCE_THRESHOLD")
        or os.environ.get("REVIEWER_CONFIDENCE_THRESHOLD"),
        70,
    )
    smoke_result: ReviewResult | None = None
    semantic_result: ReviewResult | None = None
    if backend == "codex":
        smoke_out = os.path.join(run_dir, "review.codex.smoke.md")
        smoke_result = run_codex_structured(
            "smoke", diff, smoke_out, base_dir, context_file, project_root
        )
        if smoke_result and has_blocking_findings(smoke_result, confidence_threshold):
            return smoke_result, None
        semantic_out = os.path.join(run_dir, "review.codex.semantic.md")
        semantic_result = run_codex_structured(
            "semantic", diff, semantic_out, base_dir, context_file, project_root
        )
    elif backend == "claude":
        smoke_out = os.path.join(run_dir, "review.claude.smoke.md")
        smoke_result = run_claude_structured(
            "smoke", diff_file, run_dir, smoke_out, base_dir, context_file, project_root
        )
        if smoke_result and has_blocking_findings(smoke_result, confidence_threshold):
            return smoke_result, None
        semantic_out = os.path.join(run_dir, "review.claude.semantic.md")
        semantic_result = run_claude_structured(
            "semantic", diff_file, run_dir, semantic_out,
            base_dir, context_file, project_root
        )
    elif backend == "both":
        smoke_codex_out = os.path.join(run_dir, "review.codex.smoke.md")
        smoke_codex = run_codex_structured(
            "smoke", diff, smoke_codex_out, base_dir, context_file, project_root
        )
        smoke_claude_out = os.path.join(run_dir, "review.claude.smoke.md")
        smoke_claude = run_claude_structured(
            "smoke", diff_file, run_dir, smoke_claude_out,
            base_dir, context_file, project_root
        )
        smoke_result = smoke_codex or smoke_claude
        if smoke_result:
            codex_findings = smoke_codex.findings if smoke_codex else []
            claude_findings = smoke_claude.findings if smoke_claude else []
            smoke_result.findings = codex_findings + claude_findings
        has_codex_blockers = (
            smoke_codex and has_blocking_findings(smoke_codex, confidence_threshold)
        )
        has_claude_blockers = (
            smoke_claude and has_blocking_findings(smoke_claude, confidence_threshold)
        )
        if has_codex_blockers or has_claude_blockers:
            return smoke_result, None
        semantic_codex_out = os.path.join(run_dir, "review.codex.semantic.md")
        semantic_codex = run_codex_structured(
            "semantic", diff, semantic_codex_out, base_dir, context_file, project_root
        )
        semantic_claude_out = os.path.join(run_dir, "review.claude.semantic.md")
        semantic_claude = run_claude_structured(
            "semantic", diff_file, run_dir, semantic_claude_out,
            base_dir, context_file, project_root
        )
        semantic_result = semantic_codex or semantic_claude
        if semantic_result:
            codex_findings = semantic_codex.findings if semantic_codex else []
            claude_findings = semantic_claude.findings if semantic_claude else []
            semantic_result.findings = codex_findings + claude_findings
    return smoke_result, semantic_result


def write_run_log(
    log_md: str,
    update_json: bool,
    *,
    prompt_ts: str,
    session_id: str,
    prompt_id: str,
    prompt_cwd: str,
    gate_action: str,
    gate_question: str,
    gate_answer: str,
    files_file: str,
    diff_file: str,
    backend: str,
    codex_out: str,
    claude_out: str,
    context_file: str,
    context_files_file: str,
    log_file: str,
    structured_results: list[ReviewResult] | None = None,
) -> None:
    files = [line for line in read_lines(files_file) if line]
    diff_content = read_text(diff_file)
    context_files = [line for line in read_lines(context_files_file) if line]
    run_dir = os.path.dirname(log_md)
    structured_json = os.path.join(run_dir, "review.structured.json")

    with open(log_md, "w", encoding="utf-8") as f:
        f.write(f"timestamp: {prompt_ts}\n")
        f.write(f"session_id: {session_id}\n")
        f.write(f"prompt_id: {prompt_id}\n")
        f.write(f"cwd: {prompt_cwd}\n")
        f.write(f"gate: {gate_action}\n")
        if gate_question:
            f.write(f"gate_question: {gate_question}\n")
        if gate_answer:
            f.write(f"gate_answer: {gate_answer}\n")
        f.write("files:\n")
        for line in files:
            f.write(f"- {line}\n")
        if context_files:
            f.write("context_files:\n")
            for line in context_files:
                f.write(f"- {line}\n")
        if context_file and os.path.isfile(context_file):
            f.write(f"context_bundle: {context_file}\n")
        if os.path.isfile(structured_json):
            f.write(f"structured_output: {structured_json}\n")
        f.write("\n")
        f.write("diff:\n")
        f.write("```diff\n")
        f.write(diff_content)
        f.write("\n```\n")
        if os.path.isfile(codex_out):
            f.write("\n")
            f.write("codex_response:\n")
            f.write(read_text(codex_out))
            f.write("\n")
        if os.path.isfile(claude_out):
            f.write("\n")
            f.write("claude_response:\n")
            f.write(read_text(claude_out))
            f.write("\n")
        if structured_results:
            f.write("\n## Structured Findings\n\n")
            for result in structured_results:
                if result is None:
                    continue
                f.write(f"### {result.backend} (Pass {result.pass_number}: {result.pass_type})\n")
                if result.findings:
                    for finding in result.findings:
                        f.write(f"- **[{finding.severity.upper()}]** ({finding.confidence}%) ")
                        f.write(f"{finding.title}")
                        if finding.file_path:
                            f.write(f" [{finding.file_path}")
                            if finding.line_start:
                                f.write(f":{finding.line_start}")
                            f.write("]")
                        f.write("\n")
                else:
                    f.write("No findings.\n")
                f.write("\n")

    shutil.copyfile(log_md, os.path.join(os.path.dirname(log_file), "latest.log.md"))

    findings_summary: list[dict[str, Any]] = []
    if structured_results:
        for result in structured_results:
            if result is None:
                continue
            for finding in result.findings:
                findings_summary.append(asdict(finding))

    entry: dict[str, Any] = {
        "timestamp": prompt_ts,
        "session_id": session_id,
        "prompt_id": int(prompt_id or 0),
        "cwd": prompt_cwd,
        "files": files,
        "diff_file": diff_file,
        "gate": gate_action,
        "gate_question": gate_question,
        "gate_answer": gate_answer,
        "backend": backend,
        "codex_output": codex_out,
        "claude_output": claude_out,
        "context_file": context_file if os.path.isfile(context_file) else "",
        "context_files": context_files,
        "structured_json": structured_json if os.path.isfile(structured_json) else "",
        "findings": findings_summary,
    }

    if update_json and os.path.exists(log_file):
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()
        replaced = False
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                existing.get("session_id") == entry["session_id"]
                and int(existing.get("prompt_id", -1)) == entry["prompt_id"]
            ):
                lines[i] = json.dumps(entry) + "\n"
                replaced = True
                break
        if not replaced:
            lines.append(json.dumps(entry) + "\n")
        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def main() -> int:
    data = read_json_stdin()
    hook_event_name = json_get(data, "hook_event_name")
    session_id = json_get(data, "session_id")
    tool_name = json_get(data, "tool_name")
    file_path = json_get(data, "tool_input.file_path")
    cwd = json_get(data, "cwd")
    prompt_text = json_get(data, "prompt")

    cwd_path = os.path.abspath(cwd) if cwd else os.getcwd()
    project_root = find_project_root(cwd_path) or cwd_path
    config_file = resolve_config_path(cwd_path)
    config = parse_config(config_file)
    backend = config.get("REVIEWER_BACKEND") or os.environ.get("REVIEWER_BACKEND", "codex")
    block = config.get("REVIEWER_BLOCK") or os.environ.get("REVIEWER_BLOCK", "0")

    if not session_id:
        return 0

    base_dir = resolve_log_dir(config)
    state_dir = os.path.join(base_dir, "state")
    runs_dir = os.path.join(base_dir, "runs")
    log_file = os.path.join(base_dir, "log.jsonl")

    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    session_dir = os.path.join(state_dir, session_id)
    prompt_id_file = os.path.join(session_dir, "prompt_id")
    files_file = os.path.join(session_dir, "files")
    prompt_ts_file = os.path.join(session_dir, "prompt_ts")
    cwd_file = os.path.join(session_dir, "cwd")
    gate_file = os.path.join(session_dir, "review_gate")
    gate_answer_file = os.path.join(session_dir, "review_gate_answer")
    gate_question_file = os.path.join(session_dir, "review_gate_question")
    session_skip_file = os.path.join(session_dir, "review_session_skip")
    pending_run_file = os.path.join(session_dir, "pending_log_run")

    review_prompt = (
        "You are a strict reviewer. Find correctness bugs, security issues, "
        "behavior regressions, and missing tests. The diff is the source of truth; "
        "use any provided context files only to interpret the diff, not to review "
        "unrelated code. IMPORTANT: Verify every claim. IMPORTANT: Use web search if "
        "needed to validate specifics. Respond with:\n"
        "BLOCKERS: <list or 'none'>\n"
        "NOTES: <short list of improvements>"
    )

    if hook_event_name == "UserPromptSubmit":
        os.makedirs(session_dir, exist_ok=True)
        prompt_id = read_text(prompt_id_file).strip() or "0"
        if not prompt_id.isdigit():
            prompt_id = "0"
        prompt_id = str(int(prompt_id) + 1)
        write_text(prompt_id_file, f"{prompt_id}\n")
        prompt_ts = utc_now()
        write_text(prompt_ts_file, f"{prompt_ts}\n")
        write_text(cwd_file, f"{cwd}\n")
        write_text(files_file, "")
        for fp in (gate_file, gate_answer_file, gate_question_file):
            if os.path.exists(fp):
                os.remove(fp)
        gate_action, gate_enable = parse_gate_prompt(prompt_text or "")
        if gate_enable:
            if os.path.exists(session_skip_file):
                os.remove(session_skip_file)
        if gate_action:
            write_text(gate_file, f"{gate_action}\n")
            if gate_action == "skip":
                write_text(session_skip_file, "1\n")
            elif gate_action == "apply" and os.path.exists(session_skip_file):
                os.remove(session_skip_file)
        run_dir = os.path.join(runs_dir, session_id, f"prompt-{prompt_id}")
        os.makedirs(os.path.join(run_dir, "before"), exist_ok=True)
        return 0

    if hook_event_name == "PostToolUse":
        if tool_name != "AskUserQuestion":
            return 0
        gate_info = parse_gate_tool_response(data)
        if not gate_info:
            return 0
        gate_action = gate_info.get("action", "")
        gate_enable = gate_info.get("enable", "")
        gate_answer = gate_info.get("answer", "")
        gate_question = gate_info.get("question", "")
        if gate_enable == "1" and os.path.exists(session_skip_file):
            os.remove(session_skip_file)
        if gate_action:
            os.makedirs(session_dir, exist_ok=True)
            write_text(gate_file, f"{gate_action}\n")
            if gate_action == "skip":
                write_text(session_skip_file, "1\n")
            elif gate_action == "apply" and os.path.exists(session_skip_file):
                os.remove(session_skip_file)
        if gate_answer:
            os.makedirs(session_dir, exist_ok=True)
            write_text(gate_answer_file, f"{gate_answer}\n")
        if gate_question:
            os.makedirs(session_dir, exist_ok=True)
            write_text(gate_question_file, f"{gate_question}\n")

        pending_run_dir = read_text(pending_run_file).strip()
        prompt_id = ""
        if pending_run_dir:
            run_dir = pending_run_dir
            run_prompt = os.path.basename(run_dir)
            if run_prompt.startswith("prompt-"):
                prompt_id = run_prompt[len("prompt-") :]
        else:
            prompt_id = read_text(prompt_id_file).strip()
            run_dir = os.path.join(runs_dir, session_id, f"prompt-{prompt_id}")
        if not prompt_id:
            prompt_id = read_text(prompt_id_file).strip()

        diff_file = os.path.join(run_dir, "diff.patch")
        if os.path.isfile(diff_file) and os.path.getsize(diff_file) > 0:
            prompt_ts = read_text(prompt_ts_file).strip() or utc_now()
            prompt_cwd = read_text(cwd_file).strip()
            gate_action = read_text(gate_file).strip()
            gate_answer = read_text(gate_answer_file).strip()
            gate_question = read_text(gate_question_file).strip()
            codex_out = os.path.join(run_dir, "review.codex.md")
            claude_out = os.path.join(run_dir, "review.claude.md")
            log_md = os.path.join(run_dir, "review.log.md")
            context_file = os.path.join(run_dir, "context.txt")
            context_files_file = os.path.join(run_dir, "context.files")
            update_json = os.path.exists(log_md)
            write_run_log(
                log_md,
                update_json,
                prompt_ts=prompt_ts,
                session_id=session_id,
                prompt_id=prompt_id,
                prompt_cwd=prompt_cwd,
                gate_action=gate_action,
                gate_question=gate_question,
                gate_answer=gate_answer,
                files_file=files_file,
                diff_file=diff_file,
                backend=backend,
                codex_out=codex_out,
                claude_out=claude_out,
                context_file=context_file,
                context_files_file=context_files_file,
                log_file=log_file,
            )
            if os.path.exists(pending_run_file):
                os.remove(pending_run_file)
        return 0

    if hook_event_name == "PreToolUse":
        if tool_name not in {"Edit", "Write"}:
            return 0
        prompt_id = read_text(prompt_id_file).strip()
        if not prompt_id:
            return 0
        run_dir = os.path.join(runs_dir, session_id, f"prompt-{prompt_id}")
        os.makedirs(os.path.join(run_dir, "before"), exist_ok=True)
        if not file_path:
            return 0
        os.makedirs(session_dir, exist_ok=True)
        if not os.path.exists(files_file):
            write_text(files_file, "")
        if file_path not in read_lines(files_file):
            append_text(files_file, f"{file_path}\n")
        snap_path = os.path.join(run_dir, "before", file_path.lstrip("/"))
        if not os.path.isfile(snap_path):
            ensure_parent(snap_path)
            if os.path.isfile(file_path):
                shutil.copyfile(file_path, snap_path)
            else:
                write_text(snap_path, "")
        return 0

    if hook_event_name == "Stop":
        prompt_id = read_text(prompt_id_file).strip()
        if not prompt_id:
            return 0
        if os.path.exists(session_skip_file):
            return 0
        if not os.path.isfile(files_file) or not any(read_lines(files_file)):
            return 0
        run_dir = os.path.join(runs_dir, session_id, f"prompt-{prompt_id}")
        os.makedirs(run_dir, exist_ok=True)
        diff_file = os.path.join(run_dir, "diff.patch")
        write_text(diff_file, "")
        for fp in read_lines(files_file):
            if not fp:
                continue
            snap_path = os.path.join(run_dir, "before", fp.lstrip("/"))
            if os.path.isfile(snap_path):
                if os.path.isfile(fp):
                    append_diff(
                        diff_file,
                        snap_path,
                        fp,
                        f"{fp} (before)",
                        f"{fp} (after)",
                    )
                else:
                    append_diff(
                        diff_file,
                        snap_path,
                        "/dev/null",
                        f"{fp} (before)",
                        f"{fp} (deleted)",
                    )
        if not os.path.isfile(diff_file) or os.path.getsize(diff_file) == 0:
            return 0

        multi_pass = parse_bool(
            config.get("REVIEWER_MULTI_PASS") or os.environ.get("REVIEWER_MULTI_PASS"),
            True,
        )
        confidence_threshold = parse_int(
            config.get("REVIEWER_CONFIDENCE_THRESHOLD")
            or os.environ.get("REVIEWER_CONFIDENCE_THRESHOLD"),
            70,
        )

        diff = read_text(diff_file)
        prompt_ts = read_text(prompt_ts_file).strip() or utc_now()
        prompt_cwd = read_text(cwd_file).strip()
        gate_action = read_text(gate_file).strip()
        gate_answer = read_text(gate_answer_file).strip()
        gate_question = read_text(gate_question_file).strip()
        codex_out = os.path.join(run_dir, "review.codex.md")
        claude_out = os.path.join(run_dir, "review.claude.md")
        changed_files = [line for line in read_lines(files_file) if line]
        context_file, context_files_file = build_context_bundle(
            run_dir=run_dir,
            project_root=project_root,
            cwd_path=cwd_path,
            changed_files=changed_files,
            config=config,
        )

        structured_results: list[ReviewResult] = []
        if multi_pass:
            smoke, semantic = run_multi_pass_review(
                backend, diff, diff_file, run_dir, base_dir,
                context_file, project_root, config,
            )
            if smoke:
                structured_results.append(smoke)
            if semantic:
                structured_results.append(semantic)
        else:
            if backend == "codex":
                run_codex(
                    review_prompt,
                    diff,
                    codex_out,
                    base_dir,
                    context_file,
                    project_root,
                )
                if os.path.isfile(codex_out):
                    result = parse_structured_output(read_text(codex_out), 1, "codex")
                    structured_results.append(result)
            elif backend == "claude":
                run_claude(
                    review_prompt,
                    diff_file,
                    run_dir,
                    claude_out,
                    base_dir,
                    context_file,
                    project_root,
                )
                if os.path.isfile(claude_out):
                    result = parse_structured_output(read_text(claude_out), 1, "claude")
                    structured_results.append(result)
            elif backend == "both":
                run_codex(
                    review_prompt,
                    diff,
                    codex_out,
                    base_dir,
                    context_file,
                    project_root,
                )
                run_claude(
                    review_prompt,
                    diff_file,
                    run_dir,
                    claude_out,
                    base_dir,
                    context_file,
                    project_root,
                )
                if os.path.isfile(codex_out):
                    result = parse_structured_output(read_text(codex_out), 1, "codex")
                    structured_results.append(result)
                if os.path.isfile(claude_out):
                    result = parse_structured_output(read_text(claude_out), 1, "claude")
                    structured_results.append(result)
            else:
                return 0

        if structured_results:
            write_structured_json(run_dir, structured_results)

        log_md = os.path.join(run_dir, "review.log.md")
        if block == "2" and not gate_action:
            write_text(pending_run_file, f"{run_dir}\n")
        else:
            write_run_log(
                log_md,
                False,
                prompt_ts=prompt_ts,
                session_id=session_id,
                prompt_id=prompt_id,
                prompt_cwd=prompt_cwd,
                gate_action=gate_action,
                gate_question=gate_question,
                gate_answer=gate_answer,
                files_file=files_file,
                diff_file=diff_file,
                backend=backend,
                codex_out=codex_out,
                claude_out=claude_out,
                context_file=context_file,
                context_files_file=context_files_file,
                log_file=log_file,
                structured_results=structured_results,
            )
            if os.path.exists(pending_run_file):
                os.remove(pending_run_file)

        if block == "2" and gate_action:
            for fp in (gate_file, gate_answer_file, gate_question_file):
                if os.path.exists(fp):
                    os.remove(fp)
            return 0

        if block == "2":
            review_question = build_review_question_structured(
                structured_results, confidence_threshold
            )
            for result in structured_results:
                if result is None:
                    continue
                header = (
                    f"\n=== {result.backend.upper()} REVIEW "
                    f"(Pass {result.pass_number}: {result.pass_type}) ===\n"
                )
                sys.stderr.write(header)
                if result.findings:
                    for finding in result.findings:
                        sys.stderr.write(format_finding_for_display(finding) + "\n\n")
                else:
                    sys.stderr.write("No findings.\n")
                sys.stderr.write(f"Blockers: {result.blockers_summary}\n")
                sys.stderr.write(f"Notes: {result.notes_summary}\n")
            structured_json = os.path.join(run_dir, "review.structured.json")
            if os.path.isfile(structured_json):
                sys.stderr.write(f"\nStructured output: {structured_json}\n")
            sys.stderr.write(
                "\nACTION REQUIRED: Reply with 'review:apply' to act on this feedback, "
                "or 'review:skip' to skip reviews for the rest of this session. "
                "You can re-enable with 'review:enable' or 'review:resume'.\n"
            )
            sys.stderr.write(
                "\nUSER REQUEST (MANDATORY): Call AskUserQuestion now with this question: "
                f'"{review_question}" '
                "This is a user request. Do not proceed until the user answers.\n"
            )
            return 2

        if block == "1":
            blocking_found = any(
                has_blocking_findings(r, confidence_threshold)
                for r in structured_results if r
            )
            if blocking_found:
                for result in structured_results:
                    if result is None:
                        continue
                    blocking = get_blocking_findings(result, confidence_threshold)
                    for finding in blocking:
                        sys.stderr.write(format_finding_for_display(finding) + "\n")
                return 2

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
