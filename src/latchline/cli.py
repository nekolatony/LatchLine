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


@dataclass
class DiffLine:
    """A single line in a diff hunk."""
    line_type: str  # context|add|remove
    content: str
    old_line: int | None = None
    new_line: int | None = None
    finding_ids: list[str] = field(default_factory=list)


@dataclass
class DiffHunk:
    """A single hunk in a unified diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class AnnotatedDiff:
    """A complete diff with findings mapped to lines."""
    hunks: list[DiffHunk] = field(default_factory=list)
    finding_line_map: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    unlocated_findings: list[Finding] = field(default_factory=list)


@dataclass
class Symbol:
    """A code symbol (function, class, constant)."""
    name: str
    kind: str  # function|class|method|constant
    file_path: str
    line_start: int
    line_end: int | None = None
    signature: str | None = None
    parent: str | None = None


@dataclass
class SymbolChange:
    """A change to a symbol detected from diff."""
    symbol: Symbol
    change_type: str  # modified|added|removed|signature_changed
    old_signature: str | None = None
    new_signature: str | None = None
    breaking: bool = False
    breaking_reason: str | None = None


@dataclass
class ImportEdge:
    """An import relationship: importer imports from importee."""
    importer: str
    importee: str
    symbols: list[str]
    line: int


@dataclass
class Dependent:
    """A file that depends on a changed file."""
    file_path: str
    line: int
    symbol_used: str
    import_type: str  # direct|transitive
    confidence: int


@dataclass
class DependencyGraph:
    """Cached reverse dependency graph."""
    project_root: str
    cache_version: str = "1.0"
    created_at: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)
    forward_edges: dict[str, list[ImportEdge]] = field(default_factory=dict)
    reverse_edges: dict[str, list[ImportEdge]] = field(default_factory=dict)


@dataclass
class ImpactReport:
    """Summary of cross-file impact."""
    changed_symbols: list[SymbolChange] = field(default_factory=list)
    dependents: list[Dependent] = field(default_factory=list)
    breaking_changes: list[SymbolChange] = field(default_factory=list)
    transitive_depth: int = 0


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


def get_rules_paths(
    changed_files: list[str],
    project_root: str,
) -> list[tuple[str, str]]:
    """Get list of (rules_path, label) tuples for all applicable rules files.
    Returns paths in order: global → project → directories (deduplicated).
    """
    paths: list[tuple[str, str]] = []
    seen_dirs: set[str] = set()
    global_rules = os.path.expanduser("~/.latchline/rules.md")
    if os.path.isfile(global_rules):
        paths.append((global_rules, "Global Rules (~/.latchline/rules.md)"))
    if project_root:
        project_rules = os.path.join(project_root, ".latchline", "rules.md")
        if os.path.isfile(project_rules):
            paths.append((project_rules, "Project Rules"))
            seen_dirs.add(os.path.dirname(project_rules))
    if project_root:
        for changed_file in changed_files:
            if not changed_file:
                continue
            file_dir = os.path.dirname(changed_file)
            for directory in iter_dirs_to_root(file_dir, project_root):
                rules_dir = os.path.join(directory, ".latchline")
                if rules_dir in seen_dirs:
                    continue
                seen_dirs.add(rules_dir)
                rules_path = os.path.join(rules_dir, "rules.md")
                if os.path.isfile(rules_path):
                    rel_dir = os.path.relpath(directory, project_root)
                    if rel_dir == ".":
                        continue  # Already included as project rules
                    label = f"Module Rules ({rel_dir})"
                    paths.append((rules_path, label))
    return paths


def collect_custom_rules(
    changed_files: list[str],
    project_root: str,
    config: dict[str, str],
) -> str:
    """Collect and concatenate rules.md files from all applicable locations.
    Returns concatenated rules with section headers, or empty string if disabled/none found.
    """
    enabled = parse_bool(
        config.get("REVIEWER_CUSTOM_RULES") or os.environ.get("REVIEWER_CUSTOM_RULES"),
        True,
    )
    if not enabled:
        return ""
    rules_paths = get_rules_paths(changed_files, project_root)
    if not rules_paths:
        return ""
    sections: list[str] = []
    for rules_path, label in rules_paths:
        content = read_text(rules_path).strip()
        if content:
            sections.append(f"=== {label} ===\n\n{content}")
    if not sections:
        return ""
    return "\n\n".join(sections)


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


# --- Impact Analysis Functions ---

def get_cache_path(project_root: str, log_dir: str) -> str:
    """Get path to dependency graph cache: $log_dir/cache/depgraph-<hash>.json"""
    import hashlib
    project_hash = hashlib.md5(project_root.encode()).hexdigest()[:12]
    cache_dir = os.path.join(log_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"depgraph-{project_hash}.json")


def file_hash(path: str) -> str:
    """Quick hash (mtime:size) for cache invalidation."""
    try:
        stat = os.stat(path)
        return f"{stat.st_mtime}:{stat.st_size}"
    except OSError:
        return ""


def load_dependency_graph(cache_path: str) -> DependencyGraph | None:
    """Load cached dependency graph from disk."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        graph = DependencyGraph(project_root=data.get("project_root", ""))
        graph.cache_version = data.get("cache_version", "1.0")
        graph.created_at = data.get("created_at", "")
        graph.file_hashes = data.get("file_hashes", {})
        for path, edges_data in data.get("forward_edges", {}).items():
            graph.forward_edges[path] = [
                ImportEdge(
                    importer=e["importer"],
                    importee=e["importee"],
                    symbols=e["symbols"],
                    line=e["line"],
                )
                for e in edges_data
            ]
        for path, edges_data in data.get("reverse_edges", {}).items():
            graph.reverse_edges[path] = [
                ImportEdge(
                    importer=e["importer"],
                    importee=e["importee"],
                    symbols=e["symbols"],
                    line=e["line"],
                )
                for e in edges_data
            ]
        return graph
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_dependency_graph(graph: DependencyGraph, cache_path: str) -> None:
    """Save dependency graph to disk."""
    data = {
        "project_root": graph.project_root,
        "cache_version": graph.cache_version,
        "created_at": graph.created_at,
        "file_hashes": graph.file_hashes,
        "forward_edges": {
            path: [asdict(e) for e in edges]
            for path, edges in graph.forward_edges.items()
        },
        "reverse_edges": {
            path: [asdict(e) for e in edges]
            for path, edges in graph.reverse_edges.items()
        },
    }
    ensure_parent(cache_path)
    write_text(cache_path, json.dumps(data, indent=2) + "\n")


def extract_python_import_edges(path: str, text: str, project_root: str) -> list[ImportEdge]:
    """Extract import edges from a Python file using AST."""
    edges: list[ImportEdge] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return edges
    file_dir = os.path.dirname(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = resolve_python_module(alias.name, 0, file_dir, project_root)
                for target in resolved:
                    edges.append(ImportEdge(
                        importer=path,
                        importee=target,
                        symbols=[alias.name.split(".")[-1]],
                        line=node.lineno,
                    ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            resolved = resolve_python_module(module, level, file_dir, project_root)
            symbols = [alias.name for alias in node.names]
            for target in resolved:
                edges.append(ImportEdge(
                    importer=path,
                    importee=target,
                    symbols=symbols,
                    line=node.lineno,
                ))
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                subresolved = resolve_python_module(name, level, file_dir, project_root)
                for target in subresolved:
                    if target not in resolved:
                        edges.append(ImportEdge(
                            importer=path,
                            importee=target,
                            symbols=[alias.name],
                            line=node.lineno,
                        ))
    return edges


def extract_js_import_edges(path: str, text: str, project_root: str) -> list[ImportEdge]:
    """Extract import edges from a JS/TS file using regex."""
    edges: list[ImportEdge] = []
    file_dir = os.path.dirname(path)
    exts = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json"]
    import_pattern = re.compile(
        r"(?:import\s+(?:(\{[^}]+\})|(\*\s+as\s+\w+)|(\w+))"
        r"(?:\s*,\s*(?:(\{[^}]+\})|(\*\s+as\s+\w+)))?"
        r"\s+from\s+)?['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )
    for line_num, line in enumerate(text.splitlines(), 1):
        for match in import_pattern.finditer(line):
            named_imports = match.group(1) or match.group(4)
            star_import = match.group(2) or match.group(5)
            default_import = match.group(3)
            module_path = match.group(6)
            if not module_path.startswith((".", "/")):
                continue
            symbols: list[str] = []
            if named_imports:
                symbols = [s.strip().split(" as ")[0].strip()
                          for s in named_imports.strip("{}").split(",") if s.strip()]
            if star_import:
                symbols.append("*")
            if default_import:
                symbols.append("default")
            if not symbols:
                symbols = ["*"]
            base = (
                os.path.join(project_root, module_path.lstrip("/"))
                if module_path.startswith("/")
                else os.path.join(file_dir, module_path)
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
                    edges.append(ImportEdge(
                        importer=path,
                        importee=os.path.abspath(candidate),
                        symbols=symbols,
                        line=line_num,
                    ))
                    break
    return edges


def extract_import_edges(path: str, project_root: str, ext: str) -> list[ImportEdge]:
    """Extract import edges from a file based on extension."""
    text = read_text(path)
    if not text:
        return []
    if ext == ".py":
        return extract_python_import_edges(path, text, project_root)
    if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return extract_js_import_edges(path, text, project_root)
    return []


def build_reverse_graph(project_root: str, log_dir: str) -> DependencyGraph:
    """Build/update reverse dependency graph with incremental caching."""
    cache_path = get_cache_path(project_root, log_dir)
    graph = load_dependency_graph(cache_path)
    if graph is None or graph.project_root != project_root:
        graph = DependencyGraph(project_root=project_root)
    skip_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}
    scan_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"}
    files_to_scan: list[str] = []
    current_hashes: dict[str, str] = {}
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in scan_exts:
                continue
            path = os.path.abspath(os.path.join(root, name))
            h = file_hash(path)
            current_hashes[path] = h
            if graph.file_hashes.get(path) != h:
                files_to_scan.append(path)
    removed = set(graph.file_hashes.keys()) - set(current_hashes.keys())
    for path in removed:
        graph.forward_edges.pop(path, None)
        for importee, edges in list(graph.reverse_edges.items()):
            graph.reverse_edges[importee] = [e for e in edges if e.importer != path]
    for path in files_to_scan:
        ext = os.path.splitext(path)[1].lower()
        old_edges = graph.forward_edges.get(path, [])
        for edge in old_edges:
            if edge.importee in graph.reverse_edges:
                graph.reverse_edges[edge.importee] = [
                    e for e in graph.reverse_edges[edge.importee]
                    if e.importer != path
                ]
        new_edges = extract_import_edges(path, project_root, ext)
        graph.forward_edges[path] = new_edges
        for edge in new_edges:
            if edge.importee not in graph.reverse_edges:
                graph.reverse_edges[edge.importee] = []
            graph.reverse_edges[edge.importee].append(edge)
    graph.file_hashes = current_hashes
    graph.created_at = utc_now()
    save_dependency_graph(graph, cache_path)
    return graph


def extract_python_symbols(path: str, text: str) -> list[Symbol]:
    """AST-based extraction of functions, classes with signatures."""
    symbols: list[Symbol] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                if arg.annotation:
                    try:
                        arg_name += f": {ast.unparse(arg.annotation)}"
                    except (AttributeError, ValueError):
                        pass
                args.append(arg_name)
            for arg in node.args.kwonlyargs:
                arg_name = arg.arg
                if arg.annotation:
                    try:
                        arg_name += f": {ast.unparse(arg.annotation)}"
                    except (AttributeError, ValueError):
                        pass
                args.append(arg_name)
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            sig = f"def {node.name}({', '.join(args)})"
            end_line = node.end_lineno if hasattr(node, "end_lineno") else None
            parent = None
            for parent_node in ast.walk(tree):
                if isinstance(parent_node, ast.ClassDef):
                    for child in ast.iter_child_nodes(parent_node):
                        if child is node:
                            parent = parent_node.name
                            break
            symbols.append(Symbol(
                name=node.name,
                kind="method" if parent else "function",
                file_path=path,
                line_start=node.lineno,
                line_end=end_line,
                signature=sig,
                parent=parent,
            ))
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except (AttributeError, ValueError):
                    pass
            sig = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
            end_line = node.end_lineno if hasattr(node, "end_lineno") else None
            symbols.append(Symbol(
                name=node.name,
                kind="class",
                file_path=path,
                line_start=node.lineno,
                line_end=end_line,
                signature=sig,
            ))
        elif isinstance(node, ast.Assign):
            if node.col_offset == 0:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        symbols.append(Symbol(
                            name=target.id,
                            kind="constant",
                            file_path=path,
                            line_start=node.lineno,
                        ))
    return symbols


def extract_js_symbols(path: str, text: str) -> list[Symbol]:
    """Regex-based extraction for JS/TS."""
    symbols: list[Symbol] = []
    func_re = re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(\([^)]*\))",
        re.MULTILINE,
    )
    arrow_re = re.compile(
        r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(\([^)]*\)|\w+)\s*=>",
        re.MULTILINE,
    )
    class_re = re.compile(
        r"^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?",
        re.MULTILINE,
    )
    const_re = re.compile(
        r"^(?:export\s+)?const\s+([A-Z][A-Z0-9_]*)\s*=",
        re.MULTILINE,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for match in func_re.finditer(line):
            symbols.append(Symbol(
                name=match.group(1),
                kind="function",
                file_path=path,
                line_start=i,
                signature=f"function {match.group(1)}{match.group(2)}",
            ))
        for match in arrow_re.finditer(line):
            params = match.group(2)
            if not params.startswith("("):
                params = f"({params})"
            symbols.append(Symbol(
                name=match.group(1),
                kind="function",
                file_path=path,
                line_start=i,
                signature=f"const {match.group(1)} = {params} =>",
            ))
        for match in class_re.finditer(line):
            base = match.group(2)
            sig = f"class {match.group(1)}" + (f" extends {base}" if base else "")
            symbols.append(Symbol(
                name=match.group(1),
                kind="class",
                file_path=path,
                line_start=i,
                signature=sig,
            ))
        for match in const_re.finditer(line):
            symbols.append(Symbol(
                name=match.group(1),
                kind="constant",
                file_path=path,
                line_start=i,
            ))
    return symbols


def extract_symbols_from_file(path: str) -> list[Symbol]:
    """Extract symbols from a file based on extension."""
    text = read_text(path)
    if not text:
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return extract_python_symbols(path, text)
    if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return extract_js_symbols(path, text)
    return []


def extract_changed_symbols_from_diff(diff_text: str, project_root: str) -> list[SymbolChange]:
    """Parse diff to find which symbols were modified."""
    changes: list[SymbolChange] = []
    hunks = parse_unified_diff(diff_text)
    for hunk in hunks:
        file_path = hunk.file_path
        if not file_path.startswith("/"):
            file_path = os.path.join(project_root, file_path)
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            continue
        symbols = extract_symbols_from_file(file_path)
        modified_lines = set()
        added_lines = set()
        for line in hunk.lines:
            if line.line_type == "add" and line.new_line:
                added_lines.add(line.new_line)
                modified_lines.add(line.new_line)
            elif line.line_type == "remove" and line.old_line:
                modified_lines.add(line.old_line)
        for symbol in symbols:
            start = symbol.line_start
            end = symbol.line_end or start
            if any(start <= ln <= end for ln in modified_lines):
                change_type = "modified"
                if any(start <= ln <= end for ln in added_lines):
                    change_type = "added" if start in added_lines else "modified"
                changes.append(SymbolChange(
                    symbol=symbol,
                    change_type=change_type,
                ))
    seen = set()
    unique_changes: list[SymbolChange] = []
    for change in changes:
        key = (change.symbol.file_path, change.symbol.name, change.symbol.kind)
        if key not in seen:
            seen.add(key)
            unique_changes.append(change)
    return unique_changes


def compare_signatures(old_sig: str, new_sig: str) -> tuple[bool, str | None]:
    """Returns (is_breaking, reason)."""
    if old_sig == new_sig:
        return False, None
    old_match = re.search(r"\(([^)]*)\)", old_sig)
    new_match = re.search(r"\(([^)]*)\)", new_sig)
    if not old_match or not new_match:
        return False, None
    old_args = [a.strip() for a in old_match.group(1).split(",") if a.strip()]
    new_args = [a.strip() for a in new_match.group(1).split(",") if a.strip()]
    old_required = [a for a in old_args if "=" not in a and not a.startswith("*")]
    new_required = [a for a in new_args if "=" not in a and not a.startswith("*")]
    if len(new_required) > len(old_required):
        return True, "Added required arguments"
    old_names = {a.split(":")[0].split("=")[0].strip() for a in old_args}
    new_names = {a.split(":")[0].split("=")[0].strip() for a in new_args}
    removed = old_names - new_names
    if removed:
        return True, f"Removed arguments: {', '.join(sorted(removed))}"
    return False, None


def detect_breaking_changes(
    before_path: str, after_path: str, project_root: str
) -> list[SymbolChange]:
    """Compare before/after to detect breaking changes."""
    changes: list[SymbolChange] = []
    before_text = read_text(before_path)
    after_text = read_text(after_path) if os.path.isfile(after_path) else ""
    ext = os.path.splitext(before_path)[1].lower()
    if ext == ".py":
        before_symbols = extract_python_symbols(before_path, before_text)
        after_symbols = extract_python_symbols(after_path, after_text) if after_text else []
    elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        before_symbols = extract_js_symbols(before_path, before_text)
        after_symbols = extract_js_symbols(after_path, after_text) if after_text else []
    else:
        return changes
    after_by_name: dict[str, Symbol] = {s.name: s for s in after_symbols}
    for before_sym in before_symbols:
        if before_sym.kind not in {"function", "method", "class"}:
            continue
        after_sym = after_by_name.get(before_sym.name)
        if after_sym is None:
            changes.append(SymbolChange(
                symbol=before_sym,
                change_type="removed",
                old_signature=before_sym.signature,
                breaking=True,
                breaking_reason="Symbol removed",
            ))
        elif before_sym.signature and after_sym.signature:
            is_breaking, reason = compare_signatures(
                before_sym.signature, after_sym.signature
            )
            if is_breaking:
                changes.append(SymbolChange(
                    symbol=after_sym,
                    change_type="signature_changed",
                    old_signature=before_sym.signature,
                    new_signature=after_sym.signature,
                    breaking=True,
                    breaking_reason=reason,
                ))
    return changes


def find_dependents(
    graph: DependencyGraph,
    changed_files: list[str],
    max_dependents: int,
    depth: int,
) -> list[Dependent]:
    """Find files that depend on the changed files."""
    dependents: list[Dependent] = []
    seen: set[str] = set()
    frontier = [(path, "direct", 0) for path in changed_files]
    while frontier and len(dependents) < max_dependents:
        path, import_type, current_depth = frontier.pop(0)
        path = os.path.abspath(path)
        edges = graph.reverse_edges.get(path, [])
        for edge in edges:
            if edge.importer in seen:
                continue
            seen.add(edge.importer)
            for symbol in edge.symbols:
                dependents.append(Dependent(
                    file_path=edge.importer,
                    line=edge.line,
                    symbol_used=symbol,
                    import_type=import_type,
                    confidence=90 if import_type == "direct" else 70,
                ))
            if current_depth < depth:
                frontier.append((edge.importer, "transitive", current_depth + 1))
            if len(dependents) >= max_dependents:
                break
    return dependents


def analyze_cross_file_impact(
    *,
    run_dir: str,
    project_root: str,
    log_dir: str,
    changed_files: list[str],
    diff_text: str,
    config: dict[str, str],
) -> ImpactReport:
    """Main entry point for impact analysis."""
    enabled = parse_bool(
        config.get("REVIEWER_IMPACT_ANALYSIS") or os.environ.get("REVIEWER_IMPACT_ANALYSIS"),
        False,
    )
    if not enabled:
        return ImpactReport()
    max_dependents = parse_int(
        config.get("REVIEWER_IMPACT_MAX_DEPENDENTS")
        or os.environ.get("REVIEWER_IMPACT_MAX_DEPENDENTS"),
        20,
    )
    depth = parse_int(
        config.get("REVIEWER_IMPACT_DEPTH") or os.environ.get("REVIEWER_IMPACT_DEPTH"),
        1,
    )
    report = ImpactReport(transitive_depth=depth)
    graph = build_reverse_graph(project_root, log_dir)
    changed_symbols = extract_changed_symbols_from_diff(diff_text, project_root)
    report.changed_symbols = changed_symbols
    abs_changed = [os.path.abspath(f) for f in changed_files if f]
    report.dependents = find_dependents(graph, abs_changed, max_dependents, depth)
    before_dir = os.path.join(run_dir, "before")
    for path in abs_changed:
        before_path = os.path.join(before_dir, path.lstrip("/"))
        if os.path.isfile(before_path):
            breaking = detect_breaking_changes(before_path, path, project_root)
            report.breaking_changes.extend(breaking)
    return report


def create_impact_findings(report: ImpactReport) -> list[Finding]:
    """Convert ImpactReport to Finding objects."""
    findings: list[Finding] = []
    for i, change in enumerate(report.breaking_changes):
        findings.append(Finding(
            id=f"IMP{i+1:03d}",
            severity="high",
            category="impact",
            confidence=85,
            title=f"Breaking change: {change.symbol.name}",
            description=(
                f"{change.breaking_reason}. "
                f"Old: {change.old_signature or 'N/A'}. "
                f"New: {change.new_signature or 'removed'}."
            ),
            file_path=change.symbol.file_path,
            line_start=change.symbol.line_start,
            line_end=change.symbol.line_end,
        ))
    if report.dependents and len(report.dependents) >= 5:
        dep_files = list({d.file_path for d in report.dependents})[:5]
        findings.append(Finding(
            id=f"IMP{len(report.breaking_changes)+1:03d}",
            severity="medium",
            category="impact",
            confidence=75,
            title=f"{len(report.dependents)} files depend on changed code",
            description=(
                f"Changed files are imported by: {', '.join(dep_files)}"
                + (f" and {len(dep_files) - 5} more" if len(dep_files) > 5 else "")
            ),
        ))
    return findings


def write_impact_report(run_dir: str, report: ImpactReport) -> None:
    """Write impact.json and impact.md to run directory."""
    def symbol_to_dict(s: Symbol) -> dict[str, Any]:
        return {
            "name": s.name,
            "kind": s.kind,
            "file_path": s.file_path,
            "line_start": s.line_start,
            "line_end": s.line_end,
            "signature": s.signature,
            "parent": s.parent,
        }
    json_data = {
        "version": "1.0",
        "generated_at": utc_now(),
        "changed_symbols": [
            {
                "symbol": symbol_to_dict(c.symbol),
                "change_type": c.change_type,
                "old_signature": c.old_signature,
                "new_signature": c.new_signature,
                "breaking": c.breaking,
                "breaking_reason": c.breaking_reason,
            }
            for c in report.changed_symbols
        ],
        "dependents": [asdict(d) for d in report.dependents],
        "breaking_changes": [
            {
                "symbol": symbol_to_dict(c.symbol),
                "change_type": c.change_type,
                "old_signature": c.old_signature,
                "new_signature": c.new_signature,
                "breaking": c.breaking,
                "breaking_reason": c.breaking_reason,
            }
            for c in report.breaking_changes
        ],
        "transitive_depth": report.transitive_depth,
    }
    json_path = os.path.join(run_dir, "impact.json")
    write_text(json_path, json.dumps(json_data, indent=2) + "\n")
    md_lines = ["# Impact Analysis Report", "", f"Generated: {utc_now()}", ""]
    if report.breaking_changes:
        md_lines.append("## Breaking Changes")
        md_lines.append("")
        for change in report.breaking_changes:
            md_lines.append(f"- **{change.symbol.name}** ({change.symbol.kind})")
            md_lines.append(f"  - Reason: {change.breaking_reason}")
            if change.old_signature:
                md_lines.append(f"  - Old: `{change.old_signature}`")
            if change.new_signature:
                md_lines.append(f"  - New: `{change.new_signature}`")
            md_lines.append(f"  - Location: {change.symbol.file_path}:{change.symbol.line_start}")
        md_lines.append("")
    if report.changed_symbols:
        md_lines.append("## Changed Symbols")
        md_lines.append("")
        for change in report.changed_symbols:
            md_lines.append(
                f"- {change.symbol.name} ({change.symbol.kind}) - {change.change_type}"
            )
        md_lines.append("")
    if report.dependents:
        md_lines.append("## Dependent Files")
        md_lines.append("")
        for dep in report.dependents:
            md_lines.append(
                f"- {dep.file_path}:{dep.line} uses `{dep.symbol_used}` ({dep.import_type})"
            )
        md_lines.append("")
    md_path = os.path.join(run_dir, "impact.md")
    write_text(md_path, "\n".join(md_lines))


def build_context_bundle(
    *,
    run_dir: str,
    project_root: str,
    cwd_path: str,
    changed_files: list[str],
    config: dict[str, str],
    impact_report: ImpactReport | None = None,
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
    include_dependents = parse_bool(
        config.get("REVIEWER_INCLUDE_DEPENDENTS")
        or os.environ.get("REVIEWER_INCLUDE_DEPENDENTS"),
        False,
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
    dependent_files: list[str] = []
    if include_dependents and impact_report:
        dependent_files = [d.file_path for d in impact_report.dependents if os.path.isfile(d.file_path)]

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
    add_paths(dependent_files)

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
{custom_rules}
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


def build_structured_review_prompt(pass_type: str, custom_rules: str = "") -> str:
    """Build prompt requesting JSON output with findings schema."""
    rules_section = f"\n\n{custom_rules}\n" if custom_rules else ""
    if pass_type == "smoke":
        return STRUCTURED_REVIEW_TEMPLATE.format(pass_focus=SMOKE_FOCUS, custom_rules=rules_section)
    return STRUCTURED_REVIEW_TEMPLATE.format(pass_focus=SEMANTIC_FOCUS, custom_rules=rules_section)


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
            "missing_test", "style", "performance", "impact"
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


def write_structured_json(
    run_dir: str,
    results: list[ReviewResult],
    annotated_diff: AnnotatedDiff | None = None,
) -> str:
    """Write review.structured.json with all findings and diff mapping."""
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
    output: dict[str, Any] = {
        "version": "1.0",
        "generated_at": utc_now(),
        "passes": passes,
        "all_findings": all_findings,
    }
    if annotated_diff:
        output["finding_line_map"] = annotated_diff.finding_line_map
        output["unlocated_finding_ids"] = [
            f.id for f in annotated_diff.unlocated_findings
        ]
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


# Diff highlighting regex patterns
DIFF_FILE_HEADER_RE = re.compile(r'^diff --git a/(.+) b/(.+)$')
DIFF_HUNK_HEADER_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    """Parse unified diff format into structured DiffHunk objects."""
    hunks: list[DiffHunk] = []
    current_file: str = ""
    current_hunk: DiffHunk | None = None
    old_line = 0
    new_line = 0
    for line in diff_text.splitlines():
        file_match = DIFF_FILE_HEADER_RE.match(line)
        if file_match:
            current_file = file_match.group(2)
            continue
        if line.startswith('--- ') or line.startswith('+++ '):
            if line.startswith('+++ '):
                path = line[4:].strip()
                # Strip LatchLine's (before)/(after) labels
                for suffix in (' (after)', ' (before)', ' (deleted)'):
                    if path.endswith(suffix):
                        path = path[:-len(suffix)]
                        break
                if path.startswith('b/'):
                    current_file = path[2:]
                elif path != '/dev/null':
                    current_file = path
            continue
        hunk_match = DIFF_HUNK_HEADER_RE.match(line)
        if hunk_match:
            if current_hunk:
                hunks.append(current_hunk)
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
            current_hunk = DiffHunk(
                file_path=current_file,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header=line,
            )
            old_line = old_start
            new_line = new_start
            continue
        if current_hunk is None:
            continue
        if line.startswith('+') and not line.startswith('+++'):
            diff_line = DiffLine(
                line_type="add",
                content=line[1:],
                old_line=None,
                new_line=new_line,
            )
            current_hunk.lines.append(diff_line)
            new_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            diff_line = DiffLine(
                line_type="remove",
                content=line[1:],
                old_line=old_line,
                new_line=None,
            )
            current_hunk.lines.append(diff_line)
            old_line += 1
        elif line.startswith(' ') or line == '':
            content = line[1:] if line.startswith(' ') else ''
            diff_line = DiffLine(
                line_type="context",
                content=content,
                old_line=old_line,
                new_line=new_line,
            )
            current_hunk.lines.append(diff_line)
            old_line += 1
            new_line += 1
    if current_hunk:
        hunks.append(current_hunk)
    return hunks


def map_findings_to_diff(
    findings: list[Finding], hunks: list[DiffHunk]
) -> AnnotatedDiff:
    """Map findings to diff lines based on file_path and line numbers."""
    annotated = AnnotatedDiff(hunks=hunks)
    for finding in findings:
        if not finding.file_path or not finding.line_start:
            annotated.unlocated_findings.append(finding)
            continue
        located = False
        line_end = finding.line_end or finding.line_start
        for hunk in hunks:
            if not _paths_match(finding.file_path, hunk.file_path):
                continue
            for diff_line in hunk.lines:
                if diff_line.new_line is None:
                    continue
                if finding.line_start <= diff_line.new_line <= line_end:
                    diff_line.finding_ids.append(finding.id)
                    if finding.id not in annotated.finding_line_map:
                        annotated.finding_line_map[finding.id] = []
                    annotated.finding_line_map[finding.id].append(
                        (hunk.file_path, diff_line.new_line)
                    )
                    located = True
        if not located:
            annotated.unlocated_findings.append(finding)
    return annotated


def _paths_match(finding_path: str, hunk_path: str) -> bool:
    """Check if a finding path matches a hunk path (handles relative paths)."""
    if finding_path == hunk_path:
        return True
    if finding_path.endswith(hunk_path) or hunk_path.endswith(finding_path):
        return True
    finding_parts = finding_path.replace('\\', '/').split('/')
    hunk_parts = hunk_path.replace('\\', '/').split('/')
    min_len = min(len(finding_parts), len(hunk_parts))
    return finding_parts[-min_len:] == hunk_parts[-min_len:]


def get_severity_color(severity: str) -> tuple[str, str]:
    """Get ANSI color codes for severity level. Returns (start, end)."""
    colors = {
        "critical": ("\033[41;37m", "\033[0m"),  # White on red bg
        "high": ("\033[91m", "\033[0m"),          # Bright red
        "medium": ("\033[93m", "\033[0m"),        # Yellow
        "low": ("\033[94m", "\033[0m"),           # Blue
        "info": ("\033[90m", "\033[0m"),          # Gray
    }
    return colors.get(severity, ("", ""))


def render_annotated_diff(
    annotated: AnnotatedDiff,
    findings: list[Finding],
    use_color: bool = True,
) -> str:
    """Render annotated diff with findings highlighted."""
    finding_map = {f.id: f for f in findings}
    lines: list[str] = []
    c_add = "\033[32m" if use_color else ""
    c_rem = "\033[31m" if use_color else ""
    c_hdr = "\033[36m" if use_color else ""
    c_ann = "\033[33m" if use_color else ""
    c_end = "\033[0m" if use_color else ""
    for hunk in annotated.hunks:
        lines.append(f"{c_hdr}diff --git a/{hunk.file_path} b/{hunk.file_path}{c_end}")
        lines.append(f"{c_hdr}{hunk.header}{c_end}")
        for diff_line in hunk.lines:
            prefix = " "
            color = ""
            if diff_line.line_type == "add":
                prefix = "+"
                color = c_add
            elif diff_line.line_type == "remove":
                prefix = "-"
                color = c_rem
            line_content = f"{color}{prefix}{diff_line.content}{c_end}"
            if diff_line.finding_ids:
                annotations = []
                for fid in diff_line.finding_ids:
                    f = finding_map.get(fid)
                    if f:
                        sev_start, sev_end = get_severity_color(f.severity)
                        if use_color:
                            ann = f"{sev_start}[{fid}]{sev_end} {f.title}"
                        else:
                            ann = f"[{fid}] {f.title}"
                        annotations.append(ann)
                ann_str = f"  {c_ann}# {'; '.join(annotations)}{c_end}"
                line_content += ann_str
            lines.append(line_content)
    if annotated.unlocated_findings:
        lines.append("")
        lines.append(f"{c_hdr}=== Findings not mapped to diff ==={c_end}")
        for finding in annotated.unlocated_findings:
            sev_start, sev_end = get_severity_color(finding.severity)
            if use_color:
                lines.append(
                    f"{sev_start}[{finding.severity.upper()}]{sev_end} "
                    f"{finding.title}"
                )
            else:
                lines.append(f"[{finding.severity.upper()}] {finding.title}")
            if finding.file_path:
                loc = finding.file_path
                if finding.line_start:
                    loc += f":{finding.line_start}"
                lines.append(f"  Location: {loc}")
    return "\n".join(lines)


def render_annotated_diff_markdown(
    annotated: AnnotatedDiff,
    findings: list[Finding],
) -> str:
    """Render annotated diff as markdown."""
    finding_map = {f.id: f for f in findings}
    lines: list[str] = []
    for hunk in annotated.hunks:
        lines.append(f"### {hunk.file_path}")
        lines.append("")
        lines.append("```diff")
        lines.append(hunk.header)
        for diff_line in hunk.lines:
            prefix = " "
            if diff_line.line_type == "add":
                prefix = "+"
            elif diff_line.line_type == "remove":
                prefix = "-"
            lines.append(f"{prefix}{diff_line.content}")
        lines.append("```")
        findings_in_hunk = set()
        for diff_line in hunk.lines:
            findings_in_hunk.update(diff_line.finding_ids)
        if findings_in_hunk:
            lines.append("")
            lines.append("**Findings in this hunk:**")
            for fid in sorted(findings_in_hunk):
                f = finding_map.get(fid)
                if f:
                    lines.append(
                        f"- **[{f.severity.upper()}]** ({f.confidence}%) "
                        f"{f.title} (line {f.line_start})"
                    )
            lines.append("")
    if annotated.unlocated_findings:
        lines.append("### Findings not mapped to diff")
        lines.append("")
        for finding in annotated.unlocated_findings:
            loc = ""
            if finding.file_path:
                loc = f" - {finding.file_path}"
                if finding.line_start:
                    loc += f":{finding.line_start}"
            lines.append(
                f"- **[{finding.severity.upper()}]** ({finding.confidence}%) "
                f"{finding.title}{loc}"
            )
    return "\n".join(lines)


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
    custom_rules: str = "",
) -> None:
    if not shutil.which("codex"):
        return
    context_text = read_text(context_file) if context_file else ""
    prompt = review_prompt
    if custom_rules:
        prompt += f"\n\n{custom_rules}"
    input_text = f"{prompt}\n\nDIFF:\n{diff}\n"
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
    custom_rules: str = "",
) -> None:
    if not shutil.which("claude"):
        return
    full_prompt = review_prompt
    if custom_rules:
        full_prompt += f"\n\n{custom_rules}"
    prompt = f"Review the diff at {diff_file}."
    if context_file:
        prompt += f" Additional context is available at {context_file}."
    prompt += f" Use the following format:\n{full_prompt}"
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
    custom_rules: str = "",
) -> ReviewResult | None:
    """Run Codex with structured JSON prompt and parse result."""
    if not shutil.which("codex"):
        return None
    prompt = build_structured_review_prompt(pass_type, custom_rules)
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
    custom_rules: str = "",
) -> ReviewResult | None:
    """Run Claude with structured JSON prompt and parse result."""
    if not shutil.which("claude"):
        return None
    review_prompt = build_structured_review_prompt(pass_type, custom_rules)
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
    custom_rules: str = "",
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
            "smoke", diff, smoke_out, base_dir, context_file, project_root, custom_rules
        )
        if smoke_result and has_blocking_findings(smoke_result, confidence_threshold):
            return smoke_result, None
        semantic_out = os.path.join(run_dir, "review.codex.semantic.md")
        semantic_result = run_codex_structured(
            "semantic", diff, semantic_out, base_dir, context_file, project_root, custom_rules
        )
    elif backend == "claude":
        smoke_out = os.path.join(run_dir, "review.claude.smoke.md")
        smoke_result = run_claude_structured(
            "smoke", diff_file, run_dir, smoke_out, base_dir, context_file,
            project_root, custom_rules,
        )
        if smoke_result and has_blocking_findings(smoke_result, confidence_threshold):
            return smoke_result, None
        semantic_out = os.path.join(run_dir, "review.claude.semantic.md")
        semantic_result = run_claude_structured(
            "semantic", diff_file, run_dir, semantic_out,
            base_dir, context_file, project_root, custom_rules
        )
    elif backend == "both":
        smoke_codex_out = os.path.join(run_dir, "review.codex.smoke.md")
        smoke_codex = run_codex_structured(
            "smoke", diff, smoke_codex_out, base_dir, context_file, project_root, custom_rules
        )
        smoke_claude_out = os.path.join(run_dir, "review.claude.smoke.md")
        smoke_claude = run_claude_structured(
            "smoke", diff_file, run_dir, smoke_claude_out,
            base_dir, context_file, project_root, custom_rules
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
            "semantic", diff, semantic_codex_out, base_dir, context_file, project_root, custom_rules
        )
        semantic_claude_out = os.path.join(run_dir, "review.claude.semantic.md")
        semantic_claude = run_claude_structured(
            "semantic", diff_file, run_dir, semantic_claude_out,
            base_dir, context_file, project_root, custom_rules
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
        custom_rules = collect_custom_rules(changed_files, project_root, config)
        impact_report = analyze_cross_file_impact(
            run_dir=run_dir,
            project_root=project_root,
            log_dir=base_dir,
            changed_files=changed_files,
            diff_text=diff,
            config=config,
        )
        context_file, context_files_file = build_context_bundle(
            run_dir=run_dir,
            project_root=project_root,
            cwd_path=cwd_path,
            changed_files=changed_files,
            config=config,
            impact_report=impact_report,
        )

        structured_results: list[ReviewResult] = []
        if multi_pass:
            smoke, semantic = run_multi_pass_review(
                backend, diff, diff_file, run_dir, base_dir,
                context_file, project_root, config, custom_rules,
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
                    custom_rules,
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
                    custom_rules,
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
                    custom_rules,
                )
                run_claude(
                    review_prompt,
                    diff_file,
                    run_dir,
                    claude_out,
                    base_dir,
                    context_file,
                    project_root,
                    custom_rules,
                )
                if os.path.isfile(codex_out):
                    result = parse_structured_output(read_text(codex_out), 1, "codex")
                    structured_results.append(result)
                if os.path.isfile(claude_out):
                    result = parse_structured_output(read_text(claude_out), 1, "claude")
                    structured_results.append(result)
            else:
                return 0

        if impact_report.breaking_changes or impact_report.dependents:
            impact_findings = create_impact_findings(impact_report)
            if structured_results and structured_results[0]:
                structured_results[0].findings.extend(impact_findings)
            write_impact_report(run_dir, impact_report)

        annotated_diff: AnnotatedDiff | None = None
        if structured_results:
            all_findings: list[Finding] = []
            for result in structured_results:
                if result:
                    all_findings.extend(result.findings)
            if all_findings:
                hunks = parse_unified_diff(diff)
                annotated_diff = map_findings_to_diff(all_findings, hunks)
                diff_md_path = os.path.join(run_dir, "review.diff.md")
                diff_md = render_annotated_diff_markdown(annotated_diff, all_findings)
                write_text(diff_md_path, diff_md)
            write_structured_json(run_dir, structured_results, annotated_diff)

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
            if annotated_diff and annotated_diff.finding_line_map:
                use_color = sys.stderr.isatty()
                sys.stderr.write("\n=== ANNOTATED DIFF ===\n")
                highlighted = render_annotated_diff(
                    annotated_diff, all_findings, use_color=use_color
                )
                sys.stderr.write(highlighted + "\n")
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
