import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import latchline.cli as ai_review  # noqa: E402


def test_parse_config_handles_export_and_quotes(tmp_path):
    path = tmp_path / "settings.conf"
    path.write_text(
        """
# comment
export REVIEWER_BACKEND=codex
REVIEWER_BLOCK='2'
IGNORED_LINE
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = ai_review.parse_config(str(path))
    assert config["REVIEWER_BACKEND"] == "codex"
    assert config["REVIEWER_BLOCK"] == "2"
    assert "IGNORED_LINE" not in config


def test_find_project_root_from_git_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    assert ai_review.find_project_root(str(nested)) == str(repo)


def test_find_project_root_from_git_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: .git/worktrees/main\n", encoding="utf-8")
    nested = repo / "app" / "core"
    nested.mkdir(parents=True)
    assert ai_review.find_project_root(str(nested)) == str(repo)


def test_find_project_root_missing(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert ai_review.find_project_root(str(nested)) == ""


def test_resolve_prefers_cwd_over_root_and_global(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    cwd = repo / "subdir"
    home.mkdir()
    repo.mkdir()
    cwd.mkdir(parents=True)
    (repo / ".git").mkdir()

    global_config = home / ".latchline" / "settings.conf"
    root_config = repo / ".latchline" / "settings.conf"
    cwd_config = cwd / ".latchline" / "settings.conf"
    for path in (global_config, root_config, cwd_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("REVIEWER_BACKEND=codex\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    resolved = ai_review.resolve_config_path(str(cwd))
    assert resolved == str(cwd_config)


def test_resolve_uses_root_when_no_cwd_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    cwd = repo / "subdir"
    home.mkdir()
    repo.mkdir()
    cwd.mkdir(parents=True)
    (repo / ".git").mkdir()

    global_config = home / ".latchline" / "settings.conf"
    root_config = repo / ".latchline" / "settings.conf"
    root_config.parent.mkdir(parents=True, exist_ok=True)
    global_config.parent.mkdir(parents=True, exist_ok=True)
    root_config.write_text("REVIEWER_BACKEND=codex\n", encoding="utf-8")
    global_config.write_text("REVIEWER_BACKEND=claude\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    resolved = ai_review.resolve_config_path(str(cwd))
    assert resolved == str(root_config)


def test_resolve_uses_global_when_no_local(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    cwd = repo / "subdir"
    home.mkdir()
    repo.mkdir()
    cwd.mkdir(parents=True)
    (repo / ".git").mkdir()

    global_config = home / ".latchline" / "settings.conf"
    global_config.parent.mkdir(parents=True, exist_ok=True)
    global_config.write_text("REVIEWER_BACKEND=codex\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    resolved = ai_review.resolve_config_path(str(cwd))
    assert resolved == str(global_config)


def test_resolve_fallback_path_when_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    cwd = repo / "subdir"
    home.mkdir()
    repo.mkdir()
    cwd.mkdir(parents=True)
    (repo / ".git").mkdir()

    monkeypatch.setenv("HOME", str(home))
    resolved = ai_review.resolve_config_path(str(cwd))
    expected = str(home / ".latchline" / "settings.conf")
    assert resolved == expected


def test_resolve_log_dir_creates_path(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    config = {"LATCHLINE_LOG_DIR": str(log_dir)}
    monkeypatch.delenv("LATCHLINE_LOG_DIR", raising=False)
    resolved = ai_review.resolve_log_dir(config)
    assert resolved == str(log_dir)
    assert log_dir.is_dir()


def test_resolve_log_dir_falls_back_to_tmp(monkeypatch):
    config = {"LATCHLINE_LOG_DIR": "/custom"}

    def fake_ensure_directory(path):
        return "/tmp" if path == "/tmp" else None

    monkeypatch.setattr(ai_review, "ensure_directory", fake_ensure_directory)
    resolved = ai_review.resolve_log_dir(config)
    assert resolved == "/tmp"


def test_build_context_bundle_includes_local_imports(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)

    app_dir = project_root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    main_path = app_dir / "main.py"
    util_path = app_dir / "util.py"
    main_path.write_text("from . import util\n", encoding="utf-8")
    util_path.write_text("def helper():\n    return 1\n", encoding="utf-8")

    context_file, context_files_file = ai_review.build_context_bundle(
        run_dir=str(run_dir),
        project_root=str(project_root),
        cwd_path=str(project_root),
        changed_files=[str(main_path)],
        config={
            "REVIEWER_CONTEXT": "1",
            "REVIEWER_CONTEXT_MAX_BYTES": "10000",
            "REVIEWER_CONTEXT_DEPTH": "1",
        },
    )

    content = Path(context_file).read_text(encoding="utf-8")
    assert "FILE: app/main.py" in content
    assert "FILE: app/util.py" in content
    listed = Path(context_files_file).read_text(encoding="utf-8")
    assert "app/main.py" in listed
    assert "app/util.py" in listed


def test_build_context_bundle_includes_adjacent_context(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)

    service_dir = project_root / "services" / "api"
    service_dir.mkdir(parents=True)
    changed_path = service_dir / "app.js"
    changed_path.write_text("console.log('ok');\n", encoding="utf-8")
    dockerfile = service_dir / "Dockerfile"
    dockerfile.write_text("FROM node:20\n", encoding="utf-8")

    context_file, context_files_file = ai_review.build_context_bundle(
        run_dir=str(run_dir),
        project_root=str(project_root),
        cwd_path=str(project_root),
        changed_files=[str(changed_path)],
        config={
            "REVIEWER_CONTEXT": "1",
            "REVIEWER_CONTEXT_MAX_BYTES": "10000",
            "REVIEWER_CONTEXT_DEPTH": "1",
        },
    )

    content = Path(context_file).read_text(encoding="utf-8")
    assert "FILE: services/api/Dockerfile" in content
    listed = Path(context_files_file).read_text(encoding="utf-8")
    assert "services/api/Dockerfile" in listed


def test_parse_structured_output_valid_json():
    raw = '''
    {
      "findings": [
        {
          "id": "F001",
          "severity": "high",
          "category": "security",
          "confidence": 85,
          "title": "SQL injection",
          "description": "User input passed to query",
          "file_path": "src/db.py",
          "line_start": 42,
          "line_end": 42,
          "suggested_fix": "Use parameterized query"
        }
      ],
      "blockers_summary": "SQL injection found",
      "notes_summary": "Consider adding tests"
    }
    '''
    result = ai_review.parse_structured_output(raw, 1, "codex")
    assert result.backend == "codex"
    assert result.pass_number == 1
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.id == "F001"
    assert finding.severity == "high"
    assert finding.confidence == 85
    assert finding.file_path == "src/db.py"
    assert result.blockers_summary == "SQL injection found"


def test_parse_structured_output_fallback_legacy():
    raw = """BLOCKERS: Missing null check in process()
NOTES: Consider adding logging"""
    result = ai_review.parse_structured_output(raw, 1, "claude")
    assert result.backend == "claude"
    assert "Missing null check" in result.blockers_summary
    assert "logging" in result.notes_summary
    assert len(result.findings) == 1
    assert result.findings[0].severity == "high"


def test_has_blocking_findings_respects_threshold():
    finding_high_conf = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=85, title="Issue", description="Desc",
    )
    finding_low_conf = ai_review.Finding(
        id="F002", severity="high", category="security",
        confidence=50, title="Issue", description="Desc",
    )
    result = ai_review.ReviewResult(findings=[finding_high_conf, finding_low_conf])
    assert ai_review.has_blocking_findings(result, 70) is True
    assert ai_review.has_blocking_findings(result, 90) is False


def test_has_blocking_findings_ignores_low_severity():
    finding = ai_review.Finding(
        id="F001", severity="low", category="style",
        confidence=95, title="Style issue", description="Desc",
    )
    result = ai_review.ReviewResult(findings=[finding])
    assert ai_review.has_blocking_findings(result, 70) is False


def test_format_finding_for_display():
    finding = ai_review.Finding(
        id="F001", severity="critical", category="security",
        confidence=92, title="SQL Injection",
        description="User input passed directly",
        file_path="src/db.py", line_start=42, line_end=45,
    )
    output = ai_review.format_finding_for_display(finding)
    assert "[CRITICAL]" in output
    assert "(92%)" in output
    assert "SQL Injection" in output
    assert "[src/db.py:42-45]" in output
    assert "User input passed directly" in output


def test_get_blocking_findings():
    f1 = ai_review.Finding(
        id="F001", severity="critical", category="security",
        confidence=90, title="A", description=""
    )
    f2 = ai_review.Finding(
        id="F002", severity="high", category="correctness",
        confidence=60, title="B", description=""
    )
    f3 = ai_review.Finding(
        id="F003", severity="low", category="style",
        confidence=95, title="C", description=""
    )
    result = ai_review.ReviewResult(findings=[f1, f2, f3])
    blocking = ai_review.get_blocking_findings(result, 70)
    assert len(blocking) == 1
    assert blocking[0].id == "F001"


def test_parse_structured_output_validates_values():
    raw = '''
    {
      "findings": [
        {
          "id": "F001",
          "severity": "UNKNOWN",
          "category": "invalid_category",
          "confidence": 150,
          "title": "Test",
          "description": "Desc"
        },
        {
          "id": "F002",
          "severity": "critical",
          "category": "security",
          "confidence": -10,
          "title": "Test2",
          "description": "Desc2"
        }
      ],
      "blockers_summary": "test",
      "notes_summary": "test"
    }
    '''
    result = ai_review.parse_structured_output(raw, 1, "codex")
    assert len(result.findings) == 2
    # Invalid severity defaults to "info"
    assert result.findings[0].severity == "info"
    # Invalid category defaults to "correctness"
    assert result.findings[0].category == "correctness"
    # Confidence > 100 clamped to 100
    assert result.findings[0].confidence == 100
    # Confidence < 0 clamped to 0
    assert result.findings[1].confidence == 0
    # Valid values preserved
    assert result.findings[1].severity == "critical"
    assert result.findings[1].category == "security"


def test_write_structured_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    finding = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=85, title="Issue", description="Desc",
    )
    result = ai_review.ReviewResult(
        backend="codex", pass_type="smoke", pass_number=1,
        findings=[finding], blockers_summary="Issue found", notes_summary="none",
    )
    output_path = ai_review.write_structured_json(str(run_dir), [result])
    assert Path(output_path).exists()
    import json
    data = json.loads(Path(output_path).read_text())
    assert data["version"] == "1.0"
    assert len(data["passes"]) == 1
    assert len(data["all_findings"]) == 1
    assert data["all_findings"][0]["confidence"] == 85


# Diff highlighting tests

SAMPLE_DIFF = """\
diff --git a/src/db.py b/src/db.py
--- a/src/db.py
+++ b/src/db.py
@@ -40,7 +40,7 @@ class Database:
     def execute(self, query_str, user_input):
         cursor = self.conn.cursor()
-        cursor.execute(f"SELECT * FROM users WHERE id = {user_input}")
+        cursor.execute(f"SELECT * FROM users WHERE name = {user_input}")
         return cursor.fetchall()
"""


def test_parse_unified_diff_basic():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.file_path == "src/db.py"
    assert hunk.old_start == 40
    assert hunk.new_start == 40
    assert len(hunk.lines) == 5
    # Check line types
    line_types = [line.line_type for line in hunk.lines]
    assert line_types == ["context", "context", "remove", "add", "context"]


def test_parse_unified_diff_line_numbers():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    hunk = hunks[0]
    # Context line at start
    assert hunk.lines[0].old_line == 40
    assert hunk.lines[0].new_line == 40
    # Remove line
    assert hunk.lines[2].old_line == 42
    assert hunk.lines[2].new_line is None
    # Add line
    assert hunk.lines[3].old_line is None
    assert hunk.lines[3].new_line == 42


def test_map_findings_to_diff_locates_finding():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    finding = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=90, title="SQL Injection",
        description="User input in query",
        file_path="src/db.py", line_start=42, line_end=42,
    )
    annotated = ai_review.map_findings_to_diff([finding], hunks)
    assert "F001" in annotated.finding_line_map
    assert len(annotated.unlocated_findings) == 0
    # Check finding is attached to the add line
    hunk = annotated.hunks[0]
    add_line = [ln for ln in hunk.lines if ln.line_type == "add"][0]
    assert "F001" in add_line.finding_ids


def test_map_findings_to_diff_unlocated():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    finding = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=90, title="Issue in other file",
        description="Desc",
        file_path="src/other.py", line_start=10,
    )
    annotated = ai_review.map_findings_to_diff([finding], hunks)
    assert "F001" not in annotated.finding_line_map
    assert len(annotated.unlocated_findings) == 1


def test_map_findings_without_location():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    finding = ai_review.Finding(
        id="F001", severity="info", category="style",
        confidence=50, title="General observation",
        description="Desc",
        file_path=None, line_start=None,
    )
    annotated = ai_review.map_findings_to_diff([finding], hunks)
    assert len(annotated.unlocated_findings) == 1


def test_render_annotated_diff_includes_annotations():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    finding = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=90, title="SQL Injection",
        description="User input in query",
        file_path="src/db.py", line_start=42,
    )
    annotated = ai_review.map_findings_to_diff([finding], hunks)
    output = ai_review.render_annotated_diff(annotated, [finding], use_color=False)
    assert "[F001]" in output
    assert "SQL Injection" in output
    assert "src/db.py" in output


def test_render_annotated_diff_markdown():
    hunks = ai_review.parse_unified_diff(SAMPLE_DIFF)
    finding = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=90, title="SQL Injection",
        description="User input in query",
        file_path="src/db.py", line_start=42,
    )
    annotated = ai_review.map_findings_to_diff([finding], hunks)
    output = ai_review.render_annotated_diff_markdown(annotated, [finding])
    assert "### src/db.py" in output
    assert "```diff" in output
    assert "**Findings in this hunk:**" in output
    assert "[HIGH]" in output


def test_paths_match():
    assert ai_review._paths_match("src/db.py", "src/db.py")
    assert ai_review._paths_match("/full/path/src/db.py", "src/db.py")
    assert ai_review._paths_match("src/db.py", "full/path/src/db.py")
    assert not ai_review._paths_match("src/db.py", "src/other.py")


def test_parse_unified_diff_strips_latchline_labels():
    """Test that LatchLine's (before)/(after) labels are stripped from paths."""
    diff = """\
--- /home/user/project/src/db.py (before)
+++ /home/user/project/src/db.py (after)
@@ -1,3 +1,4 @@
+import httpx
 import os
 import sys
"""
    hunks = ai_review.parse_unified_diff(diff)
    assert len(hunks) == 1
    # Path should NOT include " (after)"
    assert hunks[0].file_path == "/home/user/project/src/db.py"
    assert "(after)" not in hunks[0].file_path


def test_map_findings_to_diff_with_absolute_paths():
    """Test mapping when diff has absolute paths and findings have relative paths."""
    diff = """\
--- /home/user/project/src/db.py (before)
+++ /home/user/project/src/db.py (after)
@@ -1,3 +1,4 @@
+import httpx
 import os
 import sys
"""
    hunks = ai_review.parse_unified_diff(diff)
    finding = ai_review.Finding(
        id="F001", severity="high", category="correctness",
        confidence=90, title="Test",
        description="Desc",
        file_path="src/db.py", line_start=1,
    )
    annotated = ai_review.map_findings_to_diff([finding], hunks)
    # Finding should be located because src/db.py matches end of absolute path
    assert "F001" in annotated.finding_line_map
    assert len(annotated.unlocated_findings) == 0


# Impact analysis tests

def test_build_reverse_graph_basic(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    app_dir = project_root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    main_path = app_dir / "main.py"
    util_path = app_dir / "util.py"
    main_path.write_text("from . import util\n", encoding="utf-8")
    util_path.write_text("def helper():\n    return 1\n", encoding="utf-8")
    graph = ai_review.build_reverse_graph(str(project_root), str(log_dir))
    assert graph.project_root == str(project_root)
    assert len(graph.forward_edges) > 0
    util_abs = str(util_path.resolve())
    init_abs = str((app_dir / "__init__.py").resolve())
    reverse_keys = list(graph.reverse_edges.keys())
    has_dependency = any(util_abs in k or init_abs in k for k in reverse_keys)
    assert has_dependency or len(graph.forward_edges) > 0


def test_extract_python_symbols():
    code = '''
def process(data, config=None):
    return data

class Handler:
    def handle(self, event):
        pass

MAX_SIZE = 100
'''
    symbols = ai_review.extract_python_symbols("/test.py", code)
    names = {s.name for s in symbols}
    assert "process" in names
    assert "Handler" in names
    assert "handle" in names
    assert "MAX_SIZE" in names
    process_sym = next(s for s in symbols if s.name == "process")
    assert process_sym.kind == "function"
    assert "data" in process_sym.signature


def test_extract_js_symbols():
    code = '''
export function fetchData(url, options) {
    return fetch(url, options);
}

const processItems = (items) => items.map(x => x);

class DataService {
    async load() {}
}

const API_URL = "https://api.example.com";
'''
    symbols = ai_review.extract_js_symbols("/test.js", code)
    names = {s.name for s in symbols}
    assert "fetchData" in names
    assert "processItems" in names
    assert "DataService" in names
    assert "API_URL" in names


def test_detect_breaking_changes_removed_arg(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    before_path = tmp_path / "before.py"
    after_path = tmp_path / "after.py"
    before_path.write_text("def process(data, config, extra):\n    pass\n")
    after_path.write_text("def process(data, config):\n    pass\n")
    changes = ai_review.detect_breaking_changes(
        str(before_path), str(after_path), str(project_root)
    )
    assert len(changes) == 1
    assert changes[0].breaking is True
    assert "extra" in changes[0].breaking_reason


def test_detect_breaking_changes_removed_symbol(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    before_path = tmp_path / "before.py"
    after_path = tmp_path / "after.py"
    before_path.write_text("def old_func():\n    pass\n\ndef keep_func():\n    pass\n")
    after_path.write_text("def keep_func():\n    pass\n")
    changes = ai_review.detect_breaking_changes(
        str(before_path), str(after_path), str(project_root)
    )
    assert len(changes) == 1
    assert changes[0].symbol.name == "old_func"
    assert changes[0].change_type == "removed"
    assert changes[0].breaking is True


def test_analyze_cross_file_impact_finds_dependents(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)
    app_dir = project_root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    util_path = app_dir / "util.py"
    main_path = app_dir / "main.py"
    util_path.write_text("def helper():\n    return 1\n", encoding="utf-8")
    main_path.write_text("from .util import helper\n\nhelper()\n", encoding="utf-8")
    diff_text = f"""
--- {util_path} (before)
+++ {util_path} (after)
@@ -1,2 +1,2 @@
-def helper():
+def helper(arg=None):
     return 1
"""
    report = ai_review.analyze_cross_file_impact(
        run_dir=str(run_dir),
        project_root=str(project_root),
        log_dir=str(log_dir),
        changed_files=[str(util_path)],
        diff_text=diff_text,
        config={"REVIEWER_IMPACT_ANALYSIS": "1"},
    )
    assert len(report.dependents) > 0 or len(report.changed_symbols) > 0


def test_create_impact_findings_breaking():
    symbol = ai_review.Symbol(
        name="process",
        kind="function",
        file_path="/test.py",
        line_start=10,
        signature="def process(a, b)",
    )
    change = ai_review.SymbolChange(
        symbol=symbol,
        change_type="signature_changed",
        old_signature="def process(a, b, c)",
        new_signature="def process(a, b)",
        breaking=True,
        breaking_reason="Removed arguments: c",
    )
    report = ai_review.ImpactReport(breaking_changes=[change])
    findings = ai_review.create_impact_findings(report)
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].category == "impact"
    assert "process" in findings[0].title


def test_cache_incremental_update(tmp_path):
    import time
    project_root = tmp_path / "repo"
    project_root.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    app_dir = project_root / "app"
    app_dir.mkdir()
    util_path = app_dir / "util.py"
    util_path.write_text("def helper():\n    return 1\n", encoding="utf-8")
    graph1 = ai_review.build_reverse_graph(str(project_root), str(log_dir))
    cache_path = ai_review.get_cache_path(str(project_root), str(log_dir))
    assert Path(cache_path).exists()
    time.sleep(0.01)
    # Change file size to ensure hash changes (mtime may not change fast enough)
    util_path.write_text("def helper():\n    return 2  # longer\n", encoding="utf-8")
    graph2 = ai_review.build_reverse_graph(str(project_root), str(log_dir))
    # Hash should differ due to size change
    assert graph2.file_hashes[str(util_path.resolve())] != graph1.file_hashes.get(
        str(util_path.resolve()), ""
    )


# Custom rules tests

def test_collect_custom_rules_global_only(tmp_path, monkeypatch):
    """Test that global rules.md is collected when it's the only one."""
    home = tmp_path / "home"
    home.mkdir()
    global_rules = home / ".latchline" / "rules.md"
    global_rules.parent.mkdir(parents=True)
    global_rules.write_text("# Global Rules\n- Rule 1\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    result = ai_review.collect_custom_rules([], "", {"REVIEWER_CUSTOM_RULES": "1"})
    assert "=== Global Rules" in result
    assert "Rule 1" in result


def test_collect_custom_rules_project_only(tmp_path, monkeypatch):
    """Test that project rules.md is collected."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project_root = tmp_path / "repo"
    project_root.mkdir()
    project_rules = project_root / ".latchline" / "rules.md"
    project_rules.parent.mkdir(parents=True)
    project_rules.write_text("# Project Rules\n- Project rule\n", encoding="utf-8")
    result = ai_review.collect_custom_rules([], str(project_root), {"REVIEWER_CUSTOM_RULES": "1"})
    assert "=== Project Rules ===" in result
    assert "Project rule" in result


def test_collect_custom_rules_hierarchy(tmp_path, monkeypatch):
    """Test global + project + module rules are all collected."""
    home = tmp_path / "home"
    home.mkdir()
    global_rules = home / ".latchline" / "rules.md"
    global_rules.parent.mkdir(parents=True)
    global_rules.write_text("# Global\n- Global rule\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    project_root = tmp_path / "repo"
    project_root.mkdir()
    project_rules = project_root / ".latchline" / "rules.md"
    project_rules.parent.mkdir(parents=True)
    project_rules.write_text("# Project\n- Project rule\n", encoding="utf-8")
    module_dir = project_root / "services" / "payments"
    module_dir.mkdir(parents=True)
    module_rules = module_dir / ".latchline" / "rules.md"
    module_rules.parent.mkdir(parents=True)
    module_rules.write_text("# Payments\n- Payment rule\n", encoding="utf-8")
    changed_files = [str(module_dir / "handler.py")]
    result = ai_review.collect_custom_rules(
        changed_files, str(project_root), {"REVIEWER_CUSTOM_RULES": "1"}
    )
    assert "=== Global Rules" in result
    assert "Global rule" in result
    assert "=== Project Rules ===" in result
    assert "Project rule" in result
    assert "=== Module Rules (services/payments) ===" in result
    assert "Payment rule" in result


def test_collect_custom_rules_disabled(tmp_path, monkeypatch):
    """Test that REVIEWER_CUSTOM_RULES=0 returns empty string."""
    home = tmp_path / "home"
    home.mkdir()
    global_rules = home / ".latchline" / "rules.md"
    global_rules.parent.mkdir(parents=True)
    global_rules.write_text("# Global Rules\n- Rule 1\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    result = ai_review.collect_custom_rules([], "", {"REVIEWER_CUSTOM_RULES": "0"})
    assert result == ""


def test_collect_custom_rules_no_files(tmp_path, monkeypatch):
    """Test that no rules.md files returns empty string."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project_root = tmp_path / "repo"
    project_root.mkdir()
    result = ai_review.collect_custom_rules([], str(project_root), {"REVIEWER_CUSTOM_RULES": "1"})
    assert result == ""


def test_get_rules_paths_deduplication(tmp_path, monkeypatch):
    """Test that same directory is not included twice."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project_root = tmp_path / "repo"
    project_root.mkdir()
    module_dir = project_root / "services"
    module_dir.mkdir(parents=True)
    module_rules = module_dir / ".latchline" / "rules.md"
    module_rules.parent.mkdir(parents=True)
    module_rules.write_text("# Module\n", encoding="utf-8")
    changed_files = [
        str(module_dir / "file1.py"),
        str(module_dir / "file2.py"),
    ]
    paths = ai_review.get_rules_paths(changed_files, str(project_root))
    labels = [label for _, label in paths]
    assert labels.count("Module Rules (services)") == 1


# ============================================================================
# Edge Case Tests - Config Parsing
# ============================================================================

def test_parse_config_empty_file(tmp_path):
    """Empty config file returns empty dict."""
    path = tmp_path / "settings.conf"
    path.write_text("", encoding="utf-8")
    config = ai_review.parse_config(str(path))
    assert config == {}


def test_parse_config_double_quotes(tmp_path):
    """Double quoted values are handled."""
    path = tmp_path / "settings.conf"
    path.write_text('REVIEWER_BACKEND="claude"\n', encoding="utf-8")
    config = ai_review.parse_config(str(path))
    assert config["REVIEWER_BACKEND"] == "claude"


def test_parse_config_spaces_around_equals(tmp_path):
    """Spaces around = are trimmed."""
    path = tmp_path / "settings.conf"
    path.write_text("REVIEWER_BACKEND = codex\n", encoding="utf-8")
    config = ai_review.parse_config(str(path))
    assert config["REVIEWER_BACKEND"] == "codex"


def test_parse_config_nonexistent_file(tmp_path):
    """Nonexistent file returns empty dict."""
    path = tmp_path / "nonexistent.conf"
    config = ai_review.parse_config(str(path))
    assert config == {}


def test_parse_config_comments_only(tmp_path):
    """File with only comments returns empty dict."""
    path = tmp_path / "settings.conf"
    path.write_text("# comment 1\n# comment 2\n", encoding="utf-8")
    config = ai_review.parse_config(str(path))
    assert config == {}


# ============================================================================
# Edge Case Tests - Structured Output Parsing
# ============================================================================

def test_parse_structured_output_empty_findings():
    """Empty findings array is valid."""
    raw = '{"findings": [], "blockers_summary": "none", "notes_summary": "none"}'
    result = ai_review.parse_structured_output(raw, 1, "codex")
    assert len(result.findings) == 0
    assert result.blockers_summary == "none"


def test_parse_structured_output_missing_optional_fields():
    """Findings with missing optional fields still parse."""
    raw = '''
    {
      "findings": [
        {
          "id": "F001",
          "severity": "low",
          "category": "style",
          "confidence": 50,
          "title": "Minor issue",
          "description": "Desc"
        }
      ],
      "blockers_summary": "none",
      "notes_summary": "none"
    }
    '''
    result = ai_review.parse_structured_output(raw, 1, "codex")
    assert len(result.findings) == 1
    assert result.findings[0].file_path is None
    assert result.findings[0].line_start is None
    assert result.findings[0].suggested_fix is None


def test_parse_structured_output_json_in_markdown():
    """JSON wrapped in markdown code blocks is extracted."""
    raw = '''Here is the review:
```json
{
  "findings": [],
  "blockers_summary": "none",
  "notes_summary": "all good"
}
```
'''
    result = ai_review.parse_structured_output(raw, 1, "claude")
    assert result.notes_summary == "all good"


def test_parse_structured_output_malformed_json():
    """Malformed JSON falls back to legacy parsing."""
    raw = '{"findings": [invalid json'
    result = ai_review.parse_structured_output(raw, 1, "codex")
    # Should not crash, falls back to legacy
    assert result is not None


def test_parse_structured_output_extra_text_around_json():
    """JSON with extra text before/after is extracted."""
    raw = '''
    Some preamble text...
    {"findings": [], "blockers_summary": "none", "notes_summary": "ok"}
    Some trailing text...
    '''
    result = ai_review.parse_structured_output(raw, 1, "codex")
    assert result.notes_summary == "ok"


# ============================================================================
# Edge Case Tests - Diff Parsing
# ============================================================================

def test_parse_unified_diff_multiple_hunks():
    """Multiple hunks in the same file are parsed."""
    diff = """\
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,4 @@ def foo():
     pass
+    return True
@@ -50,3 +51,4 @@ def bar():
     pass
+    return False
"""
    hunks = ai_review.parse_unified_diff(diff)
    assert len(hunks) == 2
    assert hunks[0].old_start == 10
    assert hunks[1].old_start == 50


def test_parse_unified_diff_multiple_files():
    """Multiple files in diff are parsed."""
    diff = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,1 +1,2 @@
+import os
 pass
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -1,1 +1,2 @@
+import sys
 pass
"""
    hunks = ai_review.parse_unified_diff(diff)
    assert len(hunks) == 2
    assert hunks[0].file_path == "src/a.py"
    assert hunks[1].file_path == "src/b.py"


def test_parse_unified_diff_new_file():
    """New file (all additions) is parsed."""
    diff = """\
diff --git a/src/new.py b/src/new.py
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,3 @@
+def new_func():
+    pass
+
"""
    hunks = ai_review.parse_unified_diff(diff)
    assert len(hunks) == 1
    assert hunks[0].file_path == "src/new.py"
    assert all(line.line_type in ("add", "context") for line in hunks[0].lines)


def test_parse_unified_diff_deleted_file():
    """Deleted file (all removals) is parsed."""
    diff = """\
diff --git a/src/old.py b/src/old.py
--- a/src/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_func():
-    pass
-
"""
    hunks = ai_review.parse_unified_diff(diff)
    assert len(hunks) == 1
    assert all(line.line_type in ("remove", "context") for line in hunks[0].lines)


def test_parse_unified_diff_empty():
    """Empty diff returns empty list."""
    hunks = ai_review.parse_unified_diff("")
    assert hunks == []


def test_parse_unified_diff_no_hunks():
    """Diff with headers but no hunks returns empty."""
    diff = """\
diff --git a/src/app.py b/src/app.py
"""
    hunks = ai_review.parse_unified_diff(diff)
    assert hunks == []


# ============================================================================
# Edge Case Tests - Path Matching
# ============================================================================

def test_paths_match_exact():
    """Exact paths match."""
    assert ai_review._paths_match("src/db.py", "src/db.py")


def test_paths_match_suffix():
    """Suffix matching works."""
    assert ai_review._paths_match("/home/user/project/src/db.py", "src/db.py")
    assert ai_review._paths_match("src/db.py", "/home/user/project/src/db.py")


def test_paths_match_different_files():
    """Different files don't match."""
    assert not ai_review._paths_match("src/db.py", "src/api.py")
    assert not ai_review._paths_match("src/db.py", "test/db.py")


def test_paths_match_partial_name():
    """Partial filename doesn't match."""
    assert not ai_review._paths_match("src/database.py", "src/db.py")


# ============================================================================
# Edge Case Tests - Context Bundle
# ============================================================================

def test_build_context_bundle_disabled(tmp_path):
    """REVIEWER_CONTEXT=0 returns empty paths."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)
    changed = project_root / "app.py"
    changed.write_text("print('hello')\n", encoding="utf-8")
    context_file, files_file = ai_review.build_context_bundle(
        run_dir=str(run_dir),
        project_root=str(project_root),
        cwd_path=str(project_root),
        changed_files=[str(changed)],
        config={"REVIEWER_CONTEXT": "0"},
    )
    assert context_file == ""
    assert files_file == ""


def test_build_context_bundle_depth_zero(tmp_path):
    """Depth 0 includes only changed files, no imports."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)
    main_path = project_root / "main.py"
    util_path = project_root / "util.py"
    main_path.write_text("from util import helper\n", encoding="utf-8")
    util_path.write_text("def helper(): pass\n", encoding="utf-8")
    context_file, _ = ai_review.build_context_bundle(
        run_dir=str(run_dir),
        project_root=str(project_root),
        cwd_path=str(project_root),
        changed_files=[str(main_path)],
        config={
            "REVIEWER_CONTEXT": "1",
            "REVIEWER_CONTEXT_MAX_BYTES": "10000",
            "REVIEWER_CONTEXT_DEPTH": "0",
        },
    )
    content = Path(context_file).read_text(encoding="utf-8")
    assert "FILE: main.py" in content
    # util.py should not be included with depth 0
    assert "FILE: util.py" not in content


def test_build_context_bundle_missing_import(tmp_path):
    """Missing import files are skipped gracefully."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)
    main_path = project_root / "main.py"
    main_path.write_text("from nonexistent import something\n", encoding="utf-8")
    context_file, _ = ai_review.build_context_bundle(
        run_dir=str(run_dir),
        project_root=str(project_root),
        cwd_path=str(project_root),
        changed_files=[str(main_path)],
        config={
            "REVIEWER_CONTEXT": "1",
            "REVIEWER_CONTEXT_MAX_BYTES": "10000",
            "REVIEWER_CONTEXT_DEPTH": "1",
        },
    )
    # Should not crash
    content = Path(context_file).read_text(encoding="utf-8")
    assert "FILE: main.py" in content


# ============================================================================
# Edge Case Tests - Impact Analysis
# ============================================================================

def test_detect_breaking_changes_added_function(tmp_path):
    """Adding a new function is not breaking."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    before_path = tmp_path / "before.py"
    after_path = tmp_path / "after.py"
    before_path.write_text("def existing(): pass\n", encoding="utf-8")
    after_path.write_text("def existing(): pass\n\ndef new_func(): pass\n", encoding="utf-8")
    changes = ai_review.detect_breaking_changes(
        str(before_path), str(after_path), str(project_root)
    )
    # Adding is not breaking
    breaking = [c for c in changes if c.breaking]
    assert len(breaking) == 0


def test_detect_breaking_changes_added_required_arg(tmp_path):
    """Adding a required argument is breaking."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    before_path = tmp_path / "before.py"
    after_path = tmp_path / "after.py"
    before_path.write_text("def process(a, b): pass\n", encoding="utf-8")
    after_path.write_text("def process(a, b, c): pass\n", encoding="utf-8")
    changes = ai_review.detect_breaking_changes(
        str(before_path), str(after_path), str(project_root)
    )
    breaking = [c for c in changes if c.breaking]
    assert len(breaking) == 1
    assert "Added required arguments" in breaking[0].breaking_reason


def test_detect_breaking_changes_added_optional_arg(tmp_path):
    """Adding an optional argument is detected as a signature change.

    Note: Current implementation doesn't distinguish optional from required args
    in signature comparison, so adding any argument is flagged as potentially breaking.
    """
    project_root = tmp_path / "repo"
    project_root.mkdir()
    before_path = tmp_path / "before.py"
    after_path = tmp_path / "after.py"
    before_path.write_text("def process(a, b): pass\n", encoding="utf-8")
    after_path.write_text("def process(a, b, c=None): pass\n", encoding="utf-8")
    changes = ai_review.detect_breaking_changes(
        str(before_path), str(after_path), str(project_root)
    )
    # Implementation flags this as breaking (conservative approach)
    assert len(changes) == 1


def test_analyze_cross_file_impact_disabled(tmp_path):
    """REVIEWER_IMPACT_ANALYSIS=0 returns empty report."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "before").mkdir(parents=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    report = ai_review.analyze_cross_file_impact(
        run_dir=str(run_dir),
        project_root=str(project_root),
        log_dir=str(log_dir),
        changed_files=[],
        diff_text="",
        config={"REVIEWER_IMPACT_ANALYSIS": "0"},
    )
    assert len(report.dependents) == 0
    assert len(report.changed_symbols) == 0
    assert len(report.breaking_changes) == 0


# ============================================================================
# Edge Case Tests - Custom Rules
# ============================================================================

def test_collect_custom_rules_empty_file(tmp_path, monkeypatch):
    """Empty rules.md returns empty string."""
    home = tmp_path / "home"
    home.mkdir()
    global_rules = home / ".latchline" / "rules.md"
    global_rules.parent.mkdir(parents=True)
    global_rules.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    result = ai_review.collect_custom_rules([], "", {"REVIEWER_CUSTOM_RULES": "1"})
    assert result == ""


def test_collect_custom_rules_whitespace_only(tmp_path, monkeypatch):
    """Rules.md with only whitespace returns empty string."""
    home = tmp_path / "home"
    home.mkdir()
    global_rules = home / ".latchline" / "rules.md"
    global_rules.parent.mkdir(parents=True)
    global_rules.write_text("   \n\n  \n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    result = ai_review.collect_custom_rules([], "", {"REVIEWER_CUSTOM_RULES": "1"})
    assert result == ""


def test_collect_custom_rules_nested_subdirectories(tmp_path, monkeypatch):
    """Rules in deeply nested directories are collected."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project_root = tmp_path / "repo"
    project_root.mkdir()
    deep_dir = project_root / "a" / "b" / "c" / "d"
    deep_dir.mkdir(parents=True)
    deep_rules = deep_dir / ".latchline" / "rules.md"
    deep_rules.parent.mkdir(parents=True)
    deep_rules.write_text("# Deep Rules\n- Rule deep\n", encoding="utf-8")
    changed_files = [str(deep_dir / "file.py")]
    result = ai_review.collect_custom_rules(
        changed_files, str(project_root), {"REVIEWER_CUSTOM_RULES": "1"}
    )
    assert "Deep Rules" in result
    assert "Module Rules (a/b/c/d)" in result


def test_collect_custom_rules_env_fallback(tmp_path, monkeypatch):
    """REVIEWER_CUSTOM_RULES env var is used when config doesn't have the key."""
    home = tmp_path / "home"
    home.mkdir()
    global_rules = home / ".latchline" / "rules.md"
    global_rules.parent.mkdir(parents=True)
    global_rules.write_text("# Rules\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("REVIEWER_CUSTOM_RULES", "0")
    # Config doesn't have the key, so env var is used
    result = ai_review.collect_custom_rules([], "", {})
    assert result == ""


# ============================================================================
# Edge Case Tests - Finding Display
# ============================================================================

def test_format_finding_no_location():
    """Finding without file location is formatted correctly."""
    finding = ai_review.Finding(
        id="F001", severity="info", category="style",
        confidence=50, title="General Note",
        description="Some observation",
        file_path=None, line_start=None,
    )
    output = ai_review.format_finding_for_display(finding)
    assert "[INFO]" in output
    assert "General Note" in output
    assert ":" not in output or "[F001]" in output  # No file:line


def test_format_finding_single_line():
    """Finding with single line shows line correctly."""
    finding = ai_review.Finding(
        id="F001", severity="medium", category="correctness",
        confidence=75, title="Issue",
        description="Desc",
        file_path="src/app.py", line_start=42, line_end=42,
    )
    output = ai_review.format_finding_for_display(finding)
    assert "[src/app.py:42]" in output


def test_format_finding_line_range():
    """Finding with line range shows range correctly."""
    finding = ai_review.Finding(
        id="F001", severity="high", category="security",
        confidence=90, title="Issue",
        description="Desc",
        file_path="src/app.py", line_start=10, line_end=20,
    )
    output = ai_review.format_finding_for_display(finding)
    assert "[src/app.py:10-20]" in output


# ============================================================================
# Edge Case Tests - Blocking Logic
# ============================================================================

def test_has_blocking_findings_empty():
    """Empty findings is not blocking."""
    result = ai_review.ReviewResult(findings=[])
    assert ai_review.has_blocking_findings(result, 70) is False


def test_has_blocking_findings_info_severity():
    """Info severity is never blocking."""
    finding = ai_review.Finding(
        id="F001", severity="info", category="style",
        confidence=100, title="Note", description="Desc",
    )
    result = ai_review.ReviewResult(findings=[finding])
    assert ai_review.has_blocking_findings(result, 0) is False


def test_has_blocking_findings_medium_severity():
    """Medium severity never blocks (only critical/high block)."""
    finding = ai_review.Finding(
        id="F001", severity="medium", category="correctness",
        confidence=85, title="Issue", description="Desc",
    )
    result = ai_review.ReviewResult(findings=[finding])
    # Medium severity doesn't block regardless of confidence
    assert ai_review.has_blocking_findings(result, 70) is False
    assert ai_review.has_blocking_findings(result, 50) is False


def test_get_blocking_findings_filters_correctly():
    """get_blocking_findings returns only qualifying findings."""
    findings = [
        ai_review.Finding(id="F001", severity="critical", category="security",
                          confidence=95, title="A", description=""),
        ai_review.Finding(id="F002", severity="high", category="correctness",
                          confidence=50, title="B", description=""),  # Below threshold
        ai_review.Finding(id="F003", severity="low", category="style",
                          confidence=95, title="C", description=""),  # Low severity
        ai_review.Finding(id="F004", severity="info", category="style",
                          confidence=100, title="D", description=""),  # Info
        ai_review.Finding(id="F005", severity="high", category="security",
                          confidence=80, title="E", description=""),
    ]
    result = ai_review.ReviewResult(findings=findings)
    blocking = ai_review.get_blocking_findings(result, 70)
    ids = [f.id for f in blocking]
    assert "F001" in ids  # Critical + high conf
    assert "F005" in ids  # High + high conf
    assert "F002" not in ids  # Below threshold
    assert "F003" not in ids  # Low severity
    assert "F004" not in ids  # Info


# ============================================================================
# Edge Case Tests - Symbol Extraction
# ============================================================================

def test_extract_python_symbols_async():
    """Async functions are extracted."""
    code = '''
async def fetch_data(url):
    return await get(url)
'''
    symbols = ai_review.extract_python_symbols("/test.py", code)
    names = {s.name for s in symbols}
    assert "fetch_data" in names


def test_extract_python_symbols_decorated():
    """Decorated functions are extracted."""
    code = '''
@decorator
def decorated_func():
    pass

@app.route("/")
@auth_required
def handler():
    pass
'''
    symbols = ai_review.extract_python_symbols("/test.py", code)
    names = {s.name for s in symbols}
    assert "decorated_func" in names
    assert "handler" in names


def test_extract_python_symbols_empty():
    """Empty file returns empty list."""
    symbols = ai_review.extract_python_symbols("/test.py", "")
    assert symbols == []


def test_extract_js_symbols_arrow_functions():
    """Arrow functions are extracted."""
    code = '''
const add = (a, b) => a + b;
const process = async (data) => {
    return data;
};
'''
    symbols = ai_review.extract_js_symbols("/test.js", code)
    names = {s.name for s in symbols}
    assert "add" in names
    assert "process" in names


def test_extract_js_symbols_empty():
    """Empty file returns empty list."""
    symbols = ai_review.extract_js_symbols("/test.js", "")
    assert symbols == []


# ============================================================================
# Component Tests - Review Prompt Building
# ============================================================================

def test_build_structured_review_prompt_smoke():
    """Smoke pass prompt contains smoke focus."""
    prompt = ai_review.build_structured_review_prompt("smoke", "")
    assert "surface-level" in prompt.lower() or "obvious" in prompt.lower()


def test_build_structured_review_prompt_semantic():
    """Semantic pass prompt contains deep analysis focus."""
    prompt = ai_review.build_structured_review_prompt("semantic", "")
    assert "semantic" in prompt.lower() or "deep" in prompt.lower()


def test_build_structured_review_prompt_with_rules():
    """Custom rules are included in prompt."""
    rules = "# My Rules\n- No print statements"
    prompt = ai_review.build_structured_review_prompt("smoke", rules)
    assert "My Rules" in prompt
    assert "No print statements" in prompt


def test_build_structured_review_prompt_cites_rules():
    """Prompt includes instruction to cite rules."""
    rules = "# Rules\n- Rule 1"
    prompt = ai_review.build_structured_review_prompt("smoke", rules)
    assert "cite" in prompt.lower() or "rule" in prompt.lower()


# ============================================================================
# Component Tests - Annotated Diff Rendering
# ============================================================================

def test_render_annotated_diff_no_findings():
    """Diff with no findings renders cleanly."""
    diff = """\
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,1 +1,2 @@
+import os
 pass
"""
    hunks = ai_review.parse_unified_diff(diff)
    annotated = ai_review.map_findings_to_diff([], hunks)
    output = ai_review.render_annotated_diff(annotated, [], use_color=False)
    assert "src/app.py" in output
    assert "+import os" in output


def test_render_annotated_diff_multiple_findings_same_line():
    """Multiple findings on same line are shown."""
    diff = """\
diff --git a/src/db.py b/src/db.py
--- a/src/db.py
+++ b/src/db.py
@@ -1,1 +1,2 @@
+query = f"SELECT * FROM {table}"
 pass
"""
    hunks = ai_review.parse_unified_diff(diff)
    findings = [
        ai_review.Finding(id="F001", severity="high", category="security",
                          confidence=90, title="SQL Injection",
                          description="Desc", file_path="src/db.py", line_start=1),
        ai_review.Finding(id="F002", severity="medium", category="style",
                          confidence=70, title="Use parameterized query",
                          description="Desc", file_path="src/db.py", line_start=1),
    ]
    annotated = ai_review.map_findings_to_diff(findings, hunks)
    output = ai_review.render_annotated_diff(annotated, findings, use_color=False)
    assert "F001" in output
    assert "F002" in output
