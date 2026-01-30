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
