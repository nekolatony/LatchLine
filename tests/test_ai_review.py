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
