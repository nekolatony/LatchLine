from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


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


def run_codex(review_prompt: str, diff: str, codex_out: str, base_dir: str) -> None:
    if not shutil.which("codex"):
        return
    input_text = f"{review_prompt}\n\nDIFF:\n{diff}\n"
    cmd = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        codex_out,
        "-",
    ]
    subprocess.run(cmd, input=input_text, text=True, check=False)
    if os.path.isfile(codex_out):
        shutil.copyfile(codex_out, os.path.join(base_dir, "latest.codex.md"))


def run_claude(
    review_prompt: str, diff_file: str, run_dir: str, claude_out: str, base_dir: str
) -> None:
    if not shutil.which("claude"):
        return
    prompt = f"Review the diff at {diff_file}. Use the following format:\n{review_prompt}"
    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "Read",
        "--add-dir",
        run_dir,
        prompt,
    ]
    with open(claude_out, "w", encoding="utf-8") as out:
        subprocess.run(cmd, stdout=out, text=True, check=False)
    if os.path.isfile(claude_out):
        shutil.copyfile(claude_out, os.path.join(base_dir, "latest.claude.md"))


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
    log_file: str,
) -> None:
    files = [line for line in read_lines(files_file) if line]
    diff_content = read_text(diff_file)

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

    shutil.copyfile(log_md, os.path.join(os.path.dirname(log_file), "latest.log.md"))

    entry = {
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
        "behavior regressions, and missing tests. Review only the diff below "
        "(changes from the current prompt). Respond with:\n"
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

        diff = read_text(diff_file)
        prompt_ts = read_text(prompt_ts_file).strip() or utc_now()
        prompt_cwd = read_text(cwd_file).strip()
        gate_action = read_text(gate_file).strip()
        gate_answer = read_text(gate_answer_file).strip()
        gate_question = read_text(gate_question_file).strip()
        codex_out = os.path.join(run_dir, "review.codex.md")
        claude_out = os.path.join(run_dir, "review.claude.md")

        if backend == "codex":
            run_codex(review_prompt, diff, codex_out, base_dir)
        elif backend == "claude":
            run_claude(review_prompt, diff_file, run_dir, claude_out, base_dir)
        elif backend == "both":
            run_codex(review_prompt, diff, codex_out, base_dir)
            run_claude(review_prompt, diff_file, run_dir, claude_out, base_dir)
        else:
            return 0

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
                log_file=log_file,
            )
            if os.path.exists(pending_run_file):
                os.remove(pending_run_file)

        if block == "2" and gate_action:
            for fp in (gate_file, gate_answer_file, gate_question_file):
                if os.path.exists(fp):
                    os.remove(fp)
            return 0

        if block == "2":
            review_question = build_review_question(backend, codex_out, claude_out)
            if backend == "both":
                if os.path.isfile(codex_out):
                    sys.stderr.write(f"CODEX REVIEW:\n{read_text(codex_out)}\n")
                if os.path.isfile(claude_out):
                    sys.stderr.write(f"CLAUDE REVIEW:\n{read_text(claude_out)}\n")
            else:
                review_file = codex_out if backend != "claude" else claude_out
                if os.path.isfile(review_file):
                    sys.stderr.write(read_text(review_file) + "\n")
                else:
                    sys.stderr.write(f"Review completed; see {log_md}\n")
            sys.stderr.write(
                "\nACTION REQUIRED: Reply with 'review:apply' to act on this feedback, "
                "or 'review:skip' to skip reviews for the rest of this session. "
                "You can re-enable with 'review:enable' or 'review:resume'.\n"
            )
            sys.stderr.write(
                "\nUSER REQUEST (MANDATORY): Call AskUserQuestion now with this question: "
                f'"{review_question}" This is a user request. Do not proceed until the user answers.\n'
            )
            return 2

        if block == "1":
            if review_has_blockers(codex_out):
                line = first_blockers_line(codex_out)
                if line:
                    sys.stderr.write(line + "\n")
                return 2
            if review_has_blockers(claude_out):
                line = first_blockers_line(claude_out)
                if line:
                    sys.stderr.write(line + "\n")
                return 2

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
