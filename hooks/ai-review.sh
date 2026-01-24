#!/usr/bin/env bash
set -euo pipefail

input_file="$(mktemp)"
trap 'rm -f "$input_file"' EXIT
cat > "$input_file"

json_get() {
  local key="$1"
  python3 - "$input_file" "$key" <<'PY'
import json
import sys

path = sys.argv[2].split(".")
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

cur = data
for part in path:
    if isinstance(cur, dict):
        cur = cur.get(part, "")
    else:
        cur = ""
        break

if isinstance(cur, (dict, list)):
    cur = ""
print(cur if cur is not None else "")
PY
}

hook_event_name="$(json_get "hook_event_name")"
session_id="$(json_get "session_id")"
tool_name="$(json_get "tool_name")"
file_path="$(json_get "tool_input.file_path")"
cwd="$(json_get "cwd")"
prompt_text="$(json_get "prompt")"

config_file="${HOME}/.claude/reviewer.conf"
if [ -f "$config_file" ]; then
  # shellcheck disable=SC1090
  . "$config_file"
fi

backend="${REVIEWER_BACKEND:-codex}"
block="${REVIEWER_BLOCK:-0}"

if [ -z "$session_id" ]; then
  exit 0
fi

base_dir="${HOME}/.claude/reviews"
state_dir="${base_dir}/state"
runs_dir="${base_dir}/runs"
log_file="${base_dir}/log.jsonl"

mkdir -p "$state_dir" "$runs_dir"

session_dir="${state_dir}/${session_id}"
prompt_id_file="${session_dir}/prompt_id"
files_file="${session_dir}/files"
prompt_ts_file="${session_dir}/prompt_ts"
cwd_file="${session_dir}/cwd"
gate_file="${session_dir}/review_gate"
gate_answer_file="${session_dir}/review_gate_answer"
gate_question_file="${session_dir}/review_gate_question"
session_skip_file="${session_dir}/review_session_skip"
pending_run_file="${session_dir}/pending_log_run"

review_prompt="You are a strict reviewer. Find correctness bugs, security issues,\
 behavior regressions, and missing tests. Review only the diff below (changes\
 from the current prompt). Respond with:\n\
BLOCKERS: <list or 'none'>\n\
NOTES: <short list of improvements>"

run_codex() {
  if ! command -v codex >/dev/null 2>&1; then
    return 0
  fi
  printf "%b\n\nDIFF:\n%s\n" "$review_prompt" "$diff" | \
    codex exec --sandbox read-only --output-last-message "$codex_out" -
  cp -f "$codex_out" "${base_dir}/latest.codex.md"
}

run_claude() {
  if ! command -v claude >/dev/null 2>&1; then
    return 0
  fi
  claude -p \
    --permission-mode bypassPermissions \
    --tools "Read" \
    --add-dir "$run_dir" \
    "Review the diff at $diff_file. Use the following format:\n$review_prompt" \
    > "$claude_out"
  cp -f "$claude_out" "${base_dir}/latest.claude.md"
}

extract_review_field() {
  local file="$1"
  local label="$2"
  if [ ! -f "$file" ]; then
    printf ""
    return
  fi
  awk -v label="$label" 'BEGIN{IGNORECASE=1} $0 ~ "^"label":" {sub("^[^:]*:[[:space:]]*", "", $0); print; exit}' "$file"
}

build_review_question() {
  local question=""
  if [ "$backend" = "both" ]; then
    local codex_blockers codex_notes claude_blockers claude_notes
    codex_blockers="$(extract_review_field "$codex_out" "BLOCKERS")"
    codex_notes="$(extract_review_field "$codex_out" "NOTES")"
    claude_blockers="$(extract_review_field "$claude_out" "BLOCKERS")"
    claude_notes="$(extract_review_field "$claude_out" "NOTES")"
    codex_blockers="${codex_blockers:-none}"
    codex_notes="${codex_notes:-none}"
    claude_blockers="${claude_blockers:-none}"
    claude_notes="${claude_notes:-none}"
    question="Review complete. CODEX BLOCKERS: ${codex_blockers}. CODEX NOTES: ${codex_notes}. CLAUDE BLOCKERS: ${claude_blockers}. CLAUDE NOTES: ${claude_notes}. Apply feedback (review:apply) or skip reviews for the rest of this session (review:skip)? You can re-enable with review:enable or review:resume."
  else
    local review_file blockers notes
    review_file="$codex_out"
    if [ "$backend" = "claude" ]; then
      review_file="$claude_out"
    fi
    blockers="$(extract_review_field "$review_file" "BLOCKERS")"
    notes="$(extract_review_field "$review_file" "NOTES")"
    blockers="${blockers:-none}"
    notes="${notes:-none}"
    question="Review complete. BLOCKERS: ${blockers}. NOTES: ${notes}. Apply feedback (review:apply) or skip reviews for the rest of this session (review:skip)? You can re-enable with review:enable or review:resume."
  fi
  question="${question//$'\n'/ }"
  printf "%s" "$question" | tr -s ' '
}

write_run_log() {
  local log_md="$1"
  local update_json="${2:-0}"
  {
    printf "timestamp: %s\n" "$prompt_ts"
    printf "session_id: %s\n" "$session_id"
    printf "prompt_id: %s\n" "$prompt_id"
    printf "cwd: %s\n" "$prompt_cwd"
    printf "gate: %s\n" "$gate_action"
    if [ -n "$gate_question" ]; then
      printf "gate_question: %s\n" "$gate_question"
    fi
    if [ -n "$gate_answer" ]; then
      printf "gate_answer: %s\n" "$gate_answer"
    fi
    printf "files:\n"
    sed 's/^/- /' "$files_file"
    printf "\n"
    printf "diff:\n"
    printf "```diff\n"
    cat "$diff_file"
    printf "\n```\n"
    if [ -f "$codex_out" ]; then
      printf "\n"
      printf "codex_response:\n"
      cat "$codex_out"
      printf "\n"
    fi
    if [ -f "$claude_out" ]; then
      printf "\n"
      printf "claude_response:\n"
      cat "$claude_out"
      printf "\n"
    fi
  } > "$log_md"
  cp -f "$log_md" "${base_dir}/latest.log.md"

  PROMPT_TS="$prompt_ts" \
  SESSION_ID="$session_id" \
  PROMPT_ID="$prompt_id" \
  PROMPT_CWD="$prompt_cwd" \
  GATE_ACTION="$gate_action" \
  GATE_QUESTION="$gate_question" \
  GATE_ANSWER="$gate_answer" \
  DIFF_FILE="$diff_file" \
  BACKEND="$backend" \
  CODEX_OUT="$codex_out" \
  CLAUDE_OUT="$claude_out" \
  LOG_UPDATE="$update_json" \
  python3 - "$files_file" "$log_file" <<'PY'
import json
import os
import sys

files = []
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            files.append(line)

entry = {
    "timestamp": os.environ.get("PROMPT_TS", ""),
    "session_id": os.environ.get("SESSION_ID", ""),
    "prompt_id": int(os.environ.get("PROMPT_ID", "0") or 0),
    "cwd": os.environ.get("PROMPT_CWD", ""),
    "files": files,
    "diff_file": os.environ.get("DIFF_FILE", ""),
    "gate": os.environ.get("GATE_ACTION", ""),
    "gate_question": os.environ.get("GATE_QUESTION", ""),
    "gate_answer": os.environ.get("GATE_ANSWER", ""),
    "backend": os.environ.get("BACKEND", ""),
    "codex_output": os.environ.get("CODEX_OUT", ""),
    "claude_output": os.environ.get("CLAUDE_OUT", ""),
}

log_path = sys.argv[2]
update_mode = os.environ.get("LOG_UPDATE", "") == "1"

if update_mode and os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
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
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
else:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
PY
}

case "$hook_event_name" in
  UserPromptSubmit)
    mkdir -p "$session_dir"
    prompt_id="$(cat "$prompt_id_file" 2>/dev/null || echo 0)"
    if ! [[ "$prompt_id" =~ ^[0-9]+$ ]]; then
      prompt_id=0
    fi
    prompt_id=$((prompt_id + 1))
    printf "%s\n" "$prompt_id" > "$prompt_id_file"
    prompt_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf "%s\n" "$prompt_ts" > "$prompt_ts_file"
    printf "%s\n" "$cwd" > "$cwd_file"
    : > "$files_file"
    rm -f "$gate_file" "$gate_answer_file" "$gate_question_file"
    gate_info="$(python3 - <<'PY'
import re
import sys

text = sys.stdin.read()
action = ""
if re.search(r'\breview:apply\b', text, re.IGNORECASE):
    action = "apply"
elif re.search(r'\breview:skip\b', text, re.IGNORECASE):
    action = "skip"
enable = bool(re.search(r'\breview:(enable|resume)\b', text, re.IGNORECASE))
print(f"action={action}")
print(f"enable={1 if enable else 0}")
PY
<<< "${prompt_text:-}")"
    gate_action=""
    gate_enable=""
    while IFS= read -r line; do
      case "$line" in
        action=*) gate_action="${line#action=}" ;;
        enable=*) gate_enable="${line#enable=}" ;;
      esac
    done <<< "$gate_info"
    if [ "$gate_enable" = "1" ]; then
      rm -f "$session_skip_file"
    fi
    if [ -n "$gate_action" ]; then
      printf "%s\n" "$gate_action" > "$gate_file"
      if [ "$gate_action" = "skip" ]; then
        printf "1\n" > "$session_skip_file"
      elif [ "$gate_action" = "apply" ]; then
        rm -f "$session_skip_file"
      fi
    fi
    run_dir="${runs_dir}/${session_id}/prompt-${prompt_id}"
    mkdir -p "$run_dir/before"
    exit 0
    ;;
  PostToolUse)
    if [ "$tool_name" != "AskUserQuestion" ]; then
      exit 0
    fi
    gate_info="$(python3 - "$input_file" <<'PY'
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

if data.get("tool_name") != "AskUserQuestion":
    sys.exit(0)

resp = data.get("tool_response") or {}
answers = []
questions = []

if isinstance(resp, dict):
    answers_obj = resp.get("answers")
    if isinstance(answers_obj, dict):
        answers.extend([str(v) for v in answers_obj.values()])
    questions_obj = resp.get("questions")
    if isinstance(questions_obj, list):
        for q in questions_obj:
            if isinstance(q, dict):
                questions.append(str(q.get("question", "")))

# Only act on our review gate question.
review_question = any(
    ("review:apply" in q.lower() or "review:skip" in q.lower() or "review complete" in q.lower())
    for q in questions
)
if not review_question:
    sys.exit(0)

answers_text = " | ".join([a.replace("\\n", " ").strip() for a in answers if a.strip()])
questions_text = " | ".join([q.replace("\\n", " ").strip() for q in questions if q.strip()])

action = ""
enable = False
for text in answers:
    if re.search(r'\breview:skip\b', text, re.IGNORECASE):
        action = "skip"
        break
for text in answers:
    if re.search(r'\breview:apply\b', text, re.IGNORECASE):
        action = "apply"
        break
for text in answers:
    if re.search(r'\breview:(enable|resume)\b', text, re.IGNORECASE):
        action = "apply"
        enable = True
        break

# Fallback: accept plain "apply"/"skip" wording if user typed freeform.
if not action:
    for text in answers:
        if re.search(r'\bapply\b', text, re.IGNORECASE):
            action = "apply"
            break
        if re.search(r'\bskip\b', text, re.IGNORECASE):
            action = "skip"
            break

print(f"action={action}")
print(f"enable={1 if enable else 0}")
print(f"answer={answers_text}")
print(f"question={questions_text}")
PY
)"
    gate_action=""
    gate_enable=""
    gate_answer=""
    gate_question=""
    while IFS= read -r line; do
      case "$line" in
        action=*) gate_action="${line#action=}" ;;
        enable=*) gate_enable="${line#enable=}" ;;
        answer=*) gate_answer="${line#answer=}" ;;
        question=*) gate_question="${line#question=}" ;;
      esac
    done <<< "$gate_info"
    if [ "$gate_enable" = "1" ]; then
      rm -f "$session_skip_file"
    fi
    if [ -n "$gate_action" ]; then
      mkdir -p "$session_dir"
      printf "%s\n" "$gate_action" > "$gate_file"
      if [ "$gate_action" = "skip" ]; then
        printf "1\n" > "$session_skip_file"
      elif [ "$gate_action" = "apply" ]; then
        rm -f "$session_skip_file"
      fi
    fi
    if [ -n "$gate_answer" ]; then
      mkdir -p "$session_dir"
      printf "%s\n" "$gate_answer" > "$gate_answer_file"
    fi
    if [ -n "$gate_question" ]; then
      mkdir -p "$session_dir"
      printf "%s\n" "$gate_question" > "$gate_question_file"
    fi
    pending_run_dir="$(cat "$pending_run_file" 2>/dev/null || echo "")"
    prompt_id=""
    if [ -n "$pending_run_dir" ]; then
      run_dir="$pending_run_dir"
      run_prompt="${run_dir##*/}"
      if [[ "$run_prompt" == prompt-* ]]; then
        prompt_id="${run_prompt#prompt-}"
      fi
    else
      prompt_id="$(cat "$prompt_id_file" 2>/dev/null || echo "")"
      run_dir="${runs_dir}/${session_id}/prompt-${prompt_id}"
    fi
    if [ -z "$prompt_id" ]; then
      prompt_id="$(cat "$prompt_id_file" 2>/dev/null || echo "")"
    fi
    diff_file="${run_dir}/diff.patch"
    if [ -s "$diff_file" ]; then
      prompt_ts="$(cat "$prompt_ts_file" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")"
      prompt_cwd="$(cat "$cwd_file" 2>/dev/null || echo "")"
      gate_action="$(cat "$gate_file" 2>/dev/null || echo "")"
      gate_answer="$(cat "$gate_answer_file" 2>/dev/null || echo "")"
      gate_question="$(cat "$gate_question_file" 2>/dev/null || echo "")"
      codex_out="${run_dir}/review.codex.md"
      claude_out="${run_dir}/review.claude.md"
      log_md="${run_dir}/review.log.md"
      update_json=0
      if [ -f "$log_md" ]; then
        update_json=1
      fi
      write_run_log "$log_md" "$update_json"
      rm -f "$pending_run_file"
    fi
    exit 0
    ;;
  PreToolUse)
    if [ "$tool_name" != "Edit" ] && [ "$tool_name" != "Write" ]; then
      exit 0
    fi
    prompt_id="$(cat "$prompt_id_file" 2>/dev/null || echo "")"
    if [ -z "$prompt_id" ]; then
      exit 0
    fi
    run_dir="${runs_dir}/${session_id}/prompt-${prompt_id}"
    mkdir -p "$run_dir/before"
    if [ -z "$file_path" ]; then
      exit 0
    fi
    mkdir -p "$session_dir"
    touch "$files_file"
    if ! grep -Fxq "$file_path" "$files_file"; then
      printf "%s\n" "$file_path" >> "$files_file"
    fi
    snap_path="$run_dir/before/${file_path#/}"
    if [ ! -f "$snap_path" ]; then
      mkdir -p "$(dirname "$snap_path")"
      if [ -f "$file_path" ]; then
        cp -- "$file_path" "$snap_path"
      else
        : > "$snap_path"
      fi
    fi
    exit 0
    ;;
  Stop)
    prompt_id="$(cat "$prompt_id_file" 2>/dev/null || echo "")"
    if [ -z "$prompt_id" ]; then
      exit 0
    fi
    if [ -f "$session_skip_file" ]; then
      exit 0
    fi
    if [ ! -f "$files_file" ] || ! grep -q '.' "$files_file"; then
      exit 0
    fi
    run_dir="${runs_dir}/${session_id}/prompt-${prompt_id}"
    mkdir -p "$run_dir"
    diff_file="${run_dir}/diff.patch"
    : > "$diff_file"
    while IFS= read -r fp || [ -n "$fp" ]; do
      [ -n "$fp" ] || continue
      snap_path="${run_dir}/before/${fp#/}"
      if [ -f "$snap_path" ]; then
        if [ -f "$fp" ]; then
          diff -u --label "$fp (before)" --label "$fp (after)" "$snap_path" "$fp" >> "$diff_file" || true
        else
          diff -u --label "$fp (before)" --label "$fp (deleted)" "$snap_path" /dev/null >> "$diff_file" || true
        fi
      fi
    done < "$files_file"
    if [ ! -s "$diff_file" ]; then
      exit 0
    fi

    diff="$(cat "$diff_file")"
    prompt_ts="$(cat "$prompt_ts_file" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")"
    prompt_cwd="$(cat "$cwd_file" 2>/dev/null || echo "")"
    gate_action="$(cat "$gate_file" 2>/dev/null || echo "")"
    gate_answer="$(cat "$gate_answer_file" 2>/dev/null || echo "")"
    gate_question="$(cat "$gate_question_file" 2>/dev/null || echo "")"
    codex_out="${run_dir}/review.codex.md"
    claude_out="${run_dir}/review.claude.md"

    case "$backend" in
      codex)
        run_codex
        ;;
      claude)
        run_claude
        ;;
      both)
        run_codex
        run_claude
        ;;
      *)
        exit 0
        ;;
    esac

    log_md="${run_dir}/review.log.md"
    if [ "$block" = "2" ] && [ -z "$gate_action" ]; then
      printf "%s\n" "$run_dir" > "$pending_run_file"
    else
      write_run_log "$log_md"
      rm -f "$pending_run_file"
    fi

    if [ "$block" = "2" ] && [ -n "$gate_action" ]; then
      rm -f "$gate_file" "$gate_answer_file" "$gate_question_file"
      exit 0
    fi

    if [ "$block" = "2" ]; then
      review_question="$(build_review_question)"
      if [ "$backend" = "both" ]; then
        if [ -f "$codex_out" ]; then
          printf "CODEX REVIEW:\n%s\n" "$(cat "$codex_out")" >&2
        fi
        if [ -f "$claude_out" ]; then
          printf "CLAUDE REVIEW:\n%s\n" "$(cat "$claude_out")" >&2
        fi
      else
        review_file="$codex_out"
        if [ "$backend" = "claude" ]; then
          review_file="$claude_out"
        fi
        if [ -f "$review_file" ]; then
          cat "$review_file" >&2
        else
          printf "Review completed; see %s\n" "$log_md" >&2
        fi
      fi
      printf "\nACTION REQUIRED: Reply with 'review:apply' to act on this feedback, or 'review:skip' to skip reviews for the rest of this session. You can re-enable with 'review:enable' or 'review:resume'.\n" >&2
      printf "\nTOOL REQUEST: You MUST call AskUserQuestion now with this question: \"%s\" Do not proceed until the user answers.\n" "$review_question" >&2
      exit 2
    fi

    if [ "$block" = "1" ]; then
      if [ -f "$codex_out" ] && ! grep -qi '^BLOCKERS:[[:space:]]*none' "$codex_out"; then
        grep -i -m1 '^BLOCKERS:' "$codex_out" >&2 || true
        exit 2
      fi
      if [ -f "$claude_out" ] && ! grep -qi '^BLOCKERS:[[:space:]]*none' "$claude_out"; then
        grep -i -m1 '^BLOCKERS:' "$claude_out" >&2 || true
        exit 2
      fi
    fi
    ;;
  *)
    exit 0
    ;;
esac

exit 0
