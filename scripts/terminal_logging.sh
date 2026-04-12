#!/usr/bin/env bash

# 이미 활성화되어 있으면 중복 방지
if [[ -n "$PROJECT_TERMINAL_LOGGING_ACTIVE" ]]; then
    return 0 2>/dev/null || exit 0
fi
export PROJECT_TERMINAL_LOGGING_ACTIVE=1

# interactive shell 에서만 동작
case $- in
    *i*) ;;
    *) return 0 2>/dev/null || exit 0 ;;
esac

# 프로젝트 루트 결정
if [[ -n "$PROJECT_WORKSPACE_FOLDER" ]]; then
    project_root="$PROJECT_WORKSPACE_FOLDER"
elif project_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    project_root="$PWD"
fi

log_dir="$project_root/logs/terminal"
mkdir -p "$log_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"
host="$(hostname 2>/dev/null | tr ' ' '_' || echo unknown_host)"
user_name="$(whoami 2>/dev/null || echo unknown_user)"
session_id="${timestamp}_${host}_${user_name}"

log_file="$log_dir/${session_id}.txt"

export PROJECT_TERMINAL_LOG_FILE="$log_file"
export PROJECT_TERMINAL_SESSION_ID="$session_id"

# 첫 시작 안내
{
    printf '================================================================\n'
    printf '[terminal logging started]\n'
    printf 'SESSION : "%s"\n' "$PROJECT_TERMINAL_SESSION_ID"
    printf 'LOG     : "%s"\n' "$PROJECT_TERMINAL_LOG_FILE"
    printf '================================================================\n\n'
} >> "$PROJECT_TERMINAL_LOG_FILE"

# stdout/stderr 전체 저장
exec > >(tee -a "$PROJECT_TERMINAL_LOG_FILE") 2>&1

# 마지막으로 기록한 history 번호
export __PROJECT_LAST_LOGGED_HISTNO=""

__project_log_command() {
    local exit_code hist_entry hist_no cli cmd cwd

    exit_code=$?
    cwd="$(pwd)"

    history -a
    history -n

    hist_entry="$(history 1)"
    hist_no="${hist_entry%% *}"
    cli="${hist_entry#* }"

    if [[ -z "$hist_no" || "$hist_no" == "$__PROJECT_LAST_LOGGED_HISTNO" ]]; then
        return $exit_code
    fi

    __PROJECT_LAST_LOGGED_HISTNO="$hist_no"

    # 첫 단어를 CMD로 사용
    cmd="${cli%% *}"

    {
        printf '\n================================================================\n'
        printf '[%s]\n' "$(date '+%F %T')"
        printf 'CWD     : "%s"\n' "$cwd"
        printf 'CMD     : "%s"\n' "$cmd"
        printf 'CLI     : "%s"\n' "$cli"
        printf '⇒\n'
    } >> "$PROJECT_TERMINAL_LOG_FILE"

    return $exit_code
}

__project_log_footer() {
    {
        printf '\n================================================================\n'
    } >> "$PROJECT_TERMINAL_LOG_FILE"
}
PROMPT_COMMAND="__project_log_command; __project_log_footer${PROMPT_COMMAND:+;$PROMPT_COMMAND}"

echo "[project terminal logging started]"
echo "[log file] $PROJECT_TERMINAL_LOG_FILE"

