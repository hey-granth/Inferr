# Inferr shell integration — source this in ~/.bashrc
# Captures command name, exit code, and real stdout/stderr output

_inferr_url="${INFERR_URL:-http://127.0.0.1:7331}"
_inferr_tmp=""
_inferr_last_cmd=""

_inferr_preexec() {
    _inferr_last_cmd="$BASH_COMMAND"
    _inferr_tmp="$(mktemp /tmp/inferr_out.XXXXXX 2>/dev/null)"
}

_inferr_capture() {
    local exit_code=$?
    [[ -z "$_inferr_last_cmd" ]] && return

    local output=""
    local output_lines=0

    if [[ -n "$_inferr_tmp" && -f "$_inferr_tmp" ]]; then
        output="$(cat "$_inferr_tmp" 2>/dev/null)"
        output_lines="$(wc -l < "$_inferr_tmp" 2>/dev/null || echo 0)"
        rm -f "$_inferr_tmp"
        _inferr_tmp=""
    fi

    # Truncate to 8KB
    if [[ ${#output} -gt 8192 ]]; then
        output="${output:0:8192}"
    fi

    local payload
    payload=$(python3 -c "
import json, sys
cmd = sys.argv[1]
out = sys.argv[2]
lines = int(sys.argv[3])
exit_c = int(sys.argv[4])
print(json.dumps({'command': cmd, 'exit_code': exit_c, 'output': out, 'output_lines': lines}))
" "$_inferr_last_cmd" "$output" "$output_lines" "$exit_code" 2>/dev/null)

    if [[ -n "$payload" ]]; then
        curl -s -X POST "${_inferr_url}/capture/output" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            --max-time 0.5 \
            --silent \
            --fail \
            2>/dev/null &
    fi

    _inferr_last_cmd=""
}

trap '_inferr_preexec' DEBUG
PROMPT_COMMAND="_inferr_capture${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
