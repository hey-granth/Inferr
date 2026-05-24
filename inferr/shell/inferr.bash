# Inferr shell integration — source this in ~/.bashrc
# Sends command to Inferr's capture endpoint

_inferr_url="${INFERR_URL:-http://127.0.0.1:7331}"

_inferr_capture() {
    local exit_code=$?
    local last_cmd
    last_cmd=$(history 1 | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//')
    [[ -z "$last_cmd" ]] && return

    local payload
    payload=$(printf '{"command":%s,"exit_code":%d}' \
        "$(echo -n "$last_cmd" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
        "$exit_code")

    curl -s -X POST "${_inferr_url}/capture/command" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 0.5 \
        --silent \
        --fail \
        2>/dev/null &
}

PROMPT_COMMAND="_inferr_capture${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
