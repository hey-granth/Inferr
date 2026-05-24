# Inferr shell integration — source this in ~/.zshrc
# Sends command + output to Inferr's capture endpoint

_inferr_url="${INFERR_URL:-http://127.0.0.1:7331}"
_inferr_enabled=1

_inferr_preexec() {
    _inferr_last_cmd="$1"
    _inferr_cmd_start=$SECONDS
}

_inferr_precmd() {
    local exit_code=$?
    [[ -z "$_inferr_last_cmd" ]] && return
    [[ "$_inferr_enabled" != "1" ]] && return

    local payload
    payload=$(printf '{"command":%s,"exit_code":%d}' \
        "$(echo -n "$_inferr_last_cmd" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
        "$exit_code")

    curl -s -X POST "${_inferr_url}/capture/command" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 0.5 \
        --silent \
        --fail \
        2>/dev/null &

    _inferr_last_cmd=""
}

autoload -Uz add-zsh-hook
add-zsh-hook preexec _inferr_preexec
add-zsh-hook precmd _inferr_precmd
