#!/usr/bin/env bash
# Block mutating Azure CLI commands
set -euo pipefail

cmd=$(jq -r '.tool_input.command' <<<"$(cat)")

if [[ "$cmd" == az\ * ]] && echo "$cmd" | grep -qiE '\b(create|delete|update|stop|restart|assign|remove|move|start|tag)\b'; then
  echo "{\"continue\": false, \"systemMessage\": \"BLOCKED: mutating Azure CLI command ($cmd). Only read-only az commands (list, show, get-*) are allowed in this project.\", \"hookSpecificOutput\": {\"hookEventName\": \"PreToolUse\", \"permissionDecision\": \"deny\", \"permissionDecisionReason\": \"Mutating az verb detected\"}}"
fi
