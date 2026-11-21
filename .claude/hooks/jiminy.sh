#!/usr/bin/env bash
# Command-hook body for Claude Code. Claude Code writes the hook event JSON to
# this script's stdin; we forward it to the JIMINY service and echo the
# service's JSON response back on our stdout so Claude Code consumes it.
#
# Register in .claude/settings.json:
#   "UserPromptSubmit": [{ "hooks": [{ "type":"command", "command":"/abs/path/hook.sh", "timeout":30 }] }]
#   "PreToolUse": [{ "matcher":"Bash", "hooks": [{ "type":"command", "command":"/abs/path/hook.sh", "timeout":30 }] }]
#
# Required environment:
#   JIMINY_URL    e.g. http://localhost:5000   (or http://host.docker.internal:5000)
#   JIMINY_TOKEN  the service's bearer token (empty if the service has auth disabled)

set -euo pipefail

JIMINY_TOKEN=jiminy-dev-token
JIMINY_URL=http://tirion.tail039f38.ts.net:5551

: "${JIMINY_URL:?set JIMINY_URL, e.g. http://tirion.tail039f38.ts.net:5551}"

INPUT="$(cat)"

AUTH_HEADER=()
if [ -n "${JIMINY_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${JIMINY_TOKEN}")
fi

RESPONSE="$(curl -s -X POST "${JIMINY_URL}/hook" \
  -H 'Content-Type: application/json' \
  -H "User-Agent: jiminy-claude-code (`whoami`)" \
  "${AUTH_HEADER[@]}" \
  -d "$INPUT")"

# Empty response means "no decision, take no action" — but print valid JSON in
# case a downstream consumer expects a parseable body.
echo "${RESPONSE:-{\}}"
