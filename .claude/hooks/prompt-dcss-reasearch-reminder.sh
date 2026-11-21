#!/bin/bash
# Fires on every submitted prompt. If the prompt contains EITHER "read up" and
# "research" (case-insensitive), inject an invisible context reminder.
INPUT="$(cat)"
prompt="$(echo "$INPUT" | jq -r '.prompt // empty')"
cwd="$(echo "$INPUT" | jq -r '.cwd // empty')"
rcwd="$(realpath $cwd)"

skyrocket_dcss="skyrocket/sdcounty"

lower="$(printf '%s' "$prompt" | tr '[:upper:]' '[:lower:]')"
lower_cwd="$(printf '%s' "$rcwd" | tr '[:upper:]' '[:lower:]')"

if printf '%s' "$lower" | grep -q 'read up\|research' \
    && printf '%s' "$lower_cwd" | grep -q "$skyrocket_dcss"; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: "Reminder: The canonical design docs are in Slite. Check there first."
    }
  }'
else
  exit 0  # no decision; nothing injected
fi
