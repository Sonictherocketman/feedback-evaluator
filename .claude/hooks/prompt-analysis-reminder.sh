#!/bin/bash
# Fires on every submitted prompt, injects an invisible context reminder.
INPUT="$(cat)"
prompt="$(echo "$INPUT" | jq -r '.prompt // empty')"

lower="$(printf '%s' "$prompt" | tr '[:upper:]' '[:lower:]')"

if printf '%s' "$lower" | grep -q 'analyze\|analyse\|analysis' \
    && printf '%s' "$lower" | grep -q 'data'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: "Reminder: If you are being asked for data analysis, you have R available to you (via Rscript). You DO NOT have pandas and never attempt to install it."
    }
  }'
else
  exit 0  # no decision; nothing injected
fi
