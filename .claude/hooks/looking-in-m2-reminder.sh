#!/bin/bash
# PreToolUse hook: if claude tries to read ~/.m2 remind it to stop
INPUT="$(cat)"
command="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"

lower="$(printf '%s' "$command" | tr '[:upper:]' '[:lower:]')"

# if the cmd contains /.m2 (e.g. ~/.m2 or /usr/home/.m2)
if printf '%s' "$lower" | grep -Eq '/.m2'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Reminder: you can check the docs online rather than diving into the m2 dir. Check the pom for versions."
    }
  }'
  exit 0
else
  exit 0  # no decision; normal permission flow applies
fi
