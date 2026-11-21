#!/bin/bash
# PreToolUse hook: deny any Bash command that would unzip a JAR,
# and hand Claude the reason it should use the JDK instead.
INPUT="$(cat)"
command="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"

lower="$(printf '%s' "$command" | tr '[:upper:]' '[:lower:]')"

# Two greps: an "unzip" signal AND a ".jar" signal, in the same command.
if printf '%s' "$lower" | grep -Eq 'unzip|extract' \
   && printf '%s' "$lower" | grep -q '\.jar'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Reminder: please do not unzip random jars. If you need to check method signatures please check the documentation."
    }
  }'
  exit 0
else
  exit 0  # no decision; normal permission flow applies
fi
