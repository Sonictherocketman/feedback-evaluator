# Repo rules

## Python: feedback_evaluator/

After changing **any `*.py` file** under `feedback_evaluator/`
(package code *or* tests), always run both checks from the **repo root**
and make them pass before finishing:

```bash
make format   # black, 79 cols — config in feedback_evaluator/pyproject.toml
make test     # pytest suite (offline; the agent is never called)
make coverage # The coverage suite. Keep it above 80%.
```

Write new code already black-clean (79-col lines) so `make format` is a
no-op. `make format-check` must not report diffs.
