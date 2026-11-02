# imagehub Iteration Loop

This file owns the iteration process. Product requirements are in `.ai/GOAL.md`; repository
rules are in `.ai/AGENTS.md`.

## Start every iteration

1. Read `.ai/AGENTS.md`, `.ai/GOAL.md`, this file, new maintainer messages and `git status`.
2. Record the branch, the commit you start from and any uncommitted changes that are not yours.
   Never discard changes you did not make.
3. Read the skill that `.ai/AGENTS.md` routes the task to. For reviews, read
   `.ai/review-rules.md`.
4. When a decision depends on upstream behavior (a reference library, a checkpoint, a Hub
   API), verify it against the exact installed version or pinned revision and write down the
   version or commit.

## Select one gap

Pick the single highest-impact bounded gap, in this order:

1. Broken public behavior, data or download safety, packaging, or a failing supported-platform
   check.
2. A parity regression: any recorded measurement that no longer reproduces.
3. Shared-core changes that the next roadmap model needs (for example non-persistent buffers
   under meta-device construction, new checkpoint container layouts, configurable tokenizer
   special tokens).
4. The next model in the `.ai/GOAL.md` roadmap.
5. Missing or inaccurate documentation: catalog rows, model pages, validation summaries,
   claims that go beyond the recorded evidence.
6. Simplifications: duplicated code, unused configuration, inconsistent error types.

One iteration delivers one gap. Note other findings for later instead of widening scope.

## Design before code

For a new model or any change to a public API, write a short design in
`docs/design/` and an implementation plan in `docs/design/plans/` before
writing code. The plan contains the exact code and tests for each task and is reviewed before
execution. Small fixes need only a clear commit message.

## Implement

- Work on a feature branch. Write the failing test first, watch it fail, then implement.
- Keep commits small and meaningful: one task or one fix per commit.
- Prefer verified prototypes: when porting numerics, prototype outside the repository, compare
  against the reference with random weights, and only then put the code in the plan.
- Keep a progress ledger for multi-task work in a git-ignored workspace so it survives context
  resets.

## Verify

Before claiming completion, run and read the output of:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy
```

For model or numerics changes, also run the reference tests that cover the change against the
real snapshots, and regenerate `docs/validation.json` if a recorded measurement changed. Never
report a check as passed if it was not run.

## Deliver

1. Review the diff against `.ai/review-rules.md` (or have a reviewer do it) and fix Critical
   and Important findings.
2. Open a pull request with an English summary, what changed and the verification results.
   Ask the maintainer before pushing to `main` or merging.
3. Update `.ai/GOAL.md` (the roadmap) when a milestone item is finished.
