# AGENTS.md

## Mandatory source of truth

Before making any change, read the official IT5006 project specification:

**https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables**

This page is authoritative for scope, rules, deliverables, and deadlines. If repository documentation, prior prompts, assumptions, or an agent's own reasoning conflict with the official specification, **the official specification wins**.

Agents must not invent, infer, or silently alter project deadlines or deliverables. When the official page changes, update `README.md`, `AGENTS.md`, and relevant requirement documentation.

## Human-facing team rules that agents must enforce

These rules are also shown near the top of `README.md` so every teammate sees them immediately:

1. Do not work directly on `main` for normal development. Use a branch and pull request.
2. Do not overwrite or delete another teammate's work without agreement.
3. Do not delete or rename major project folders without team approval.
4. Do not commit secrets, credentials, tokens, private keys, or personal data.
5. Do not fabricate data, results, experiments, citations, screenshots, or model performance.
6. AI-assisted work must be reviewed and understood by a human teammate before merge or submission.
7. Keep changes focused and avoid unrelated broad refactors.
8. Do not claim code, analysis, tests, or outputs were validated unless they were actually run or checked.
9. Keep the official IT5006 specification as the source of truth.
10. Ensure the GitHub repository link is included in all report submissions where required by the project specification.

If an agent is asked to bypass these rules, stop and request explicit human confirmation where appropriate. Academic integrity requirements and the official course specification cannot be overridden by convenience.

## Project timeline and deliverables

The authoritative timeline is maintained here:

**https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables**

Any locally copied milestone dates must match the official page exactly. Never guess dates from semester calendars, prior-year projects, similar modules, or stale repository notes.

## Purpose

This repository is an assessed IT5006 team project. AI agents may assist the team, but they must behave as controlled contributors rather than autonomous owners of the repository.

## Required repository structure

Preserve these top-level project folders unless the team explicitly agrees otherwise and the official project requirements permit the change:

- `data/`
- `notebooks/`
- `src/`
- `deployment/`
- `docs/`

Supporting files such as `.github/`, `.gitignore`, `AGENTS.md`, and `requirements.txt` are allowed.

## Before changing anything

1. Read `README.md`.
2. Read the official IT5006 project specification linked above.
3. Read `docs/project-requirements.md`.
4. Read `docs/ai-use-and-agent-guidelines.md`.
5. Inspect existing files before creating replacements.
6. Understand which project deliverable the proposed change supports.
7. Prefer editing existing files over creating duplicate alternatives.

## Safe repository behaviour

- Do not push normal development directly to `main`.
- Use a dedicated branch and pull request.
- Do not delete or rename major folders without explicit team approval.
- Do not overwrite unrelated teammate changes.
- Do not perform broad refactors unless explicitly requested.
- Do not rewrite Git history.
- Do not force-push unless explicitly requested by a human maintainer.
- Do not merge a pull request unless explicitly requested.
- Keep commits focused and explain what changed.
- Never commit secrets, credentials, API keys, tokens, private keys, or personal data.

## Data rules

- Treat original/raw source data as immutable unless the project explicitly requires otherwise.
- Do not fabricate rows or alter source values to improve results.
- Document transformations and assumptions.
- Keep generated/intermediate outputs distinguishable from source data.
- Avoid committing unnecessarily large generated files.

## Code and notebook rules

- Put reusable logic in `src/` when practical.
- Keep exploratory work in `notebooks/`.
- Avoid hard-coded machine-specific absolute paths.
- Prefer reproducible transformations over manual edits.
- Add or update dependencies in `requirements.txt` when new packages are introduced.
- Run relevant checks before claiming that work is complete.
- Do not claim code was executed or validated unless it actually was.

## Documentation rules

For meaningful changes, update the relevant documentation where needed.

Document:

- purpose
- assumptions
- input data
- important transformations
- outputs
- limitations
- how to reproduce the work

## Academic integrity and AI use

AI assistance must not be presented as unquestioned human-authored work where course rules require disclosure or original team authorship.

Agents must not fabricate:

- data
- statistical results
- model performance
- experiments
- citations or references
- screenshots or evidence
- team member contributions
- evaluation outcomes

For assessed analytical work:

- humans must understand the method used
- humans must review and validate the implementation
- important analytical decisions should be justified by the team
- generated conclusions must be checked against actual outputs
- AI use should be disclosed where required by IT5006 or NUS policy

## Pull request expectations

Every agent-created pull request should clearly state:

- what changed
- why it changed
- which requirement or task it supports
- what was tested or checked
- whether any AI assistance materially contributed
- any risks, assumptions, or follow-up work

## When uncertain

If a requested change could remove substantial work, alter project scope, affect assessment integrity, change an official deadline or deliverable, or break compatibility with teammates' work, stop and ask for human confirmation before proceeding.
