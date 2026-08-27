# AGENTS.md

## Purpose

This repository is an assessed IT5006 team project. AI agents may assist the team, but they must behave as controlled contributors rather than autonomous owners of the repository.

## Source of truth

Before making changes, read the official IT5006 project specification:

https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables

If repository documentation conflicts with the official project specification, the official project specification wins.

## Required repository structure

Preserve these top-level project folders unless the team explicitly agrees otherwise:

- `data/`
- `notebooks/`
- `src/`
- `deployment/`
- `docs/`

Supporting files such as `.github/`, `.gitignore`, `AGENTS.md`, and `requirements.txt` are allowed.

## Before changing anything

1. Read `README.md`.
2. Read `docs/project-requirements.md`.
3. Read `docs/ai-use-and-agent-guidelines.md`.
4. Inspect existing files before creating replacements.
5. Understand which project deliverable the proposed change supports.
6. Prefer editing existing files over creating duplicate alternatives.

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

If a requested change could remove substantial work, alter project scope, affect assessment integrity, or break compatibility with teammates' work, stop and ask for human confirmation before proceeding.
