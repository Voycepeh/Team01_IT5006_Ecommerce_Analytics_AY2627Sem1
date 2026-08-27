# IT5006 E-commerce Analytics — Team 01

Repository for Team 01's IT5006 e-commerce analytics project.

## Source of truth

The official IT5006 project specification is the source of truth for requirements, deliverables, and timelines:

https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables

If anything in this repository conflicts with the official project specification, follow the official specification.

## Required repository structure

```text
project-root/
├── data/
├── notebooks/
├── src/
├── deployment/
├── docs/
└── README.md
```

This repository also includes supporting files such as `AGENTS.md`, `.gitignore`, `.github/`, and `requirements.txt` to make collaboration safer and reproducible.

## Folder guide

- `data/` — project datasets and documented data handling conventions.
- `notebooks/` — exploratory analysis, experiments, and notebook-based analysis.
- `src/` — reusable Python/source code used by notebooks or deployment.
- `deployment/` — deployment artefacts, configuration templates, and serving-related files when required.
- `docs/` — project documentation, decisions, requirements notes, and team/AI usage guidance.

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install project dependencies with:

```bash
pip install -r requirements.txt
```

4. Follow any task-specific instructions documented in the relevant folder or notebook.

Dependencies will be added to `requirements.txt` as the project implementation develops.

## Collaboration workflow

- Do not work directly on `main` for normal development.
- Create a focused branch for each change.
- Keep changes small enough for teammates to review.
- Open a pull request before merging into `main`.
- Review generated or AI-assisted work before merging.
- Never commit credentials, tokens, passwords, or private keys.
- Do not silently replace another teammate's work.

AI agents must also follow `AGENTS.md` and `docs/ai-use-and-agent-guidelines.md`.

## Academic integrity

AI tools may support development, debugging, documentation, code review, and analysis where permitted by the course rules. Team members remain responsible for understanding, validating, and appropriately disclosing AI-assisted work where required.

Do not use AI to fabricate data, analysis results, experiments, citations, team contributions, or evaluation outcomes.

## Submission reminder

The IT5006 requirements state that the GitHub repository link should be included in all report submissions.
