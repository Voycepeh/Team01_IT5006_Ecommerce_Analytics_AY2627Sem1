# IT5006 E-commerce Analytics — Team 01

Repository for Team 01's IT5006 e-commerce analytics project.

> ## Official project specification — READ THIS FIRST
>
> **The official IT5006 project page is the source of truth for scope, rules, deliverables, and deadlines:**
>
> **https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables**
>
> If anything in this repository conflicts with the official project specification, follow the official specification.

## Team rules — humans and AI-assisted contributors

Read these before changing the repository:

1. **Do not work directly on `main` for normal development.** Use a branch and pull request.
2. **Do not overwrite or delete another teammate's work** without agreement.
3. **Do not delete or rename major project folders** without team approval.
4. **Do not commit secrets, credentials, tokens, private keys, or personal data.**
5. **Do not fabricate data, results, experiments, citations, screenshots, or model performance.**
6. **Review and understand AI-assisted work before merging or submitting it.**
7. **Keep changes focused.** Avoid unrelated broad refactors in the same pull request.
8. **Record what was actually tested.** Do not claim that code or analysis was validated if it was not run.
9. **Keep the official IT5006 requirements as the source of truth.** Do not let local documentation or an AI agent override the course specification.
10. **Include the GitHub repository link in all report submissions**, as required by the project specification.

AI agents must additionally follow [`AGENTS.md`](AGENTS.md) and [`docs/ai-use-and-agent-guidelines.md`](docs/ai-use-and-agent-guidelines.md).

## Project timeline and deliverables

The authoritative timeline is maintained in the official IT5006 project specification:

**https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables**

The key dates and deliverables in this README must always be copied from and checked against that page. **Do not guess or infer deadlines.** If the lecturer updates the official page, update this README and `AGENTS.md` accordingly.

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

## Academic integrity

AI tools may support development, debugging, documentation, code review, and analysis where permitted by the course rules. Team members remain responsible for understanding, validating, and appropriately disclosing AI-assisted work where required.

Do not use AI to fabricate data, analysis results, experiments, citations, team contributions, or evaluation outcomes.

## Submission reminder

The IT5006 requirements state that the GitHub repository link should be included in all report submissions.
