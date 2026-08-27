# AGENTS.md

This is an assessed IT5006 team project. AI agents are helpers, not owners of the repository.

## Read first

1. Read `README.md`.
2. Read `docs/project-requirements.md` before making scope, deadline or deliverable decisions.
3. Use the official course page as the final authority when accessible:
   https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables

## Keep the lecturer-required structure

Use these top-level folders:

- `data/` — datasets
- `notebooks/` — EDA, modelling and experiments
- `src/` — reusable `.py` code
- `deployment/` — Streamlit/FastAPI/Gradio apps, Power BI or other deployable dashboard files
- `docs/` — reports, slides and project documentation

Do not create new top-level folders unless a human teammate explicitly asks for one.

## Repository safety

- Do normal work on a separate branch and open a pull request.
- Do not overwrite or delete unrelated teammate work.
- Do not delete or rename the required top-level folders.
- Do not rewrite Git history or force-push unless explicitly approved by a human maintainer.
- Do not merge into `main` unless explicitly asked.
- Never commit passwords, tokens, API keys, private keys or other secrets.
- Keep changes focused and place files in the correct existing folder.

## Data and analysis

- Keep original source data unchanged where practical.
- Do not fabricate data, results, metrics, experiments, citations or execution evidence.
- Prevent target leakage.
- Do not claim code, notebooks, models or deployments were tested unless they were actually run or checked.
- Update `requirements.txt` when new Python packages are required.

## Academic work

- Humans must review and understand AI-assisted work before merge or submission.
- AI usage must be declared or credited where required by the course.
- Important analytical choices and conclusions must be checked by the team.

## IT5006 constraints to remember

- Phase 1 due: **13 Sep 2026, 23:59**.
- Phase 2 due: **11 Oct 2026, 23:59**.
- Presentation deck locked: **4 Nov 2026, 23:59**.
- Final report + deployment ZIP due: **8 Nov 2026, 23:59**.
- Project must cover both classification and regression.
- Use 1–2 analytics problems maximum.
- Use 2–3 model families total across the project.
- Include the GitHub repository link in report submissions.

For full details, use `docs/project-requirements.md`.
