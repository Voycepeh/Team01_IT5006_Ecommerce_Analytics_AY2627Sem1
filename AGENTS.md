# AGENTS.md

## Mandatory source of truth

Before making any change, read the project requirements in this order:

1. **Readable repository version:** [`docs/project-requirements.md`](docs/project-requirements.md)
2. **Official live course page:** https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables
3. **Archived original PDF:** [`docs/Project Requirements.pdf`](docs/Project%20Requirements.pdf)

Use the Markdown version for day-to-day work because it is easier for humans and agents to search and parse. The latest official course page or Canvas announcement remains authoritative if any local copy differs.

Agents must not invent, infer, or silently alter project deadlines or deliverables.

## Human-facing team rules that agents must enforce

1. Do not work directly on `main` for normal development. Use a focused branch and pull request.
2. Do not overwrite or delete another teammate's work without agreement.
3. Do not delete or rename major project folders without team approval.
4. Do not commit secrets, credentials, tokens, private keys, or personal data.
5. Do not fabricate data, results, experiments, citations, screenshots, deployment evidence, or model performance.
6. AI-assisted work must be reviewed and understood by a human teammate before merge or submission.
7. Keep changes focused and avoid unrelated broad refactors.
8. Do not claim code, analysis, tests, deployment, or outputs were validated unless they were actually run or checked.
9. Keep the official IT5006 specification as the source of truth.
10. Ensure the GitHub repository link is included in all report submissions.

## Official milestone schedule

| Milestone | Due date | Deliverable | Weight |
| --- | --- | --- | ---: |
| Team formation | Sun, 23 Aug 2026 | Canvas team signup | — |
| Phase 1: Literature Survey & EDA | Sun, 13 Sep 2026, 23:59 | Report + dashboard + GitHub link | 20% |
| Recess / problem scoping | 19–27 Sep 2026 | Team problem scoping | — |
| Phase 2: Problem Definition & Modelling | Sun, 11 Oct 2026, 23:59 | Technical report + code; 1–2 problems; 2–3 model families total | 40% |
| Phase 3: Presentation slides locked | Wed, 4 Nov 2026, 23:59 | Final deck (PDF/PPT) | Part of Phase 3 |
| Phase 3: Deployment & Final Report | Sun, 8 Nov 2026, 23:59 | Report + deployment + slides + GitHub link in ZIP | 40% combined Phase 3 |
| Live presentation | Thu, 5 Nov or Thu, 12 Nov 2026 | 10 min presentation + 5 min Q&A | Part of Phase 3 |
| Peer evaluation | Sun, 15 Nov 2026, 23:59 | Individual assessment | Part of Phase 3 |

Teaching period: 10 Aug–13 Nov 2026. Class: Thursdays, 6:30–9:30 PM. Reading week: 14–20 Nov 2026. Examinations: 21 Nov–5 Dec 2026.

## Deliverable constraints agents must respect

### Phase 1

- 4–5 page report excluding cover, references, and appendices.
- Literature review and EDA.
- Interactive dashboard using Streamlit, Tableau Public, or Power BI.
- Combined PDF must include dashboard link and GitHub repository link.
- GitHub repository contains raw code/notebooks.

### Phase 2

- 6–8 page report excluding cover, references, and appendices.
- Define 1 or 2 analytics problems maximum.
- Project must cover both classification and regression.
- Use 2–3 model families total across the project; prioritise quality over quantity.
- Use disciplined train/test or train/validation/test methodology, cross-validation, leakage prevention, reproducible random seeds, suitable metrics, interpretability, and model comparison.
- Classification metrics may include Precision, Recall, F1, ROC-AUC / PR-AUC as appropriate.
- Regression metrics may include MAE, RMSE, and R² as appropriate.
- Technical report must include GitHub repository link; repository should include notebooks/Python scripts and model-performance summaries.

### Phase 3

- Final report: 15–20 pages excluding cover, references, and appendices.
- Deploy a POC/MVP through Streamlit Cloud, FastAPI, Gradio, or similar.
- Live URL, clear usage instructions, edge-case handling, and no hard-coded credentials.
- At least one deployed model from the project.
- Presentation deck is locked at the 4 Nov deadline and must match the version used live and included in the final ZIP.
- Final ZIP is due 8 Nov and includes final report PDF, locked presentation slides, and GitHub repository link.
- Live presentation is 10 minutes plus 5 minutes Q&A.

## Course-wide technical and academic rules

- Use the Olist Brazilian E-Commerce dataset as the primary required data source.
- Join and document relevant tables for the chosen problem(s).
- For customer-level analysis, verify whether `customer_id` or `customer_unique_id` is the correct identifier before feature engineering.
- Prevent target leakage. Do not use information that is only available after the prediction point.
- For imbalanced classification, use suitable metrics/handling and do not rely on accuracy alone.
- Prefer scikit-learn Pipelines where possible, especially to keep preprocessing inside cross-validation folds.
- Start with the simplest model variant within a family and justify added complexity with evidence.
- Set and document random seeds for reproducibility.
- Document transformations, assumptions, error handling, and validation checks.
- Required tooling includes Python, a dashboard tool, a deployment platform, GitHub, and documented notebooks.
- Deployment must be tested before submission.
- Cite all sources appropriately. Original analysis and interpretation are required.
- Collaboration within the team is encouraged; collaboration between teams is discouraged.
- AI tools must be declared and appropriately credited.
- Reports are PDF, 12pt font, single-spaced, with figure captions/table labels and consistent formatting.
- Late policy: 0–24h = 10% penalty; 24–48h = 20%; more than 48h = 50%; more than 7 days = zero marks.

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
2. Read `docs/project-requirements.md` fully for project-scope work.
3. Check the official live project page when accessible, especially for deadlines or changed instructions.
4. Use `docs/Project Requirements.pdf` only as the archived original when needed.
5. Read `docs/ai-use-and-agent-guidelines.md`.
6. Inspect existing files before creating replacements.
7. Understand which project deliverable the proposed change supports.
8. Prefer editing existing files over creating duplicate alternatives.

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

## Academic integrity and AI use

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
