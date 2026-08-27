# IT5006 E-commerce Analytics — Team 01

Repository for Team 01's IT5006 e-commerce analytics project.

> ## Official project specification — READ THIS FIRST
>
> **The official IT5006 project page is the source of truth for scope, rules, deliverables, and deadlines:**
>
> **https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables**
>
> This README summarises the official specification for convenience. If anything here conflicts with the official page or a later Canvas announcement, follow the official course source.

## Team rules — read before contributing

1. **Do not work directly on `main` for normal development.** Use a focused branch and pull request.
2. **Do not overwrite or delete another teammate's work** without agreement.
3. **Do not delete or rename major project folders** without team approval.
4. **Do not commit secrets, credentials, tokens, private keys, or personal data.**
5. **Do not fabricate data, results, experiments, citations, screenshots, or model performance.**
6. **Review and understand AI-assisted work before merging or submitting it.** AI use must be declared and appropriately credited where required.
7. **Keep changes focused.** Avoid unrelated broad refactors in the same pull request.
8. **Record what was actually tested.** Never claim code, analysis, deployment, or outputs were validated if they were not run or checked.
9. **Respect the official IT5006 scope.** Do not let local notes or an AI agent override the course specification.
10. **Include this GitHub repository link in all report submissions.**

AI agents must additionally follow [`AGENTS.md`](AGENTS.md) and [`docs/ai-use-and-agent-guidelines.md`](docs/ai-use-and-agent-guidelines.md).

## Key dates and deliverables

| Milestone | Due date | Deliverable | Weight |
| --- | --- | --- | ---: |
| Team formation | **Sun, 23 Aug 2026** | Canvas team signup | — |
| Phase 1: Literature Survey & EDA | **Sun, 13 Sep 2026, 23:59** | Report + interactive dashboard + GitHub repository link | **20%** |
| Recess / problem scoping | **19–27 Sep 2026** | Finalise team problem scope informally | — |
| Phase 2: Problem Definition & Modelling | **Sun, 11 Oct 2026, 23:59** | Technical report + code; 1–2 problems; 2–3 model families total | **40%** |
| Phase 3: Presentation slides locked | **Wed, 4 Nov 2026, 23:59** | Final presentation deck (PDF/PPT), same deck used for live presentation and final ZIP | Part of Phase 3 |
| Phase 3: Deployment & Final Report | **Sun, 8 Nov 2026, 23:59** | Final report + deployment + presentation slides + GitHub repository link in ZIP | **40% combined Phase 3** |
| Live presentation | **Thu, 5 Nov or Thu, 12 Nov 2026** | 10-minute presentation + 5-minute Q&A during class | Part of Phase 3 |
| Peer evaluation | **Sun, 15 Nov 2026, 23:59** | Individual peer assessment | Part of Phase 3 |

**Teaching period:** 10 Aug–13 Nov 2026, Weeks 1–13. **Class:** Thursdays, 6:30–9:30 PM. **Reading week:** 14–20 Nov 2026. **Examinations:** 21 Nov–5 Dec 2026.

### Phase 1 — Literature Survey & Exploratory Data Analysis

**Deadline:** Sun, 13 Sep 2026 at 23:59 · **Weight:** 20% · **Report length:** 4–5 pages excluding cover, references, and appendices.

Key deliverables:

- Literature review of relevant e-commerce analytics approaches, modelling methods, feature engineering, evaluation methods, and key gaps/opportunities.
- EDA covering dataset overview, temporal patterns, customer/seller/product analysis, correlation/relationship exploration, candidate problem exploration, and key insights.
- Interactive dashboard using Streamlit (recommended), Tableau Public, or Power BI.
- Combined PDF report containing the dashboard link and GitHub repository link.
- GitHub repository containing raw code/notebooks.

### Phase 2 — Problem Definition, Model Building & Evaluation

**Deadline:** Sun, 11 Oct 2026 at 23:59 · **Weight:** 40% · **Report length:** 6–8 pages excluding cover, references, and appendices.

Key deliverables and constraints:

- Define **1 or 2 analytics problems maximum**.
- The project must cover **both classification and regression**; this can be two problems or one problem with dual framing.
- Use a **2–3 model-family budget across the entire project**, focusing on quality over quantity.
- Use disciplined train/test or train/validation/test methodology, cross-validation, reproducible random seeds, leakage prevention, suitable metrics, interpretability, and model comparison.
- Classification metrics may include Precision, Recall, F1, ROC-AUC / PR-AUC as appropriate for imbalance.
- Regression metrics may include MAE, RMSE, and R² as appropriate.
- Submit technical report PDF with GitHub repository link, Jupyter notebooks/Python scripts, and model-performance summary tables.

### Phase 3 — Integration & Communication

**Combined weight:** 40% for deployment/final report, live presentation, and peer evaluation together.

Key deliverables:

- Final report, **15–20 pages** excluding cover, references, and appendices.
- Deployed POC/MVP using Streamlit Cloud, FastAPI, Gradio, or similar, accessible via live URL.
- At least one deployed model from the project with clear usage instructions and edge-case handling.
- Presentation slides submitted and locked by **4 Nov 2026, 23:59**.
- Final ZIP due **8 Nov 2026, 23:59** containing final report PDF, identical locked slides, and GitHub repository link.
- Live presentation on **5 or 12 Nov 2026**, depending on assigned class slot.
- Peer evaluation due **15 Nov 2026, 23:59**.

## Course requirements that affect everyone

- Use the **Olist Brazilian E-Commerce dataset** as the primary required data source.
- Join and document the relevant tables for the chosen problem(s).
- For customer-level analysis, verify whether `customer_id` or `customer_unique_id` is appropriate before feature engineering.
- **Prevent target leakage.** Do not use information that would only be known after the prediction point.
- Handle **class imbalance** with suitable metrics/techniques; do not rely on accuracy alone when the target is imbalanced.
- Prefer **scikit-learn Pipelines** where possible and keep preprocessing inside cross-validation folds when needed to prevent leakage.
- Start with the simplest model variant in each model family, then justify added complexity with evidence.
- Keep analysis reproducible with documented random seeds, transformations, assumptions, and data checks.
- Required tools include Python, a dashboard tool, a deployment platform, GitHub, and documented notebooks.
- Deployment must have a live URL, clear usage instructions, error handling, no hard-coded credentials, and must be tested before submission.
- Cite sources appropriately. Original analysis and interpretation are required. Collaboration within the team is encouraged; collaboration between teams is discouraged.
- AI tools must be declared and appropriately credited.
- Reports are submitted in **PDF, 12pt font, single-spaced**, with figure captions/table labels and consistent professional formatting.
- Late policy: 0–24h = 10% penalty; 24–48h = 20%; more than 48h = 50%; more than 7 days = zero marks.

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
