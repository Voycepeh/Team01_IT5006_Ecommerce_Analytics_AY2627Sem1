# Contributing

This repository follows the folder structure required for the IT5006 project. Keep contributions simple, easy to review, and in the correct folder.

## Basic workflow

1. Create a branch for your work.
2. Put files in the correct lecturer-required folder.
3. Commit only the files related to your task.
4. Open a pull request before merging to `main`.
5. Review your own changes before asking teammates to review.

## Data files

**Do not commit project data files to GitHub.**

All files under `data/` are intended to stay local and are covered by `.gitignore`. This includes raw, cleaned, processed, sampled and model-ready datasets.

For the lecturer-provided Olist dataset, download the official ZIP from the IT5006 Canvas Week 1 module and extract the files locally into:

```text
data/raw/
```

Only README instruction files inside `data/` should be committed. Do not force-add ignored data files with `git add -f`.

## Where to put files

| File / asset | Folder |
| --- | --- |
| Local project data | `data/` (do not commit) |
| EDA, modelling and experiment notebooks (`.ipynb`) | `notebooks/` |
| Reusable Python scripts (`.py`) | `src/` |
| Streamlit, FastAPI or Gradio app files | `deployment/` |
| Power BI (`.pbix`) or other deployable dashboard files | `deployment/` |
| Reports (`.pdf`, `.docx`) | `docs/` |
| Presentation slides (`.pptx`, `.pdf`) | `docs/` |
| Project notes, references and requirement documents | `docs/` |

Do not add new top-level project folders unless the team agrees and the lecturer's required structure is still preserved.

## Team rules

1. Do not commit any project data files to GitHub.
2. Do not overwrite or delete a teammate's work without agreement.
3. Do not rename or remove the lecturer-required top-level folders without team agreement.
4. Never commit passwords, tokens, API keys or other secrets.
5. Do not fabricate data, model results, tests, citations or screenshots.
6. Review and understand AI-assisted work before merging or submitting it.
7. Keep changes focused and avoid unrelated edits in the same pull request.
8. Only state that something was tested or validated if it was actually checked.

## AI-assisted contributions

AI tools and coding agents are allowed to assist where consistent with the course requirements. Human teammates remain responsible for reviewing, understanding and validating the work.

AI agents must follow [`AGENTS.md`](AGENTS.md).

## Project requirements

Before making scope or deliverable decisions, check [`docs/project-requirements.md`](docs/project-requirements.md).

Official course page:
https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables

If local documentation differs from the latest official course page or Canvas announcement, follow the latest official course source.
