# Contributing

This is a data analytics project, so keep the workflow simple.

## First-time setup

1. Open your IDE, for example VS Code.
2. Clone this repository:

```bash
git clone https://github.com/Voycepeh/Team01_IT5006_Ecommerce_Analytics_AY2627Sem1.git
```

3. Open the cloned repository in your IDE.
4. Download the lecturer-provided Olist ZIP from the IT5006 Canvas Week 1 module.
5. Extract the CSV files into:

```text
data/raw/
```

The project should look roughly like this:

![Example local repository setup](docs/images/local-repo-example.png)

The CSV files are for local use only and are ignored by Git.

## Doing your analysis

Use the existing project folders:

| What you are working on | Put it in |
| --- | --- |
| Lecturer-provided CSVs and other local datasets | `data/` |
| EDA, modelling and analysis notebooks | `notebooks/` |
| Reusable Python scripts | `src/` |
| Streamlit, FastAPI, Gradio or Power BI deployment files | `deployment/` |
| Reports, slides, references and other project documents | `docs/` |

For most analysis work, you will mainly work inside `notebooks/` and read the local files from `data/raw/`.

## What to upload to GitHub

Upload your project work such as notebooks, Python scripts, reports, documentation and deployment files.

**Do not upload or commit any project data files.**

Everything under `data/` stays on your own computer. This includes raw, cleaned, processed, sampled and model-ready datasets. The repository `.gitignore` is already configured to ignore these files.

Only README instruction files inside `data/` should be tracked.

Before committing, check that you are only uploading your work and not the dataset.

## Working with the team

If you are making changes, use your own branch where practical and submit your work back to the repository for the team to review. Do not overwrite or delete another teammate's work without agreement.

Keep changes focused on the task you are working on.

Never commit passwords, tokens, API keys or other secrets.

## AI-assisted work

AI tools may assist with the project where consistent with the course requirements. Human teammates remain responsible for reviewing, understanding and validating AI-assisted work before submission.

AI agents must follow [`AGENTS.md`](AGENTS.md).

## Project requirements

Before making scope or deliverable decisions, check [`docs/project-requirements.md`](docs/project-requirements.md).

Official course page:
https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables

If local documentation differs from the latest official course page or Canvas announcement, follow the latest official course source.
