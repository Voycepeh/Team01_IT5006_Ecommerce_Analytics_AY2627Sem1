# Contributing

This repository follows the folder structure required for the IT5006 project. Keep contributions simple, easy to review, and in the correct folder.

## First-time local setup

If you are newer to Git or coding, follow these steps once before starting project work.

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Voycepeh/Team01_IT5006_Ecommerce_Analytics_AY2627Sem1.git
cd Team01_IT5006_Ecommerce_Analytics_AY2627Sem1
```

You now have a local copy of the project on your computer.

### 2. Add the lecturer-provided dataset locally

Download the official Olist ZIP from the IT5006 Canvas Week 1 module.

Extract the dataset files into:

```text
data/raw/
```

Example:

```text
data/
└── raw/
    ├── olist_customers_dataset.csv
    ├── olist_orders_dataset.csv
    ├── olist_order_items_dataset.csv
    └── ...
```

The data files are intentionally ignored by Git and must stay on your own computer.

### 3. Check that Git is not tracking the data

Run:

```bash
git status
```

The CSV, SQLite and other dataset files under `data/` should not appear as files waiting to be committed.

If they do appear, stop and ask a teammate before committing.

### 4. Install the project environment

From the repository folder, run:

```bash
pip install -r requirements.txt
```

The package list will be updated as the project develops.

## Starting a piece of work

Before changing files, update your local `main` branch:

```bash
git switch main
git pull
```

Then create your own branch. Use a short name that describes your task:

```bash
git switch -c your-name/task-description
```

For example:

```bash
git switch -c alice/eda-orders
```

Do your work in the correct project folder.

## Saving and sharing your work

### 1. Check what changed

```bash
git status
```

Make sure no project data files are listed for commit.

### 2. Stage the files you want to commit

Prefer adding the specific files you changed instead of blindly adding everything.

```bash
git add notebooks/your_notebook.ipynb
```

or:

```bash
git add src/your_script.py
```

### 3. Commit your changes

```bash
git commit -m "Add order EDA"
```

Use a short message that explains what you changed.

### 4. Push your branch to GitHub

```bash
git push -u origin your-name/task-description
```

### 5. Open a pull request

Open the repository on GitHub. GitHub should show your recently pushed branch and offer a **Compare & pull request** button.

Open a pull request into `main`, briefly explain what changed, and ask a teammate to review it before merging.

## Basic workflow

1. Pull the latest `main` before starting work.
2. Create a branch for your task.
3. Put files in the correct lecturer-required folder.
4. Commit only the files related to your task.
5. Push your branch and open a pull request.
6. Review your own changes before asking teammates to review.
7. Merge into `main` only after the work is ready.

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
