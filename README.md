# IT5006 E-commerce Analytics — Team 01

Repository for Team 01's IT5006 Fundamentals of Data Analytics project using the Olist Brazilian E-Commerce dataset.

## Project overview

This project applies the data analytics workflow taught in IT5006 to an e-commerce dataset. The assessed work develops across the course phases, covering exploratory data analysis, problem definition, predictive modelling, dashboarding and a deployment proof of concept.

The project will include both classification and regression work and will keep the modelling scope focused in line with the course requirements.

## Interactive dashboard

Phase 1 Streamlit dashboard:

https://team01-it5006-ecommerce.streamlit.app/

Run locally:

```bash
streamlit run deployment/streamlit/app.py
```

The dashboard presents descriptive analysis of Olist orders, delivery performance, customer experience, products and geography.

## Repository structure

```text
project-root/
├── data/
├── notebooks/
├── src/
├── deployment/
├── docs/
└── README.md
```

- `data/` contains the local project data structure. Dataset files themselves are not committed to GitHub.
- `notebooks/` contains exploratory analysis, modelling and experiments.
- `src/` contains reusable Python code where needed.
- `deployment/` contains dashboard and deployment assets.
- `docs/` contains reports, presentation materials, references and project documentation.

## Project deliverables

| Phase | Main deliverables | Due |
| --- | --- | --- |
| Phase 1 | Literature Review, EDA and interactive dashboard | **13 Sep 2026, 23:59** |
| Phase 2 | Problem Definition and Modelling | **11 Oct 2026, 23:59** |
| Phase 3 | Final report, deployment and presentation | **Nov 2026** |

Presentation slides are locked on **4 Nov 2026, 23:59** and the final report with deployment ZIP is due **8 Nov 2026, 23:59**.

## Project requirements

The repository keeps a readable local copy of the project requirements for reference:

- [`docs/project-requirements.md`](docs/project-requirements.md)
- [`docs/Project Requirements.pdf`](docs/Project%20Requirements.pdf)
- Official course page: https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables

If there is any difference, the latest official course page or Canvas announcement is the final authority.

## Environment

Project dependencies are recorded in [`requirements.txt`](requirements.txt) and will be updated as the analysis develops.

## Team contribution guide

Team workflow and local setup instructions are kept separately in [`CONTRIBUTING.md`](CONTRIBUTING.md).
