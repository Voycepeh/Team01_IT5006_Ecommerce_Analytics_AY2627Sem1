# IT5006 Project Requirements

> **Readable Markdown mirror of the official IT5006 project specification.**
>
> **Official live source:**  
> https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Aug_V2.html#project-timeline-deliverables
>
> **Archived PDF copy in this repository:**  
> [Project Requirements.pdf](Project%20Requirements.pdf)
>
> This Markdown file exists so teammates and AI agents can read the project requirements reliably. If this file, the archived PDF, or any repository note conflicts with the latest official course page or Canvas announcement, follow the latest official course source.

## Module and Project Overview

- **Module:** IT5006 — Fundamentals of Data Analytics
- **Academic period:** AY 2026/27 Semester 1
- **Project:** E-Commerce Analytics
- **Dataset:** Olist Brazilian E-Commerce Public Dataset
- **Team size:** 3 minimum, 5 maximum; no individual projects
- **Class:** Thursdays, 6:30–9:30 PM
- **Project weighting:** Phase 1 = 20%, Phase 2 = 40%, Phase 3 = 40%

The project requires each team to explore the Olist Brazilian E-Commerce dataset, define one or two analytics problems relevant to an e-commerce stakeholder, build and evaluate predictive models, and deploy a proof-of-concept application. The project culminates in a professional report and live presentation.

## Learning Objectives

By completing the project, teams should be able to:

- Apply an end-to-end data analytics methodology to real-world e-commerce data.
- Scope and justify analytics problems that fit the available data and team capacity.
- Implement both **classification and regression** using a small, reusable set of model families.
- Follow a disciplined train-test paradigm with proper model selection and evaluation.
- Recognise and avoid **data leakage** and **class imbalance** pitfalls.
- Develop interactive dashboards for stakeholder communication.
- Deploy machine learning models as proof-of-concept applications.
- Generate actionable insights for an identified e-commerce stakeholder.
- Practise professional reporting and presentation skills.

## Team Formation and Registration

- **Team size:** 3 minimum, 5 maximum.
- **Registration deadline:** **Sunday, 23 August 2026 at 23:59**.
- **Submission:** Register the team on Canvas.
- **Late registration:** Not permitted and may affect the final grade.

Recommended team capabilities include data analysis, coding, documentation, presentation, leadership, and deployment/DevOps.

## Dataset

### Olist Brazilian E-Commerce Dataset

- **Domain:** E-commerce / online marketplace in Brazil.
- **Period:** Approximately 2016–2018.
- **Format:** CSV files across multiple related tables.
- **Required source:** Use the course-provided copy from Canvas → Week 1 module rather than downloading a different copy from Kaggle or elsewhere.

### Dataset Tables

| Table | Description |
| --- | --- |
| `olist_orders_dataset.csv` | Order information, status, purchase/delivery timestamps |
| `olist_order_items_dataset.csv` | Items per order, product, seller, price, freight |
| `olist_order_payments_dataset.csv` | Payment method, instalments, value |
| `olist_order_reviews_dataset.csv` | Customer review score and comments |
| `olist_customers_dataset.csv` | Customer location information |
| `olist_sellers_dataset.csv` | Seller location information |
| `olist_products_dataset.csv` | Product category, dimensions, weight |
| `olist_geolocation_dataset.csv` | Zip-code geolocation reference |
| `product_category_name_translation.csv` | Portuguese-to-English category names |

Teams must join and merge relevant tables for their chosen problem(s) and clearly document the resulting data model.

### Customer Identifier Warning

`customer_id` and `customer_unique_id` are not interchangeable.

- `customer_id` is generated for each order and behaves like an order-customer key, not a persistent person identifier.
- `customer_unique_id` is the identifier for the same customer across multiple orders.

For repeat-purchase behaviour, customer-level aggregates/features, or any analysis requiring the same person across orders, use `customer_unique_id` and document the choice.

## Data Leakage Warning

Do not use information that would only be known after the target event has already occurred.

For example, if predicting late delivery or delivery lead time, post-outcome fields such as actual delivered carrier/customer dates must not be used as predictors. Likewise, review information cannot be used to predict a review score because that information only exists after the customer experience.

Before modelling, list each candidate feature and confirm that it is known at the intended prediction point.

## Class Imbalance

Many Olist targets may be heavily imbalanced, such as late orders or low review scores. Accuracy alone can be misleading.

Use appropriate metrics such as:

- Precision
- Recall
- F1-score
- PR-AUC
- Balanced accuracy

Consider class weights or resampling where appropriate, and discuss imbalance explicitly in the report.

## Example Analytics Problem Types

Example areas include:

1. **Operational performance** — delivery delays, fulfilment bottlenecks, order-status outcomes.
2. **Customer experience** — review score, satisfaction level, complaint risk.
3. **Commercial outcomes** — order value, payment behaviour, product demand.

These are examples only. The team must define, justify, and scope its own problem(s).

### Example Problem Ideas

#### Classification: Late Delivery Prediction

- Question: Will an order be delivered later than the estimated delivery date?
- Target: Late vs on-time.
- Stakeholder: Operations / logistics.

#### Regression: Delivery Lead Time Prediction

- Question: How many days from purchase to customer delivery?
- Target: Delivery lead time in days.
- Stakeholder: Customer service / fulfilment planning.

#### Classification: Low Customer Satisfaction Risk

- Question: Will an order receive a low review score such as 1–2 stars?
- Target: Low satisfaction vs satisfactory.
- Stakeholder: Customer experience / seller quality.

#### Dual Framing

A single business problem may be framed as both classification and regression, for example predicting delivery lead time and separately classifying the same order as late/on-time against the estimated delivery date.

## Problem Scoping Checklist

Before finalising Phase 2 problem(s), confirm:

1. **Derivable target:** The target can be computed from the provided tables.
2. **Realistic features:** Predictors are known before the outcome occurs.
3. **Sufficient signal:** Phase 1 EDA shows at least some relationship between candidate features and the target.
4. **Manageable imbalance:** If the target is imbalanced, there is an appropriate evaluation/handling plan.
5. **Clear stakeholder:** A real business role can be identified that would use the prediction and act on it.

## Project Timeline and Deliverables

### Overall Calendar

- **Teaching period:** Weeks 1–13, **10 August–13 November 2026**.
- **Recess:** **19–27 September 2026**.
- **Class:** Thursdays, 6:30–9:30 PM.
- **Reading week:** **14–20 November 2026**.
- **Examinations:** **21 November–5 December 2026**.

For any date changes, check Canvas and the official live project page.

### Timeline Summary

| Milestone | Task | Deliverable | Due date | Weight |
| --- | --- | --- | --- | ---: |
| 0 | Team formation | Canvas signup | **Sun, 23 Aug 2026** | — |
| 1 | Literature Survey & EDA | Report + dashboard | **Sun, 13 Sep 2026, 23:59** | **20%** |
| — | Recess | Team problem scoping | **19–27 Sep 2026** | — |
| 2 | Problem Definition & Modelling | Technical report + code | **Sun, 11 Oct 2026, 23:59** | **40%** |
| 3 | Presentation slides locked | Slide deck (PDF/PPT) | **Wed, 4 Nov 2026, 23:59** | Part of Phase 3 |
| 3 | Deployment & Final Report | Report + deployment + ZIP | **Sun, 8 Nov 2026, 23:59** | **40% combined Phase 3** |
| 3 | Live presentations | Team presentation in class | **Thu, 5 Nov or Thu, 12 Nov 2026** | Part of Phase 3 |
| 3 | Peer evaluation | Individual assessment | **Sun, 15 Nov 2026, 23:59** | Part of Phase 3 |

Phase 3 components are graded together as one combined 40%; peer-evaluation results may be used to make individual grade adjustments within that 40%.

## Phase 1 — Foundation: Literature Review & Exploratory Data Analysis

- **Deadline:** Sunday, 13 September 2026 at 23:59.
- **Weight:** 20%.
- **Report length:** 4–5 pages excluding cover page, references, and appendices.

### Literature Review

Approximately 2 pages covering:

- Relevant e-commerce analytics approaches.
- Review of modelling techniques relevant to the problem area, including classification and regression.
- Feature-engineering techniques for transactional data.
- Evaluation metrics and their suitability, including for imbalanced targets.
- Key trends, gaps, and opportunities.

### Exploratory Data Analysis

Approximately 2–3 pages covering:

- Dataset overview including tables used, joins, row counts, and missing values.
- Temporal patterns such as order volume and seasonality.
- Customer / seller / product distribution analysis.
- Correlation and relationship exploration.
- **Candidate problem exploration:** preliminary investigation of 2–3 possible analytics problems to inform Phase 2 scoping.
- Key insights and patterns discovered.

### Interactive Dashboard

- Build using Streamlit (recommended), Tableau Public, or Power BI.
- Include interactive e-commerce views such as orders, delivery, reviews, categories, geography, etc.
- Submit via a live link and include the link in the report.

### Phase 1 Submission

- Combined PDF report containing Literature Review + EDA, dashboard link, and GitHub repository link.
- GitHub repository containing all raw code/notebooks.

## Phase 2 — Analytics Implementation: Problem Definition, Model Building & Evaluation

- **Deadline:** Sunday, 11 October 2026 at 23:59.
- **Weight:** 40%.
- **Report length:** 6–8 pages excluding cover page, references, and appendices.

### Required Problem Scope

- Define **1 or 2 analytics problems maximum**.
- The project must cover **both classification and regression**.
- This may be two separate problems or one problem with dual framing.
- Scope must be achievable using Olist data and course techniques.
- Each problem requires business stakeholder context, target definition, and success criteria.

Use the recess period, 19–27 September, to discuss and finalise the problem scope as a team.

### Model Family Budget

Focus on a small number of model families rather than many shallow experiments.

- **2–3 model families total across the entire project.**
- Reuse the same family across classification and regression tasks where sensible.
- Start with the simplest variant in each family as a baseline.
- Any more complex variant must demonstrate improvement over the baseline.

Illustrative families include:

- Linear family: Logistic Regression / Linear Regression / Ridge.
- Tree-based family: Decision Tree / Random Forest classifier or regressor.
- Optional ensemble family: voting or stacking ensembles built from the chosen families.

### Modelling Requirements

- Use the simplest variant within a family as the starting baseline.
- Use disciplined train-test or train-validation-test methodology.
- Prevent data leakage.
- Compare model families systematically with suitable metrics.
- Use cross-validation for robust performance estimates and hyperparameter tuning.
- Set and document random seeds.
- Document preprocessing steps and pipelines.
- Do not implement many algorithms without depth; clear business interpretation is more important than a long list of poorly tuned models.

### Phase 2 Deliverables

- Problem statement(s) and stakeholder context.
- Data preprocessing and cleaning methodology.
- Feature-engineering strategy.
- Model descriptions and training process using 2–3 model families total.
- Hyperparameter tuning methodology.
- Classification metrics such as Precision, Recall, F1-score, ROC-AUC / PR-AUC as appropriate.
- Regression metrics such as MAE, RMSE, and R² as appropriate.
- Cross-validation results.
- Feature importance / interpretability analysis.
- Model comparison and final selection with justification.
- Ensemble or stacking results if used.
- Actionable insights for the e-commerce stakeholder.
- Discussion of limitations and constraints.

### Phase 2 Submission

- Technical report PDF containing the GitHub repository link.
- GitHub repository containing Jupyter notebooks and Python scripts.
- Model-performance summary tables in the report.

## Phase 3 — Integration & Communication: Deployment, Final Report & Presentation

- **Combined weight:** 40%.
- **Final report length:** 15–20 pages excluding cover, references, and appendices.

### Presentation Slides Deadline

**Wednesday, 4 November 2026 at 23:59** for all teams.

- Submit presentation slides in PDF or PPT via Canvas.
- This exact deck must be used for the live presentation and included in the final ZIP.
- No changes to the slides after the deadline.
- Late slide submission follows the same late-penalty policy as other deliverables.

### Final Report & ZIP Deadline

**Sunday, 8 November 2026 at 23:59**.

Submit a ZIP containing:

- Final report PDF including application link and GitHub link.
- Presentation slides identical to the deck locked on 4 November.
- GitHub repository link.

### Live Presentations

**Thursday, 5 November and Thursday, 12 November 2026** during class, depending on assigned presentation order.

- Presentation order will be communicated by instructors.
- **10 minutes presentation + 5 minutes Q&A**.
- Use the locked 4 November slide deck.

### Peer Evaluation

**Sunday, 15 November 2026 at 23:59**.

- Individual contribution assessment.
- May be used for individual grade adjustment.
- Submitted separately through a Canvas survey.

### Final Report Deliverables

The final report should include:

- Cover page.
- Executive summary.
- Dataset and preprocessing.
- Problem definition and stakeholder context.
- Exploratory analysis highlights.
- Modelling approaches covering classification and regression.
- Results and evaluation.
- Model deployment and application.
- Business recommendations and impact.
- References.

### Deployed Application / POC / MVP

- Use Streamlit Cloud (recommended), FastAPI, Gradio, or similar.
- Include the live URL in the final report.
- Demonstrate at least one deployed model from the project.
- Provide clear usage instructions.
- Include error handling for edge cases.
- The Olist dataset covers historical 2016–2018 orders; the app may predict on held-out historical records simulating new orders and does not need a live order feed.

### GitHub Repository Requirements

- Well-documented repository with all project code.
- README with setup instructions and project overview.
- Organised folder structure:

```text
project-root/
├── data/
├── notebooks/
├── src/
├── deployment/
├── docs/
└── README.md
```

- Include `requirements.txt` or `environment.yml`.
- Include the GitHub repository link in all report submissions.

### Presentation Slides

- Professional slides in PowerPoint or PDF.
- Include screenshots of the deployed application.
- Submit with the final report in the ZIP.

## Assessment Criteria

### Phase 1 — 20%

Assessment considers:

- Literature-review quality and relevance to e-commerce analytics.
- EDA depth and thoroughness.
- Candidate problem exploration.
- Dashboard functionality and clarity.
- Report clarity.

### Phase 2 — 40%

Assessment considers:

- Problem framing and scoping that is appropriate, justified, and achievable.
- Data preparation and feature engineering, including leakage avoidance.
- Modelling discipline, train-test methodology, no leakage, and reproducibility.
- Quality of model-family comparison rather than breadth of techniques.
- Evaluation rigor using suitable classification, regression, and imbalance metrics.
- Ensemble / stacking use where justified and well explained.
- Business interpretation for the e-commerce stakeholder.

### Phase 3 — 40%

Assessment considers:

- Report integration and coherence.
- Model deployment / POC / MVP.
- Presentation quality.
- Business impact and recommendations.
- Technical excellence.

## Technical Requirements

### Required Tools

- **Python:** pandas, scikit-learn, matplotlib/seaborn, numpy.
- **Dashboard:** Streamlit (recommended), Tableau Public, or Power BI.
- **Deployment:** Streamlit Cloud, FastAPI, Gradio, or similar.
- **Version control:** GitHub repository (mandatory).
- **Documentation:** Jupyter Notebooks with Markdown explanations.

### Modelling Guidelines

- Use scikit-learn Pipelines where possible to prevent leakage.
- Begin with the simplest variant in each model family and add complexity only when justified by evidence.
- Limit the whole project to 2–3 model families total.
- Use stratified splits for imbalanced classification targets.
- Set and document random seeds.
- Use ensemble methods such as voting or stacking only when they add demonstrable value and are built from the chosen families.
- Watch for target leakage from post-outcome columns.
- Watch for class imbalance and do not rely on accuracy alone.

### Coding Guidelines

- Write clean, well-commented code with meaningful variable names.
- Make analysis reproducible with random seeds set.
- Include error handling and data-validation checks.
- Keep clear separation between data preparation, modelling, evaluation, and deployment.
- Store all code in GitHub with descriptive commit messages.
- Maintain a comprehensive README with setup and execution instructions.

### Data Usage Guidelines

- Use the Olist dataset as the primary and only required data source.
- External data is not required; if used, justify and document it.
- Handle missing values and outliers appropriately.
- Document all data transformations and assumptions.
- Ensure reproducibility of results.

### Deployment Guidelines

- Application must be accessible via a live URL.
- Include error handling for edge cases.
- Provide clear usage instructions.
- Document any API endpoints with examples.
- Do not hard-code credentials.
- Test deployment before submission.

## Support Resources

- Consultation available on request, recommended approximately once every two weeks.
- Canvas Discussion Forum for technical questions and peer assistance.
- Course instructors available via email.
- Thursday class sessions may be used for progress check-ins and Q&A.

## Important Policies

### Academic Integrity

- Cite all sources and reference materials appropriately.
- Original analysis and interpretation are required.
- Collaboration within teams is encouraged; collaboration between teams is discouraged.
- AI tools must be declared and appropriately credited.

### Late Submission Policy

| Delay | Penalty |
| --- | ---: |
| 0–24 hours | 10% |
| 24–48 hours | 20% |
| More than 48 hours | 50% |
| More than 7 days | Zero marks |

### Formatting Requirements

- Reports must be PDF.
- 12pt font.
- Single-spaced.
- Figure captions and table labels required.
- Professional, consistent formatting.

### File Naming Convention

The official specification provides conventions for the phase report and final ZIP naming. Follow the latest course page / Canvas instructions if the naming convention is updated.

### GitHub Repository Naming

The official specification gives the naming pattern:

`TeamX_IT5006_Ecommerce_Analytics_AY2627Sem1`

## Useful Pointers

1. Start early and begin exploring Olist data immediately after team formation.
2. Use recess wisely to finalise 1–2 problem(s) and plan Phase 2.
3. Scope carefully; one well-executed problem is better than multiple rushed problems, but the project still needs both classification and regression coverage.
4. Fewer model families, better evaluation: 2–3 well-tuned, reused model families beat many shallow experiments.
5. Check for leakage first by identifying which columns are known before the prediction point.
6. Use pipelines to prevent leakage, especially by keeping preprocessing inside cross-validation folds.
7. For customer-level logic, use `customer_unique_id` rather than `customer_id` when identifying repeat customers.
8. Hold regular meetings and document decisions.
9. Use version control and commit frequently with descriptive messages.
10. Focus on business value and connect findings to an e-commerce stakeholder.
11. Test deployment early; a simple working POC is better than a complex broken app.
12. Plan for the demo with specific examples for the live presentation.

## References and Suggested Readings

The official project specification lists the Olist dataset and standard machine-learning references, including introductory statistical learning, machine learning with Python, random forests, stacking/generalisation, and ensemble methods.

For the exact reference list, refer to the official live project page or archived PDF.

## Maintenance Rule

When the lecturer or Canvas changes a deadline, scope rule, technical requirement, or submission format:

1. Update this file.
2. Update the root `README.md` if the change affects teammates directly.
3. Update `AGENTS.md` if the change affects how agents should work.
4. Treat the latest official course source as authoritative.
