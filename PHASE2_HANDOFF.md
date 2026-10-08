# Phase 2 Classification — Session Handoff

**Read this first when continuing the Phase 2 classification work in a new Claude Code session or on another device.** It records where the work stands, the decisions already made with the user, and what to do next. Update the "Last updated" line and the status sections at the end of every working session.

* **Last updated:** 8 October 2026
* **Deadline:** Phase 2 report due **Sun 11 October 2026, 23:59** (see `docs/project-requirements.md`).
* **Scope:** the **classification** problem only. The other half of the team owns the **regression** problem, so do not touch it.
* **To resume, the user can say:** *"Read PHASE2_HANDOFF.md and continue from the next steps."*

---

## 1. Where things stand

| Item | Status |
|---|---|
| Branch | `feature/delivered-order-random-forest`, pushed to GitHub. **Not merged** into `main`. |
| Commits | `89bd62b` (Random Forest family, tests, `.pkl`, `PHASE2_TREE_MODEL.md`), `386eeab` (comparison notebook, Word report draft), `722c956` (this handoff file, citation fix), then the **two-stage design** changes of 8 Oct (if committed). Run `git log --oneline` for the latest. |
| Pull request | Create/open it from https://github.com/Voycepeh/Team01_IT5006_Ecommerce_Analytics_AY2627Sem1/pull/new/feature/delivered-order-random-forest. A draft description was given in an earlier session; it follows `.github/pull_request_template.md`. |
| Main notebook | `notebooks/phase2_delivered_order_classification_combined.ipynb`: **two-stage design**, 15 steps, run end to end (about 15 minutes, no errors), and all findings text written from its outputs |
| Report draft | `docs/Phase2_Classification_Report_Sections.docx` (AI-assisted, needs team review); updated for the two-stage design |
| Tree-model write-up | `PHASE2_TREE_MODEL.md` (file-by-file changes, results of the earlier random-split notebook) |
| Phase 1 report PDF | `Team1_Phase1_IT5006_AY2627Sem1.pdf` sits in the user's local folder and is **deliberately not committed**: the repo is **public** and the PDF lists teammates' names and NUS matriculation numbers. Key facts from it are copied in section 6 below. |

## 2. Files and what they are

| File | What it is | Edit? |
|---|---|---|
| `notebooks/phase2_delivered_order_classification_combined.ipynb` | **Main comparison notebook**, written as plain-English teaching steps | Yes, this is where the work continues |
| `docs/Phase2_Classification_Report_Sections.docx` | Report write-up moved out of the notebook: deliverable map, results table, sections 1, 2, 3, 7, 11, 12, 13 | Yes, in Word, or by editing the docx XML (the Node script that generated it was session-only and is gone) |
| `notebooks/phase2_delivered_order_random_forest_classification.ipynb` | Earlier Random Forest notebook (random split, F1); made the RF `.pkl` | Keep as a record |
| `notebooks/phase2_delivered_order_negative_review_classification.ipynb` | **Voyce's** Logistic Regression notebook | **Do not modify** (a teammate's work) |
| `src/classification_tree_model.py`, `tests/test_classification_tree_model.py` | Decision Tree / Random Forest pipeline builders and their tests | Yes, keep the tests passing |
| `src/classification_model.py` | Shared save/predict helpers. Has an optional `compress` argument (default unchanged) | Only additive changes |
| `deployment/*.pkl` | Saved models from the **earlier random-split** notebooks. **Out of date** for the current design | Re-save only after the final model is agreed |

## 3. Decisions already agreed with the user (do not re-litigate)

1. **Problem: two-stage design (agreed 8 Oct 2026).** Predict which customers will leave a negative review (1–2 stars) early enough to contact them before they write it. **Stage 1**, at the promised delivery date: a rule contacts every order that has not arrived (late). **Stage 2**, at delivery: the models rank every customer who has **not reviewed yet**, and the riskiest 10% are contacted. Orders reviewed before delivery (4,653) are left out of Stage 2, because the review already exists at its prediction point and their delivery inputs come from after the review. Present it as **one problem with two decision points** (brief: 1–2 problems max).
2. **Models:** **R** = rule benchmark (rank by lateness: `-days_early`; flag = `late_delivery_flag`); **A** = plain Logistic Regression (`C=np.inf`, no class weighting); **B** = tuned Logistic Regression; **C** = single Decision Tree; **D** = Random Forest, no tuning (200 trees); **E** = tuned Random Forest. **The final comparison is B vs E.**
3. **Validation is time-based:** sort by purchase date; train on the earliest 80% of Stage 2 orders (up to 2018-05-31) and test on the latest 20%; `TimeSeriesSplit(n_splits=5)` for CV and tuning. The random stratified split appears only as a comparison (step 12).
4. **The deciding metric is top-10% precision** (Phase 1's "precision among top-ranked orders"). It is used for `GridSearchCV(refit=...)` and for model selection. F1, PR-AUC, AUC-ROC and balanced accuracy are still reported.
5. **"Worth it" rule:** a more complex option must have a higher average top-10% precision **and** win in at least 4 of the 5 time-ordered folds.
6. **No regression metrics** (MAE, RMSE, R²) in the classification work. The user had them removed: they belong to regression. Section 7 of the report states "not applicable".
7. **Report write-up lives in the Word doc, not the notebook.** The notebook keeps the step-by-step plain-English explanations ("What we are doing", "Why", "What this tells us", "Likely question") **for now**. The user will say when to remove them.
8. **No ensemble/stacking.** The Random Forest is itself a bagging ensemble.
9. **Seeds:** `random_state=42` everywhere. The notebook **does not write `.pkl` files**.

## 4. Current results (two-stage design, time-based, test = later orders)

**Stage 2 models** (91,171 orders; train 72,936 = 3 Oct 2016–31 May 2018; test 18,235 = 31 May–29 Aug 2018, 8.5% negative):

| Model | CV top-10% precision | Test top-10% precision | Unhappy customers in the 1,824-order top-10% list | Test AUC-ROC |
|---|---|---|---|---|
| R. Rule | 0.145 | 0.084 | 153 | 0.484 |
| A. Plain LR | 0.239 | 0.211 | 385 | 0.642 |
| B. Tuned LR (`C=0.001`, `class_weight="balanced"`) | 0.267 | **0.229** | **418** | **0.647** |
| C. Single tree | 0.145 | 0.140 | 256 | 0.537 |
| D. RF, no tuning | 0.251 | 0.228 | 416 | 0.638 |
| **E. Tuned RF** (`max_depth=12`, `min_samples_leaf=25`, no class weighting) | **0.275** | 0.226 | 413 | 0.635 |

* **Selection: E by the pre-set CV rule (wins all 5 folds vs B), but B, D and E are tied on later orders** (5 customers apart; B marginally ahead). **Team decision needed:** keep E or prefer the simpler B (notebook step 13).
* **The rule is no better than random in Stage 2** (8.4% vs 8.5%). Models are about 2.7× random.
* **Both families rely on order size first** (`item_count`, `seller_count`), then `delivery_days`. Lateness barely matters in Stage 2 (Stage 1 handles it).
* **Error analysis:** the Stage 2 list is essentially multi-item orders (90% of 2+-item orders listed; their negative rate 22.7% ≈ list precision 22.6%). 1,107 of the 1,130 missed negatives are single-item orders.
* **Stage 1** (whole data): 6,381 late orders, 62.4% negative, 32.5% of all negatives; 44 reviewed before the promised day ended. **Test period:** 657 late orders, 46.9% precision, 17.1% of negatives.
* **Whole system on the test period (step 14):** 2,355 contacts reach 693 of 1,806 negatives (38.4%), 29.4% precision, about 26 contacts a day.
* **Random split** flatters only a little now (0.244–0.262 vs 0.211–0.229) but would put E ahead of B, the reverse of later orders.
* **Leakage sensitivity (side check, not in the notebook):** with the pre-delivery reviewers kept, test top-10% precision was 0.333 (E) / 0.327 (B); 262 of E's 638 hits had already reviewed before delivery.

## 5. Next steps (in priority order)

*Done 8 Oct 2026:* leakage fix via the two-stage design (old item 2); ROC and precision-recall curves (notebook step 8.1); 📌 quotes from the IT5006 brief and the T08 tutorial under each method choice (journal papers are paraphrased, never quoted, because their wording could not be checked).

1. **Final Stage 2 model: team decision** (step 13): keep E (pre-set CV rule) or choose B (tied on test, simpler). State the tie in the report either way.
2. **Feature-by-feature rationale with citations** (professor's main feedback point). Add a "why each input" table to the notebook and to section 3 of the Word doc, using section 6 below. Note the new finding: order size matters most in Stage 2.
3. **Calibration check** (the Phase 1 review recommends reporting calibration with discrimination): a reliability curve for B and E. This is a classification check, not a regression metric.
4. **Smaller alignments, not yet approved by the user:** rename the stakeholder to "customer service" (Phase 1 wording); restrict to Phase 1's Jan 2017–Aug 2018 window (Stage 2 includes 259 orders from 2016); justify one-hot vs target encoding (Micci-Barreca 2001; Pargent et al. 2022); drop `purchase_month` (adds nothing).
5. **Citation check:** the temporal-leakage point is currently credited to "our Phase 1 literature review". Kapoor & Narayanan (2023) is likely the right source (their leakage taxonomy includes temporal leakage), so **verify that before citing it**.
6. After the decisions: **retrain and re-save the final Stage 2 model** to `deployment/` for Phase 3, and update `PHASE2_TREE_MODEL.md` (still describes the earlier random-split work).
7. Optional: a `RUN_TUNING` switch to skip grid searches on re-runs (the notebook now takes about 15 minutes).
8. **Team sign-off needed** on the two-stage framing, the success criteria and the "worth it" rule, and **declare AI assistance** in the report.

## 6. Phase 1 report: facts and references to reuse

**EDA findings relevant to classification**

* Usable window: Jan 2017–Aug 2018 (99.6% of orders). 2016 and the final months are sparse.
* Delivery duration correlates −0.33 with review score, and days early +0.27. Monetary variables are only −0.03 to −0.12.
* **"Cliff at zero":** on the promised day 4.03 stars; 1 day late 3.29; 4–7 days late 2.10. Arriving early saturates (7–14 days early 4.30, 15+ days early 4.32).
* Within the six largest states, the on-time vs late gap is 1.63–2.33 stars, so geography acts *through* delivery time.
* **Two-thirds of bad reviews are on on-time orders** (Fig. 6), so fixing delivery alone addresses under a third.
* A single order may involve several sellers and **arrive in separate shipments**. 64% of orders cross a state boundary.
* Late orders: 19.3 of the 20.6 extra days are carrier transit. Quotes are padded: the median is quoted 24 days and arrives in 12, and 91.9% arrive early.
* Repeat purchasing is negligible (3.12%), so order-level modelling is fine and no `customer_unique_id` grouping is needed.
* **Candidate 2 (this problem):** stakeholder = customer service, which works a fixed daily contact list; target = review_score ≤ 2 on 95,824 delivered, reviewed orders; base rate 12.81%; *"judge the model on precision among its top-ranked orders"*.

**Literature review references (what each supports)**

1. Cui, Sun, Lu & Golden (2023), *Sooner or later? Promising delivery speed in online retail*, MSOM, https://doi.org/10.1287/msom.2021.0174. Late vs early delivery promises have measurable sales and profit effects.
2. Deshpande & Pendem (2022/2023), *Logistics performance, ratings, and its impact on customer purchasing behavior and sales*, MSOM 25(3), https://doi.org/10.1287/msom.2021.1045. Longer delivery leads to lower ratings (15M orders). Supports the timing features.
3. Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 35, https://doi.org/10.48550/arXiv.2207.08815. Supports choosing tree models.
4. Kandula, Krishnamoorthy & Roy (2021), *A prescriptive analytics framework for efficient e-commerce order delivery*, Decision Support Systems 147, https://doi.org/10.1016/j.dss.2021.113584. Expected delivery date minus order date as a predictor; cost-sensitive learning works best.
5. Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in ML-based science*, Patterns 4(9), https://doi.org/10.1016/j.patter.2023.100804. Leakage inflates complex models over simple ones (AUC gap 0.14 → 0.01); use LR as a baseline control.
6. Micci-Barreca (2001), *A preprocessing scheme for high-cardinality categorical attributes*, SIGKDD Explorations 3(1), https://doi.org/10.1145/507533.507538. Target encoding.
7. Pargent, Pfisterer, Thomas & Bischl (2022), *Regularized target encoding outperforms traditional methods…*, Computational Statistics 37(5), https://doi.org/10.1007/s00180-022-01207-6. Regularised target encoding and 5-fold CV. **Not** a source for temporal leakage.
8. van den Goorbergh, van Smeden, Timmerman & Van Calster (2022), *The harm of class imbalance corrections for risk prediction models*, JAMIA 29(9), https://doi.org/10.1093/jamia/ocac093. Resampling or rebalancing hurts calibration without improving discrimination; prefer threshold tuning; report calibration.
9. Zhang et al. (2023), *A brief survey of ML and DL techniques for e-commerce research*, JTAER 18(4), https://doi.org/10.3390/jtaer18040110. Background.

**Phase 1 conclusion's recommendations:** benchmark against the platform's own delivery estimates *(done: rule R)*; report calibration alongside discrimination *(to do)*; tune the decision threshold rather than resampling *(top-k used)*; use validation that respects time order *(done)* and seller dependence *(not done)*; keep a simple control model *(done: A)*.

## 7. Environment notes (Windows laptop)

* **scikit-learn:** on the first laptop, Windows Smart App Control blocked the 1.9.1 binaries, so `pip install scikit-learn==1.9.0` was used there (same 1.9 line as the team's `.pkl`). On the second device (`D:\IT5006 Projet`, project `.venv` with Python 3.13) 1.9.1 works. 1.7.x fails with pandas 3 string columns.
* **Re-run the notebook:** `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/phase2_delivered_order_classification_combined.ipynb` (needs `pip install nbconvert`, a tool only, not in `requirements.txt`). It takes about 15 minutes. **Keep the laptop awake**: sleep stalls the run.
* **Tests:** `python -m pytest -q` (17 pass, including the dashboard tests).
* **Dates** in `order_level.csv` have mixed formats, so use `pd.to_datetime(..., format="mixed")`.
* **Documents:** no pandoc or LibreOffice. Microsoft Word is installed (its COM automation can export docx to PDF) and PyMuPDF is installed (for reading PDFs and rendering pages).
* The `gh` CLI is not installed, so PRs are opened in the browser.

## 8. How to work with this user

* **Explain simply**, as if teaching someone new to machine learning, and give reasons the user can defend.
* **Never invent numbers.** Every number in notebook or doc text must come from executed outputs. Re-check findings after every re-run.
* **Before committing:** list the changed files and wait for the go-ahead. Run the simplify pass on code before committing. Never commit the Phase 1 PDF (public repo, personal data).
* **Ask before changing the method** (metrics, validation, models); offer a recommendation with the options.
* Do not modify teammates' notebooks or `.pkl` files. Follow `AGENTS.md`.
