# PAA-NSGA-II: Intelligent CRM Lead Assignment & SLA Prediction

**An Adaptive Multi-Criteria Intelligent Decision Framework for Lead Assignment and Predictive SLA Management in Customer Relationship Management Systems**

## Team
- Ashwanth S (24BCE2035)
- Ragunandhan T (24BCE0721)
- Anushanth L G (24BCE2094)
- Mohammed Asif A (24BCE2076)

**Guide:** Dr. Deepika J, School of Computer Science and Engineering, VIT Vellore

---

## Datasets

### 1. BPI Challenge 2014

Real operational data from Rabobank Nederland Group ICT (HP Service Manager).

| File | Records | What it provides |
|------|---------|-----------------|
| `Detail_Incident.csv` | 31,238 | Priority, impact, urgency, timestamps, handle time |
| `Detail_Incident_Activity.csv` | 210,837 | 166 assignment groups, assignments, reassignments |
| `Detail_Interaction.csv` | 68,358 | First call resolution, interaction handle time |

**Source:** [4TU.ResearchData — BPI Challenge 2014](https://data.4tu.nl/collections/BPI_Challenge_2014/5065469)

### 2. Home Credit Default Risk

Real consumer lending data — loan applications with credit bureau history and previous application records.

| File | Records | What it provides |
|------|---------|-----------------|
| `application_train.csv` | 307,511 | 122 features — income, credit amount, external scores, flags |
| `bureau.csv` | 1,716,428 | Previous credits from other institutions |
| `previous_application.csv` | 1,670,214 | Previous loan applications at Home Credit |

**Source:** [Kaggle — Home Credit Default Risk](https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk)

---

## Notebooks

### SLA Breach Prediction (`SLA_Breach_Prediction.ipynb`)

Predicts whether an IT incident will breach its SLA deadline using BPI 2014 data.

**Pipeline:**
1. Loads and merges 3 BPI 2014 files
2. Engineers 20 features from raw operational data
3. 70/30 stratified train-test split (11,683 / 5,008)
4. Trains 4 models: XGBoost, LightGBM, CatBoost, Random Forest
5. Feature combination experiment: 5 → 8 → 12 → 20 features
6. Full validation: 5-fold stratified CV, precision, recall, F1, confusion matrix, ROC-AUC
7. SHAP explainability analysis
8. Isotonic calibration + Brier score
9. All comparison plots (P/R/F1 bars, CV box plot, correlation heatmap, breach distribution, 4-model confusion matrices)

**Results (70/30 split):**

| Model | Test AUC | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| **XGBoost** | **0.9001** | **0.7967** | 0.7638 | 0.8325 |
| LightGBM | 0.8986 | 0.7930 | 0.7574 | 0.8320 |
| CatBoost | 0.8976 | 0.7953 | 0.7605 | 0.8334 |
| Random Forest | 0.8975 | 0.7947 | 0.7527 | 0.8416 |

**Feature Combination (XGBoost):**

| Feature Set | # Features | Test AUC | Test F1 |
|-------------|-----------|----------|---------|
| Basic (matches prior work) | 5 | 0.7946 | 0.6955 |
| + Temporal | 8 | 0.7886 | 0.6971 |
| + Operational | 12 | 0.8567 | 0.7523 |
| All features | 20 | 0.9001 | 0.7967 |

**Key finding:** Operational features (assignment delay, groups touched, team stats) give the biggest performance jump — from AUC 0.79 to 0.86.

---

### Loan Default Prediction (`Home_Credit_Default_Prediction.ipynb`)

Predicts whether a loan applicant will default using Home Credit data.

**Pipeline:**
1. Loads `application_train.csv` (307k rows, 122 columns)
2. Cleans DAYS_EMPLOYED anomaly, creates financial ratios
3. Aggregates features from `bureau.csv` and `previous_application.csv`
4. Engineers 25 features total from 3 tables
5. 70/30 stratified train-test split (215,257 / 92,254)
6. Trains 4 models with class imbalance handling (scale_pos_weight = 11.39)
7. Feature combination experiment: 5 → 10 → 16 → 25 features
8. Full validation: 5-fold stratified CV, SHAP, calibration, all plots

**Results (70/30 split):**

| Model | Test AUC | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| **CatBoost** | **0.7535** | 0.2650 | 0.1647 | 0.6779 |
| LightGBM | 0.7533 | 0.2677 | 0.1676 | 0.6653 |
| XGBoost | 0.7527 | 0.2734 | 0.1736 | 0.6430 |
| Random Forest | 0.7410 | 0.2706 | 0.1764 | 0.5816 |

**Feature Combination (XGBoost):**

| Feature Set | # Features | Test AUC | Test F1 |
|-------------|-----------|----------|---------|
| Financial basics | 5 | 0.6470 | 0.2025 |
| + External scores & ratios | 10 | 0.7413 | 0.2609 |
| + Behavioral/personal | 16 | 0.7422 | 0.2646 |
| All features | 25 | 0.7527 | 0.2734 |

**Key finding:** EXT_SOURCE scores are the strongest predictors — adding them jumps AUC by +0.094. Bureau and previous application history adds another +0.01. Low precision is expected with 8.1% default rate (heavy class imbalance).

**Calibration:** Brier score improved from 0.1836 (raw) → 0.0680 (calibrated).

---

## Automated Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets into bpi2014/ and home_credit/ folders

# 3. Run the BPI automated pipeline
python bpi2014_ml_pipeline.py
```

Output files go to `results_bpi/`:
- `model_comparison.csv` — All 4 models side by side
- `feature_combination_results.csv` — Feature set experiment
- `feature_justifications.csv` — Written justification for each feature
- `roc_all_models.png`, `confusion_matrix.png`, `calibration_plot.png`
- `shap_importance.png`, `shap_beeswarm.png`

---

## Project Structure

```
crm-lead/
├── README.md
├── requirements.txt
├── .gitignore
│
├── SLA_Breach_Prediction.ipynb         ← BPI 2014 notebook (21 sections)
├── Home_Credit_Default_Prediction.ipynb ← Home Credit notebook (20 sections)
├── bpi2014_ml_pipeline.py              ← Automated BPI pipeline script
│
├── bpi2014/                            ← BPI dataset (not in repo)
│   ├── Detail_Incident.csv
│   ├── Detail_Incident_Activity.csv
│   └── Detail_Interaction.csv
│
├── home_credit/                        ← Home Credit dataset (not in repo)
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── previous_application.csv
│   └── ... (other supplementary files)
│
└── results_bpi/                        ← Generated outputs from pipeline
    ├── model_comparison.csv
    ├── feature_combination_results.csv
    └── *.png (plots)
```
