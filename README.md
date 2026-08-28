# PAA-NSGA-II: Intelligent CRM Lead Assignment & SLA Prediction

**An Adaptive Multi-Criteria Intelligent Decision Framework for Lead Assignment and Predictive SLA Management in Customer Relationship Management Systems**

## Team
- Ashwanth S (24BCE2035)
- Ragunandhan T (24BCE0721)
- Anushanth L G (24BCE2094)
- Mohammed Asif A (24BCE2076)

**Guide:** Dr. Deepika J, School of Computer Science and Engineering, VIT Vellore

## Dataset

**BPI Challenge 2014** — Real operational data from Rabobank Nederland Group ICT (HP Service Manager).

| File | Records | What it provides |
|------|---------|-----------------|
| `Detail_Incident.csv` | 31,238 | Priority, impact, urgency, timestamps, handle time |
| `Detail_Incident_Activity.csv` | 210,837 | 166 assignment groups, assignments, reassignments |
| `Detail_Interaction.csv` | 68,358 | First call resolution, interaction handle time |

**Source:** [4TU.ResearchData — BPI Challenge 2014](https://data.4tu.nl/collections/BPI_Challenge_2014/5065469)

## Phase 3: SLA Breach Prediction (ML Pipeline)

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download BPI 2014 data (if not already in bpi2014/ folder)
#    Get the 3 CSVs from 4TU.ResearchData link above

# 3. Run the ML pipeline
python bpi2014_ml_pipeline.py
```

### What it does

1. **Loads** the 3 BPI 2014 files and merges them
2. **Engineers 26 features** from raw operational data (each with written justification)
3. **Trains 4 models**: XGBoost, LightGBM, CatBoost, Random Forest
4. **Feature combination experiment**: 5 → 8 → 12 → 26 features (XGBoost)
5. **Full validation**: 5-fold stratified CV, precision, recall, F1, confusion matrix, ROC-AUC
6. **Calibration**: Isotonic calibration + Brier score
7. **SHAP analysis**: Global feature importance + beeswarm plot

### Results

| Model | CV-AUC | Test AUC | F1 | Precision | Recall |
|-------|--------|----------|-----|-----------|--------|
| XGBoost | 0.8954 | 0.8958 | 0.7874 | 0.7550 | 0.8227 |
| LightGBM | 0.8939 | 0.8970 | 0.7864 | 0.7531 | 0.8227 |
| CatBoost | 0.8940 | 0.8954 | 0.7924 | 0.7584 | 0.8296 |
| Random Forest | 0.8908 | 0.8956 | 0.7917 | 0.7515 | 0.8364 |

**Feature Combination Results (XGBoost):**

| Feature Set | # Features | Test AUC | Test F1 |
|-------------|-----------|----------|---------|
| Basic (matches prior work) | 5 | 0.7889 | 0.6914 |
| + Temporal | 8 | 0.7827 | 0.6914 |
| + Operational | 12 | 0.8570 | 0.7529 |
| All features | 26 | 0.8958 | 0.7874 |

### Output Files

All outputs go to `results_bpi/`:
- `model_comparison.csv` — All 4 models side by side
- `feature_combination_results.csv` — 5/8/12/26 feature experiment
- `feature_justifications.csv` — Written justification for each feature
- `roc_all_models.png` — ROC curves
- `feature_combo_comparison.png` — AUC by feature set size
- `confusion_matrix.png` — Confusion matrix heatmap
- `calibration_plot.png` — Raw vs calibrated probability
- `shap_importance.png` — Feature importance bar chart
- `shap_beeswarm.png` — SHAP beeswarm plot

## Project Structure

```
crm-lead/
├── README.md
├── requirements.txt
├── .gitignore
├── bpi2014_ml_pipeline.py      ← Main ML pipeline
├── bpi2014/                    ← Dataset (not in repo, download from 4TU)
│   ├── Detail_Incident.csv
│   ├── Detail_Incident_Activity.csv
│   └── Detail_Interaction.csv
└── results_bpi/                ← Generated outputs
    ├── model_comparison.csv
    ├── feature_combination_results.csv
    ├── feature_justifications.csv
    └── *.png (plots)
```
