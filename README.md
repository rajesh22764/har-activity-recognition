# Human Activity Recognition from Smartphone Sensors

End-to-end data-science project on the UCI HAR dataset, built to demonstrate the skills in the
ZS Insights & Analytics Associate Data Scientist role (`docs/ADS A JD.docx`).
**Read [REPORT.md](REPORT.md) first** for the findings; this file explains how the project is organised.

## What it shows (mapped to the job description)
| JD requirement | Where |
|---|---|
| High-dimensional algorithms | `src/s02_dimensionality.py` — 561 features: PCA vs. ANOVA / L1 / RF selection |
| Hypothesis testing | `src/s01_eda_tests.py` — Kruskal-Wallis + FDR, Mann-Whitney, effect sizes; `s05` — McNemar, χ² |
| Machine learning | `src/s03_models.py` — LogReg / SVM / RF / LightGBM, **subject-grouped CV**, leakage demo |
| Deep learning | `src/s04_deep.py` — 1D-CNN (PyTorch) on raw signals, 3-seed ensemble |
| Explainability / communication | `src/s05_explain_fairness.py` (SHAP, per-subject fairness, review trade-off), `REPORT.md` |
| Generative AI | `src/genai_summary.py` — Claude narrates computed facts (template fallback offline) |
| Engineering integration | `app/streamlit_app.py`, `aws/` (SageMaker handler, Dockerfile — not deployed), `run_all.sh` |

## Headline results (held-out subjects)
SVM 95.4 % · Logistic Regression 95.4 % · LightGBM 95.0 % · 1D-CNN 93.2 % · Random Forest 92.8 %.
Random-row CV inflates RF accuracy to 98.2 % vs. 91.8 % with subject-grouped CV.

## Run it
```bash
conda env create -f environment.yml   # or: conda activate har-zs (already created on this machine)
./run_all.sh                          # ~5 min on a laptop CPU; regenerates reports/ and models/
streamlit run app/streamlit_app.py
```
The raw data is not stored in git. Download "Human Activity Recognition Using Smartphones" from the UCI ML Repository and extract it so that `data/raw/UCI HAR Dataset/` exists. Outputs: `reports/*.csv|json`, `reports/figures/*.png`, `models/`.

## Layout
```
src/        common.py, s01..s05 analysis steps, genai_summary.py
app/        Streamlit demo
aws/        inference handler, Dockerfile, deployment notes
reports/    metrics, tables, figures
models/     saved sklearn / LightGBM / PyTorch models
docs/       the job description
```
