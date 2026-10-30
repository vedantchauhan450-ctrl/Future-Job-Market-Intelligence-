# 🔮 Future Job Market Intelligence

### Salary Prediction & Skill-Demand Forecasting with Regularised Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f89939?logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **An end-to-end supervised learning pipeline** that (1) predicts job salaries from role, skills, location and experience using Ridge, Lasso and ElasticNet with Bayesian tuning, and (2) **forecasts which skills will dominate the job market through 2030**, ranking them by a *Skill Opportunity Score*.

---

## Table of Contents
- [Overview](#overview) · [Dataset](#dataset) · [Project Structure](#project-structure) · [Methodology](#methodology)
- [Results](#results) · [Quick Start](#quick-start) · [Module Reference](#module-reference) · [Limitations](#limitations) · [Roadmap](#roadmap)

---

## Overview

Careers, hiring budgets and education choices all depend on two questions: **"What should this role pay?"** and **"Which skills will matter tomorrow?"** This project answers both in one pipeline:

| Part | Question | Technique |
|------|----------|-----------|
| **A. Salary Intelligence** | What is a fair salary for this profile? | Regularised regression on log-salary |
| **B. Future Skill Forecast** | Which skills will grow by 2030? | Log-linear (compound-growth) trend per skill |
| **C. Opportunity Ranking** | Which skills are worth learning now? | Growth × salary premium × current demand |

**Key techniques demonstrated**

| Technique | Purpose |
|-----------|---------|
| Log-transform of `salary_usd` | Fix right-skew; model *percentage* rather than absolute error |
| Yeo-Johnson power transform | Correct skew in numeric predictors (experience², interactions) |
| Ordinal encoding | Preserve order of education (HS < Bachelor < Master < PhD) |
| One-Hot encoding | Nominal features: job title, industry, company size, work mode |
| **Target encoding** | High-cardinality `city` feature (falls back to OHE on old sklearn) |
| Feature engineering | `SkillCount`, `AI_Score`, `ExpSquared`, `ExpXAI` interaction |
| Ridge / Lasso / ElasticNet | Control overfitting & correlated skill features |
| Optuna (TPE) | Bayesian hyperparameter search (RandomizedSearchCV fallback) |
| Stacked ensemble | Non-negative least-squares blend on out-of-fold predictions |
| Trend forecasting | Compound-growth projection of skill demand to 2030 |

---

## Dataset

The repo ships with a **synthetic job-postings generator** (`src/data_gen.py`) so the project runs instantly with zero downloads. It encodes realistic structure: salary premiums per skill, city cost multipliers, an upward wage trend, and per-skill demand growth (e.g. GenAI exploding, Excel declining).

| Property | Value |
|----------|-------|
| Rows | 6,000 (configurable via `--n_rows`) |
| Period | 2020 – 2025 |
| Target | `salary_usd` |
| Features | `job_title`, `industry`, `city`, `education`, `company_size`, `remote_type`, `experience_years`, `posting_year`, + 10 binary skill flags (`skill_python`, `skill_genai`, ...) |

**Using real data:** drop any CSV at `data/jobs.csv` with the same columns (e.g. scraped from Kaggle / LinkedIn / Naukri) and the pipeline works unchanged.

---

## Project Structure

```
job-market-intelligence/
├── data/                    ← jobs.csv (auto-generated or your own)
├── src/
│   ├── data_gen.py          ← Synthetic job-market generator
│   ├── preprocessing.py     ← Feature engineering + encoding pipeline
│   ├── models.py            ← Ridge/Lasso/ElasticNet wrappers + StackedEnsemble
│   ├── tuning.py            ← Optuna / RandomizedSearchCV tuning
│   ├── forecast.py          ← Skill-demand forecast + opportunity score
│   ├── visualise.py         ← Figures
│   ├── train.py             ← End-to-end CLI
│   └── predict.py           ← Inference on new postings
├── models/                  ← Saved .joblib models
├── outputs/                 ← model_comparison.csv, skill_forecast_2030.csv, predictions.csv
├── figures/                 ← Generated plots
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Target — Log Transform
Salaries are right-skewed and multiplicative (a skill adds ~X%, not +$X). Training on `log1p(salary)` stabilises variance, makes residuals near-symmetric, and optimises relative error. Predictions are mapped back with `expm1`.

### 2. Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `SkillCount` | Σ skill flags | Breadth of skill-stack |
| `AI_Score` | `ml + 2×genai` | Weighted AI exposure |
| `ExpSquared` | `experience²` | Non-linear seniority curve |
| `ExpXAI` | `experience × AI_Score` | Senior AI talent premium |

### 3. Encoding
- **Ordinal:** `education`
- **One-Hot (`drop='first'`):** `job_title`, `industry`, `company_size`, `remote_type`
- **Target encoding:** `city` (cross-fitted internally to avoid target leakage)
- **Passthrough:** binary skill flags

### 4. Regularisation
$$\text{Ridge: } \min_\beta \|y-X\beta\|^2+\lambda\|\beta\|_2^2 \qquad \text{Lasso: } \min_\beta \|y-X\beta\|^2+\lambda\|\beta\|_1$$
$$\text{ElasticNet: } \min_\beta \|y-X\beta\|^2+\lambda\left[\rho\|\beta\|_1+\tfrac{1-\rho}{2}\|\beta\|_2^2\right]$$

Skill flags are correlated (ML ↔ Python ↔ GenAI), so shrinkage keeps coefficients stable. Lasso can zero-out weak signals; ElasticNet balances both behaviours.

### 5. Hyperparameter Tuning
`alpha` (and `l1_ratio`) are tuned with **Optuna TPE** minimising 5-fold CV RMSE on the training split only.

### 6. Stacking
Out-of-fold predictions from each model are blended with **non-negative least squares**; weights are normalised to sum to 1.

### 7. Future Skill Forecast
For each skill, the yearly share of postings requiring it is fitted with `log(share) = a + b·year`. This captures compound growth (CAGR = e^b − 1) and is projected to 2026–2030, capped at 95%.

**Skill Opportunity Score** = `CAGR% × salary_premium × 10 + current_share × 10` — rewards skills that are *growing fast*, *pay well* and *already have real demand*.

---

## Results

Held-out 15% validation set (seed 42, default hyperparameters, 6,000 synthetic rows):

| Model | RMSE (log) | R² | Median abs. error | MAPE |
|-------|-----------:|----:|------------------:|-----:|
| Ridge | 0.1112 | 0.918 | $14,096 | 8.9% |
| Lasso | 0.1152 | 0.913 | $14,290 | 9.2% |
| ElasticNet | 0.1132 | 0.915 | $14,225 | 9.0% |
| **Stacked Ensemble** | **0.1112** | **0.918** | **$14,096** | **8.9%** |

> The ensemble put 100% weight on Ridge here, which makes sense: the synthetic data is close to linear in log-space with little feature redundancy, so one model already captures everything. On noisier real data, expect the blend to spread weights. Run with `--tune` for tuned values.

### Skill Forecast (excerpt)

| Skill | Share 2025 | Forecast 2030 | CAGR | Opportunity Score |
|-------|-----------:|--------------:|-----:|------------------:|
| GenAI / LLMs | 24% | 95% (cap) | +83% | 185.3 |
| Rust | 7% | 22% | +28% | 39.9 |
| Machine Learning | 27% | 61% | +19% | 32.7 |
| Data Engineering | 23% | 52% | +19% | 24.7 |
| Cybersecurity | 15% | 25% | +12% | 16.8 |
| Excel | 34% | 27% | −5% | 2.9 |

Full table: `outputs/skill_forecast_2030.csv`. Figures are saved to `figures/`.

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/job-market-intelligence.git
cd job-market-intelligence
pip install -r requirements.txt

# 2. Train (auto-generates data/jobs.csv on first run)
python src/train.py --save_models

# 2b. With Bayesian tuning
python src/train.py --tune --n_trials 40 --save_models

# 3. Predict salaries for new postings
python src/predict.py --model_path models/ensemble.joblib --input_csv data/jobs.csv
```

### CLI Options — `train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/` | Folder containing `jobs.csv` |
| `--models_dir` | `models/` | Where models are saved |
| `--n_rows` | 6000 | Rows of synthetic data to generate |
| `--tune` | off | Enable Optuna tuning |
| `--n_trials` | 40 | Trials per model |
| `--save_models` | off | Persist models with joblib |
| `--no_plots` | off | Skip figure generation |

---

## Module Reference

| Module | Key symbols |
|--------|-------------|
| `preprocessing.py` | `log_transform_target()`, `add_features()`, `build_preprocessing_pipeline()` |
| `models.py` | `SalaryModel.ridge/lasso/elasticnet()`, `.fit()`, `.evaluate()`, `.predict_salary()`, `StackedEnsemble` |
| `tuning.py` | `tune(kind, X, y_log, n_trials)` |
| `forecast.py` | `forecast_skills(df)` |
| `visualise.py` | `plot_model_comparison`, `plot_pred_vs_actual`, `plot_skill_forecast`, `plot_top_coefficients` |

---

## Limitations

- **Synthetic data:** metrics describe the pipeline, *not* the real job market. Validate on real postings before drawing conclusions.
- **Trend extrapolation:** log-linear growth is naive — real skills follow S-curves and saturate. The 95% cap is a pragmatic guard, not a prediction.
- **Salary is not causal:** a skill's premium may reflect seniority or role, not the skill itself.
- **Linear models:** they miss deep interactions (e.g. title × city). Tree-based models are a natural next step.

## Roadmap
- [ ] Plug in real scraped postings + NLP skill extraction from job descriptions
- [ ] Gradient boosting (LightGBM / XGBoost) comparison
- [ ] Prophet / S-curve (logistic) demand forecasting with confidence intervals
- [ ] Streamlit dashboard: *"enter your skills → get salary + best skill to learn next"*
- [ ] Explainability with SHAP

## References
- Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity*. CRC Press.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12.
- Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD '19*.
- Micci-Barreca, D. (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes. *SIGKDD Explorations*, 3(1).

---
Built for the Supervised Learning (Regression) module · MIT License
