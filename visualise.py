"""Figures."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

def _save(fig, path): fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def plot_model_comparison(results, path):
    names = list(results); fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [results[n]["rmse_log"] for n in names], color="#3776ab")
    ax.set_ylabel("Validation RMSE (log scale)"); ax.set_title("Model comparison"); _save(fig, path)

def plot_pred_vs_actual(y_true_usd, y_pred_usd, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.scatter(y_true_usd, y_pred_usd, s=8, alpha=.4)
    m = [y_true_usd.min(), y_true_usd.max()]; ax.plot(m, m, "r--")
    ax.set_xlabel("Actual salary (USD)"); ax.set_ylabel("Predicted salary (USD)"); ax.set_title("Predicted vs actual"); _save(fig, path)

def plot_skill_forecast(df, path, top=6):
    fig, ax = plt.subplots(figsize=(8, 5)); cols = [c for c in df if c.startswith("share_")]
    for _, r in df.head(top).iterrows():
        ax.plot([int(c[6:]) for c in cols], [r[c] * 100 for c in cols], marker="o", label=r.skill)
    ax.set_ylabel("% of job postings requiring skill"); ax.set_title("Forecast: skill demand 2025-2030"); ax.legend(); _save(fig, path)

def plot_top_coefficients(names, coefs, path, k=15):
    idx = np.argsort(np.abs(coefs))[-k:]; fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[idx], coefs[idx], color=["#2a9d8f" if c > 0 else "#e76f51" for c in coefs[idx]])
    ax.set_title("Top salary drivers (log-salary coefficients)"); _save(fig, path)
