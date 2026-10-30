"""End-to-end training CLI."""
import argparse, os, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from data_gen import generate
from forecast import forecast_skills
from models import SalaryModel, StackedEnsemble
from preprocessing import LOG_TARGET, TARGET, load_data, log_transform_target
from tuning import tune
import visualise as viz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/"); ap.add_argument("--models_dir", default="models/")
    ap.add_argument("--n_rows", type=int, default=6000); ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n_trials", type=int, default=40); ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--no_plots", action="store_true"); a = ap.parse_args()
    for d in (a.data_dir, a.models_dir, "outputs", "figures"): os.makedirs(d, exist_ok=True)

    csv = os.path.join(a.data_dir, "jobs.csv")
    if not os.path.exists(csv): generate(a.n_rows).to_csv(csv, index=False); print(f"Generated synthetic data -> {csv}")
    df = log_transform_target(load_data(csv))
    X, y = df.drop(columns=[TARGET, LOG_TARGET]), df[LOG_TARGET]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=.15, random_state=42)

    if a.tune:
        models = []
        for k in ("ridge", "lasso", "elasticnet"):
            m, best = tune(k, Xtr, ytr, a.n_trials); models.append(m); print("tuned", k, best)
    else: models = [SalaryModel.ridge(), SalaryModel.lasso(), SalaryModel.elasticnet()]

    results = {}
    for m in models: m.fit(Xtr, ytr); results[m.name] = m.evaluate(Xva, yva).as_dict()
    ens = StackedEnsemble(models).fit(Xtr, ytr); results["Stacked Ensemble"] = ens.evaluate(Xva, yva).as_dict()
    print(pd.DataFrame(results).T.round(4).to_string())
    print("Ensemble weights:", dict(zip([m.name for m in models], ens.weights.round(3))))
    pd.DataFrame(results).T.to_csv("outputs/model_comparison.csv")

    fc = forecast_skills(load_data(csv)); fc.to_csv("outputs/skill_forecast_2030.csv", index=False)
    print("\nTop future skills:\n", fc[["skill", "share_2025", "share_2030", "cagr_pct", "opportunity_score"]].round(2).to_string(index=False))

    if a.save_models:
        for m in models: joblib.dump(m, os.path.join(a.models_dir, f"{m.name.lower()}.joblib"))
        joblib.dump(ens, os.path.join(a.models_dir, "ensemble.joblib"))
    if not a.no_plots:
        viz.plot_model_comparison(results, "figures/model_comparison.png")
        viz.plot_pred_vs_actual(np.expm1(yva), ens.predict_salary(Xva), "figures/pred_vs_actual.png")
        viz.plot_skill_forecast(fc, "figures/skill_forecast.png")
        en = models[2].pipe; names = en.named_steps["prep"].named_steps["encode"].get_feature_names_out()
        viz.plot_top_coefficients(names, en.named_steps["reg"].coef_, "figures/top_drivers.png")

if __name__ == "__main__": main()
