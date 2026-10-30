"""Hyperparameter tuning: Optuna (TPE) with RandomizedSearchCV fallback."""
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from models import SalaryModel

def tune(kind, X, y_log, n_trials=40, seed=0):
    cv = KFold(5, shuffle=True, random_state=seed)
    try:
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return _fallback(kind, X, y_log, n_trials, cv, seed)

    def build(p):
        if kind == "ridge": return SalaryModel.ridge(p["alpha"])
        if kind == "lasso": return SalaryModel.lasso(p["alpha"])
        return SalaryModel.elasticnet(p["alpha"], p["l1_ratio"])
    def obj(t):
        p = {"alpha": t.suggest_float("alpha", 1e-2, 100, log=True) if kind == "ridge" else t.suggest_float("alpha", 1e-5, 1e-1, log=True)}
        if kind == "elasticnet": p["l1_ratio"] = t.suggest_float("l1_ratio", .1, .95)
        return -cross_val_score(build(p).pipe, X, y_log, cv=cv, scoring="neg_root_mean_squared_error").mean()
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return build(st.best_params), st.best_params

def _fallback(kind, X, y_log, n, cv, seed):
    base = getattr(SalaryModel, kind)()
    grid = {"reg__alpha": np.logspace(-4, 1, 30)}
    if kind == "elasticnet": grid["reg__l1_ratio"] = np.linspace(.1, .95, 8)
    s = RandomizedSearchCV(base.pipe, grid, n_iter=n, cv=cv, scoring="neg_root_mean_squared_error", random_state=seed).fit(X, y_log)
    base.pipe = s.best_estimator_
    return base, s.best_params_
