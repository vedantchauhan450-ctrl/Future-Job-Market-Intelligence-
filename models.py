"""Ridge / Lasso / ElasticNet wrappers + stacked ensemble (non-negative weights on out-of-fold preds)."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from preprocessing import build_preprocessing_pipeline

@dataclass
class RegressionMetrics:
    rmse_log: float; r2: float; median_abs_error_usd: float; mape_pct: float
    def as_dict(self): return self.__dict__

def score(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(p_log)
    return RegressionMetrics(float(np.sqrt(mean_squared_error(y_log, p_log))), float(r2_score(y_log, p_log)),
                             float(median_absolute_error(y, p)), float(np.mean(np.abs(y - p) / y) * 100))

class SalaryModel:
    def __init__(self, name, regressor):
        self.name = name
        self.pipe = Pipeline([("prep", build_preprocessing_pipeline()), ("reg", regressor)])
    @classmethod
    def ridge(cls, alpha=3.0): return cls("Ridge", Ridge(alpha=alpha))
    @classmethod
    def lasso(cls, alpha=1e-3): return cls("Lasso", Lasso(alpha=alpha, max_iter=20000))
    @classmethod
    def elasticnet(cls, alpha=1e-3, l1_ratio=.5): return cls("ElasticNet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000))
    def fit(self, X, y_log): self.pipe.fit(X, y_log); return self
    def predict_log(self, X): return self.pipe.predict(X)
    def predict_salary(self, X): return np.expm1(self.predict_log(X))
    def evaluate(self, X, y_log): return score(y_log, self.predict_log(X))

class StackedEnsemble:
    def __init__(self, models): self.models, self.weights = models, None
    def fit(self, X, y_log, cv=5):
        oof = np.column_stack([cross_val_predict(clone(m.pipe), X, y_log, cv=KFold(cv, shuffle=True, random_state=0)) for m in self.models])
        w, _ = nnls(oof, y_log); self.weights = w / w.sum()
        for m in self.models: m.fit(X, y_log)
        return self
    def predict_log(self, X): return np.column_stack([m.predict_log(X) for m in self.models]) @ self.weights
    def predict_salary(self, X): return np.expm1(self.predict_log(X))
    def evaluate(self, X, y_log): return score(y_log, self.predict_log(X))
