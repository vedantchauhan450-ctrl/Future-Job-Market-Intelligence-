"""Preprocessing: log-target, feature engineering, skew correction, ordinal/OHE/Target encoding."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, PowerTransformer

try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:  # sklearn < 1.3
    TargetEncoder = None

TARGET, LOG_TARGET = "salary_usd", "log_salary"
EDU_ORDER = ["High School", "Bachelor", "Master", "PhD"]
NUMERIC = ["experience_years", "posting_year", "SkillCount", "AI_Score", "ExpSquared", "ExpXAI"]
NOMINAL = ["job_title", "industry", "company_size", "remote_type"]

def load_data(path):
    return pd.read_csv(path)

def log_transform_target(df):
    df = df.copy(); df[LOG_TARGET] = np.log1p(df[TARGET]); return df

def add_features(df):
    """Composite + interaction features."""
    df = df.copy()
    skills = [c for c in df if c.startswith("skill_")]
    df["SkillCount"] = df[skills].sum(axis=1)
    df["AI_Score"] = df["skill_ml"] + 2 * df["skill_genai"]
    df["ExpSquared"] = df["experience_years"] ** 2
    df["ExpXAI"] = df["experience_years"] * df["AI_Score"]
    return df

def build_preprocessing_pipeline():
    city_enc = (TargetEncoder(target_type="continuous", random_state=0) if TargetEncoder
                else OneHotEncoder(handle_unknown="ignore"))
    ct = ColumnTransformer([
        ("num", PowerTransformer(method="yeo-johnson"), NUMERIC),
        ("edu", OrdinalEncoder(categories=[EDU_ORDER], handle_unknown="use_encoded_value", unknown_value=-1), ["education"]),
        ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), NOMINAL),
        ("city", city_enc, ["city"]),
        ("skills", "passthrough", make_column_selector(pattern="^skill_")),
    ])
    return Pipeline([("features", FunctionTransformer(add_features)), ("encode", ct)])
