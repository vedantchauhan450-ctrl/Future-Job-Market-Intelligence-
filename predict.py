"""Inference: python src/predict.py --model_path models/ensemble.joblib --input_csv data/jobs.csv"""
import argparse, joblib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/ensemble.joblib"); ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default="outputs/predictions.csv"); a = ap.parse_args()
    df = pd.read_csv(a.input_csv).drop(columns=["salary_usd", "log_salary"], errors="ignore")
    df["predicted_salary_usd"] = joblib.load(a.model_path).predict_salary(df).round(-2)
    df.to_csv(a.output_csv, index=False); print(df[["job_title", "city", "experience_years", "predicted_salary_usd"]].head())

if __name__ == "__main__": main()
