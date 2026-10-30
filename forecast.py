"""Future skill-demand forecasting (2026-2030) + Skill Opportunity Score."""
import numpy as np, pandas as pd
from data_gen import SKILLS

def forecast_skills(df, horizon=range(2026, 2031)):
    yrs = np.sort(df.posting_year.unique()); rows = []
    for k, (prem, _, _) in SKILLS.items():
        share = df.groupby("posting_year")[f"skill_{k}"].mean().reindex(yrs).values
        # log-linear (compound growth) trend: suits exponentially growing skills
        b, a = np.polyfit(yrs, np.log(np.clip(share, 1e-3, None)), 1)
        proj = {y: float(min(np.exp(a + b * y), .95)) for y in horizon}
        cagr = np.exp(b) - 1
        has = df[df[f"skill_{k}"] == 1].salary_usd.median(); no = df[df[f"skill_{k}"] == 0].salary_usd.median()
        rows.append({"skill": k, "share_2025": share[-1], **{f"share_{y}": v for y, v in proj.items()},
                     "cagr_pct": cagr * 100, "median_salary_premium_pct": (has / no - 1) * 100,
                     "opportunity_score": cagr * 100 * max(prem, 0) * 10 + share[-1] * 10})
    return pd.DataFrame(rows).sort_values("opportunity_score", ascending=False).reset_index(drop=True)
