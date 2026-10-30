"""Synthetic job-postings generator (2020-2025) with realistic salary + skill-demand dynamics.
Swap with a real dataset (Kaggle / LinkedIn / Naukri scrape) by providing data/jobs.csv with the same columns."""
import numpy as np, pandas as pd

# skill: (log-salary premium, base demand share in 2020, annual demand growth)
SKILLS = {"python": (.10, .35, .05), "sql": (.05, .40, .01), "ml": (.16, .12, .15),
          "genai": (.22, .01, .90), "cloud": (.12, .20, .10), "kubernetes": (.11, .06, .12),
          "rust": (.14, .02, .25), "excel": (.01, .45, -.05), "cybersecurity": (.13, .08, .12),
          "data_eng": (.12, .10, .14)}
TITLES = {"Data Scientist": 105e3, "ML Engineer": 120e3, "Software Engineer": 100e3, "Data Analyst": 70e3,
          "DevOps Engineer": 108e3, "Security Analyst": 95e3, "Product Manager": 115e3, "Data Engineer": 110e3}
INDUSTRIES = {"Tech": .10, "Finance": .08, "Healthcare": .0, "Retail": -.05, "Manufacturing": -.04, "Consulting": .03}
CITIES = {"San Francisco": .30, "New York": .22, "Seattle": .20, "Austin": .08, "Bangalore": -.55,
          "London": .05, "Toronto": -.02, "Remote-Global": 0.0, "Pune": -.62, "Berlin": -.08}
EDU = {"High School": -.05, "Bachelor": 0.0, "Master": .06, "PhD": .12}
SIZES = {"Startup": -.03, "Mid": 0.0, "Large": .05}
WORK = {"onsite": 0.0, "hybrid": .01, "remote": .03}

def generate(n=6000, seed=42):
    r = np.random.default_rng(seed)
    p = lambda d: r.choice(list(d), n)
    df = pd.DataFrame({"job_title": p(TITLES), "industry": p(INDUSTRIES), "city": p(CITIES),
                       "education": r.choice(list(EDU), n, p=[.05, .5, .35, .10]),
                       "company_size": p(SIZES), "remote_type": r.choice(list(WORK), n, p=[.4, .35, .25]),
                       "experience_years": np.clip(r.gamma(2.2, 2.6, n), 0, 25).round(1),
                       "posting_year": r.choice(range(2020, 2026), n)})
    log_sal = np.log(df.job_title.map(TITLES).values)
    for k, (prem, base, g) in SKILLS.items():
        share = np.clip(base * (1 + g) ** (df.posting_year - 2020), 0, .92)
        df[f"skill_{k}"] = (r.random(n) < share).astype(int)
        log_sal += prem * df[f"skill_{k}"]
    log_sal += (df.industry.map(INDUSTRIES) + df.city.map(CITIES) + df.education.map(EDU)
                + df.company_size.map(SIZES) + df.remote_type.map(WORK)
                + .045 * np.sqrt(df.experience_years) * 3 + .028 * (df.posting_year - 2020)
                + r.normal(0, .11, n))
    df["salary_usd"] = np.exp(log_sal).round(-2)
    return df
