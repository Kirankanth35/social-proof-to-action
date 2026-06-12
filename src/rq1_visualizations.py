# rq1_visualizations.py
# Visuals for RQ1:
# 1) Early persuasive signals -> Amplification (reach) by authority tier
# 2) Amplification (reach) -> High-effort engagement by authority tier
# 3) Conditional indirect effects (a*b) bar chart (uses your regression coefficients)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf

# -----------------------------
# Load data (Excel in same folder as this script)
# -----------------------------
DATA_PATH = Path(__file__).resolve().parent / "dataset_algorithmic_persuasion_10000.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="data")
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

# -----------------------------
# Feature engineering
# -----------------------------
df["log_reach"] = np.log1p(df["reach"])
df["log_high_effort"] = np.log1p(df["high_effort_engagement"])
df["month"] = df["post_datetime"].dt.to_period("M").astype(str)

# Authority tiers (Low/Mid/High)
q1, q2 = df["authority_log"].quantile([1/3, 2/3])
df["authority_tier"] = pd.cut(
    df["authority_log"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Mid", "High"]
)

# Keep required columns
req = [
    "persuasive_power_index", "log_reach", "log_high_effort",
    "authority_log", "authority_tier", "influencer_id",
    "content_type", "post_format", "verified", "account_age_years", "month"
]
df = df.dropna(subset=req).copy()

# Z-scores (for model-based indirect effects)
def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

df["zX"] = zscore(df["persuasive_power_index"])
df["zW"] = zscore(df["authority_log"])

# -----------------------------
# Helper: binned mean line plot
# -----------------------------
def binned_line_plot(data, x, y, group, bins=12, xlabel=None, ylabel=None, title=None):
    """
    Plots mean(y) against mean(x) in quantile bins, separately for each group level.
    """
    plt.figure()
    levels = [g for g in ["Low", "Mid", "High"] if g in data[group].astype(str).unique()]
    for lvl in levels:
        sub = data[data[group].astype(str) == lvl][[x, y]].dropna().copy()
        sub["bin"] = pd.qcut(sub[x], q=bins, duplicates="drop")
        g = sub.groupby("bin", observed=True).agg(x_mean=(x, "mean"), y_mean=(y, "mean")).reset_index(drop=True)
        plt.plot(g["x_mean"], g["y_mean"], marker="o", label=lvl)

    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.title(title or f"{y} vs {x} by {group}")
    plt.legend(title=group)
    plt.tight_layout()

# -----------------------------
# FIGURE 1: Early signals -> Reach by authority tier
# -----------------------------
binned_line_plot(
    df,
    x="persuasive_power_index",
    y="log_reach",
    group="authority_tier",
    bins=12,
    xlabel="Persuasive Power Index (early signals)",
    ylabel="log1p(Reach)",
    title="RQ1 (a-path): Early Persuasive Signals → Amplification (Reach), by Authority Tier"
)

# -----------------------------
# FIGURE 2: Reach -> High-effort outcomes by authority tier
# -----------------------------
binned_line_plot(
    df,
    x="log_reach",
    y="log_high_effort",
    group="authority_tier",
    bins=12,
    xlabel="log1p(Reach)",
    ylabel="log1p(High-effort engagement)",
    title="RQ1 (b-path): Amplification (Reach) → High-effort Engagement, by Authority Tier"
)

# -----------------------------
# FIGURE 3: Conditional indirect effects (a*b) bar chart
# Fit the same models you ran, then compute indirect effects at zW = -1, 0, +1
# -----------------------------
CLUSTER = "influencer_id"
controls = "C(content_type) + C(post_format) + verified + account_age_years + C(month)"
M = "log_reach"
Y = "log_high_effort"

formula_a = f"{M} ~ zX + zW + zX:zW + {controls}"
formula_b = f"{Y} ~ {M} + zX + zW + {M}:zW + zX:zW + {controls}"

def fit_clustered(formula, data):
    return smf.ols(formula=formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[CLUSTER]}
    )

ma = fit_clustered(formula_a, df)
mb = fit_clustered(formula_b, df)

def indirect_at_W(ma, mb, W_value):
    a1 = ma.params.get("zX", np.nan)
    a3 = ma.params.get("zX:zW", 0.0)
    b1 = mb.params.get(M, np.nan)
    b3 = mb.params.get(f"{M}:zW", 0.0)
    return (a1 + a3 * W_value) * (b1 + b3 * W_value)

W_levels = {"Low authority (z=-1)": -1.0, "Mid authority (z=0)": 0.0, "High authority (z=+1)": 1.0}
indirect_vals = [indirect_at_W(ma, mb, w) for w in W_levels.values()]

plt.figure()
plt.bar(list(W_levels.keys()), indirect_vals)
plt.ylabel("Conditional Indirect Effect (a × b)")
plt.title("RQ1: Conditional Indirect Effect of Early Signals on High-effort Outcomes\nthrough Amplification, by Authority Level")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.show()