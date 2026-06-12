# main_analysis.py
# Main analytics script for your project dataset:
# Title: From Social Proof to Action: Algorithmic Amplification as a Conditional Mechanism
# RQs covered:
#   RQ1: Mediation + authority shaping
#   RQ2: Diminishing marginal returns of reach
#   RQ3: High visibility reduces PP importance

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

# =============================
# 0) Load dataset
# =============================
DATA_PATH = Path(__file__).resolve().parent / "dataset_algorithmic_persuasion_10000.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="data")
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

# =============================
# 1) Feature engineering
# =============================
# Core transforms
df["log_reach"] = np.log1p(df["reach"])
df["log_high_effort"] = np.log1p(df["high_effort_engagement"])
df["month"] = df["post_datetime"].dt.to_period("M").astype(str)

# Nonlinearity for diminishing returns
df["log_reach_sq"] = df["log_reach"] ** 2

# Authority tiers (optional for descriptives/plots)
q1, q2 = df["authority_log"].quantile([1/3, 2/3])
df["authority_tier"] = pd.cut(
    df["authority_log"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Mid", "High"]
)

# Convenience engineered features (optional)
df["VE_log"] = np.log1p(df["high_effort_engagement"]) - np.log1p(np.maximum(df["reach"], 1))

# =============================
# 2) Setup: analysis choices
# =============================
CLUSTER = "influencer_id"

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

# Standardize key variables for interactions (recommended)
df["z_PP"] = zscore(df["persuasive_power_index"])
df["z_Authority"] = zscore(df["authority_log"])
df["z_log_reach"] = zscore(df["log_reach"])

# Controls (consistent with your thesis plan)
controls = "C(content_type) + C(post_format) + verified + account_age_years + C(month)"

# Keep needed rows
need = [
    "z_PP", "z_Authority", "log_reach", "log_high_effort", "log_reach_sq", "very_high_reach",
    "content_type", "post_format", "verified", "account_age_years", "month", CLUSTER
]
dfm = df.dropna(subset=need).copy()

def fit_clustered_ols(formula: str, data: pd.DataFrame):
    """OLS with clustered SEs by influencer_id."""
    return smf.ols(formula=formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[CLUSTER]}
    )

# =============================
# 3) RQ1: Mediation + authority shaping
# =============================
# a-path: Amplification model
# M = log_reach
formula_a = f"log_reach ~ z_PP + z_Authority + z_PP:z_Authority + {controls}"
model_a = fit_clustered_ols(formula_a, dfm)

# b/c' path: High-effort outcome model
# Y = log_high_effort
formula_b = f"log_high_effort ~ log_reach + z_PP + z_Authority + log_reach:z_Authority + z_PP:z_Authority + {controls}"
model_b = fit_clustered_ols(formula_b, dfm)

print("\n" + "="*90)
print("RQ1: Mediation + Authority shaping")
print("="*90)
print("\n[a-path] log_reach ~ PP + Authority + PP×Authority + Controls (clustered SEs)")
print(model_a.summary())

print("\n[b/c'] log_high_effort ~ log_reach + PP + Authority + log_reach×Authority + PP×Authority + Controls (clustered SEs)")
print(model_b.summary())

# Conditional indirect effect (moderated mediation): a(W)*b(W)
def indirect_at_W(ma, mb, W_value):
    # a(W) = a1 + a3*W
    a1 = ma.params.get("z_PP", np.nan)
    a3 = ma.params.get("z_PP:z_Authority", 0.0)
    aW = a1 + a3 * W_value

    # b(W) = b1 + b3*W
    b1 = mb.params.get("log_reach", np.nan)
    b3 = mb.params.get("log_reach:z_Authority", 0.0)
    bW = b1 + b3 * W_value

    return aW * bW

print("\nConditional indirect effects (a*b) at authority z = -1, 0, +1:")
for label, Wz in [("LOW", -1.0), ("MID", 0.0), ("HIGH", 1.0)]:
    print(f"  {label}: {indirect_at_W(model_a, model_b, Wz):.4f}")

# Cluster bootstrap CIs (resample influencers)
def cluster_bootstrap_indirect(data, n_boot=500, seed=123):
    rng = np.random.default_rng(seed)
    ids = data[CLUSTER].unique()

    vals = {"low": [], "mid": [], "high": []}
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([data[data[CLUSTER] == i] for i in sampled], ignore_index=True)

        try:
            ma = fit_clustered_ols(formula_a, boot)
            mb = fit_clustered_ols(formula_b, boot)
            vals["low"].append(indirect_at_W(ma, mb, -1.0))
            vals["mid"].append(indirect_at_W(ma, mb, 0.0))
            vals["high"].append(indirect_at_W(ma, mb, 1.0))
        except Exception:
            continue

    def ci(arr):
        arr = np.asarray(arr, dtype=float)
        return np.nanpercentile(arr, [2.5, 50, 97.5])

    return {k: ci(v) for k, v in vals.items()}

cis = cluster_bootstrap_indirect(dfm, n_boot=500, seed=123)
print("\nCluster-bootstrap 95% CIs for conditional indirect effects:")
for k in ["low", "mid", "high"]:
    lo, med, hi = cis[k]
    print(f"  {k.upper():4s}: 2.5%={lo:.4f} | 50%={med:.4f} | 97.5%={hi:.4f}")

# =============================
# 4) RQ2: Diminishing marginal returns of reach
# =============================
formula_rq2 = f"log_high_effort ~ log_reach + log_reach_sq + z_Authority + {controls}"
model_rq2 = fit_clustered_ols(formula_rq2, dfm)

print("\n" + "="*90)
print("RQ2: Diminishing marginal returns of reach")
print("="*90)
print(model_rq2.summary())

b1 = model_rq2.params.get("log_reach", np.nan)
b2 = model_rq2.params.get("log_reach_sq", np.nan)

if np.isfinite(b1) and np.isfinite(b2) and b2 != 0:
    # dY/dlog_reach = b1 + 2*b2*log_reach ; set to 0 for turning point
    log_star = -b1 / (2 * b2)
    reach_star = np.exp(log_star) - 1
    print("\nEstimated turning point (where marginal effect ~ 0):")
    print(f"  log1p(reach) ≈ {log_star:.4f}")
    print(f"  reach ≈ {reach_star:,.0f}")
    print("Interpretation: beyond this reach, extra visibility yields little additional high-effort engagement.")
else:
    print("\nTurning point not computed (b2 missing/zero).")

# =============================
# 5) RQ3: Does high visibility reduce PP importance?
# =============================
# A) Continuous interaction: PP × log_reach
formula_rq3a = f"log_high_effort ~ z_PP + log_reach + z_PP:log_reach + z_Authority + {controls}"
model_rq3a = fit_clustered_ols(formula_rq3a, dfm)

# B) Tail regime: PP × very_high_reach
formula_rq3b = f"log_high_effort ~ z_PP + log_reach + very_high_reach + z_PP:very_high_reach + z_Authority + {controls}"
model_rq3b = fit_clustered_ols(formula_rq3b, dfm)

print("\n" + "="*90)
print("RQ3: High visibility reduces PP importance?")
print("="*90)
print("\n[RQ3-A] Continuous interaction (PP × log_reach)")
print(model_rq3a.summary())

print("\n[RQ3-B] Tail regime interaction (PP × very_high_reach)")
print(model_rq3b.summary())

# =============================
# 6) Compact coefficient table (for reporting)
# =============================
def coef_table(model, keep):
    rows = []
    for term in keep:
        if term in model.params.index:
            rows.append({
                "term": term,
                "coef": model.params[term],
                "se": model.bse[term],
                "p": model.pvalues[term],
            })
    return pd.DataFrame(rows)

print("\n" + "="*90)
print("Key coefficients (quick report)")
print("="*90)

keep_a = ["z_PP", "z_Authority", "z_PP:z_Authority"]
keep_b = ["log_reach", "log_reach:z_Authority", "z_PP", "z_PP:z_Authority"]
keep_rq2 = ["log_reach", "log_reach_sq"]
keep_rq3a = ["z_PP", "log_reach", "z_PP:log_reach"]
keep_rq3b = ["z_PP", "very_high_reach", "z_PP:very_high_reach"]

print("\n[a-path] key terms:")
print(coef_table(model_a, keep_a).to_string(index=False))

print("\n[b/c'] key terms:")
print(coef_table(model_b, keep_b).to_string(index=False))

print("\n[RQ2] key terms:")
print(coef_table(model_rq2, keep_rq2).to_string(index=False))

print("\n[RQ3-A] key terms:")
print(coef_table(model_rq3a, keep_rq3a).to_string(index=False))

print("\n[RQ3-B] key terms:")
print(coef_table(model_rq3b, keep_rq3b).to_string(index=False))

print("\nDone.")