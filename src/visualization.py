import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# 1) Load data
# -----------------------------
DATA_PATH = Path(__file__).resolve().parent / "dataset_algorithmic_persuasion_10000.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="data")

# Basic cleaning / types
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

# -----------------------------
# 2) Feature engineering (used by your 3 final RQs)
# -----------------------------
# Logs for heavy-tailed counts
df["log_reach"] = np.log1p(df["reach"])
df["log_impressions"] = np.log1p(df["impressions"])
df["log_high_effort"] = np.log1p(df["high_effort_engagement"])

# Visibility Efficiency (how well reach turns into high-effort action)
df["VE_reach"] = df["high_effort_engagement"] / np.maximum(df["reach"], 1)
df["VE_impr"] = df["high_effort_engagement"] / np.maximum(df["impressions"], 1)
df["VE_log"] = np.log1p(df["high_effort_engagement"]) - np.log1p(np.maximum(df["reach"], 1))

# Early-to-total ratio (use likes because early_likes <= total_likes consistently in this dataset)
df["ETR_like"] = df["early_likes"] / np.maximum(df["total_likes"], 1)

# Authority tiers (low / mid / high) from authority_log
q1, q2 = df["authority_log"].quantile([1/3, 2/3])
df["authority_tier"] = pd.cut(
    df["authority_log"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Mid", "High"]
)

# Reach bins (for diminishing returns visuals)
df["reach_bin"] = pd.qcut(df["reach"], q=10, duplicates="drop")

# -----------------------------
# 3) Helper: binned mean plot
# -----------------------------
def plot_binned_mean(x, y, bins=20, xlabel="", ylabel="", title=""):
    """Bin x into equal-frequency bins and plot mean y per bin."""
    tmp = df[[x, y]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[x], q=bins, duplicates="drop")
    grouped = tmp.groupby("bin", observed=True).agg(
        x_mean=(x, "mean"),
        y_mean=(y, "mean"),
        n=(y, "size")
    ).reset_index(drop=True)

    plt.figure()
    plt.plot(grouped["x_mean"], grouped["y_mean"], marker="o")
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.title(title or f"Binned mean of {y} vs {x}")
    plt.tight_layout()


# -----------------------------
# 4) Core descriptive plots
# -----------------------------

# (A) Distribution: Reach (log scale view)
plt.figure()
plt.hist(df["log_reach"].dropna(), bins=50)
plt.xlabel("log1p(reach)")
plt.ylabel("Count")
plt.title("Distribution of Reach (log1p)")
plt.tight_layout()

# (B) Distribution: High-effort engagement (log scale)
plt.figure()
plt.hist(df["log_high_effort"].dropna(), bins=50)
plt.xlabel("log1p(high_effort_engagement)")
plt.ylabel("Count")
plt.title("Distribution of High-Effort Engagement (log1p)")
plt.tight_layout()

# (C) Category counts: content_type
plt.figure()
df["content_type"].value_counts().plot(kind="bar")
plt.xlabel("content_type")
plt.ylabel("Count")
plt.title("Posts by Content Type")
plt.tight_layout()

# (D) Category counts: post_format
plt.figure()
df["post_format"].value_counts().plot(kind="bar")
plt.xlabel("post_format")
plt.ylabel("Count")
plt.title("Posts by Format")
plt.tight_layout()

# -----------------------------
# 5) Visuals tied to your 3 accepted RQs
# -----------------------------

# RQ1: Early persuasive signals -> amplification -> high-effort outcomes (+ authority)
# (E) Persuasive power vs amplification (binned mean trend)
plot_binned_mean(
    x="persuasive_power_index",
    y="log_reach",
    bins=25,
    xlabel="persuasive_power_index",
    ylabel="log1p(reach)",
    title="RQ1: Early Persuasive Signals vs Amplification"
)

# (F) Amplification vs high-effort outcomes (binned mean trend)
plot_binned_mean(
    x="log_reach",
    y="log_high_effort",
    bins=25,
    xlabel="log1p(reach)",
    ylabel="log1p(high_effort_engagement)",
    title="RQ1: Amplification vs High-Effort Engagement"
)

# (G) Authority differences: amplification by authority tier
plt.figure()
tier_order = ["Low", "Mid", "High"]
data_by_tier = [df.loc[df["authority_tier"] == t, "log_reach"].dropna() for t in tier_order]
plt.boxplot(data_by_tier, labels=tier_order, showfliers=False)
plt.xlabel("authority_tier")
plt.ylabel("log1p(reach)")
plt.title("RQ1: Amplification by Authority Tier (boxplot)")
plt.tight_layout()

# (H) Authority differences: high-effort outcomes by authority tier
plt.figure()
data_by_tier = [df.loc[df["authority_tier"] == t, "log_high_effort"].dropna() for t in tier_order]
plt.boxplot(data_by_tier, labels=tier_order, showfliers=False)
plt.xlabel("authority_tier")
plt.ylabel("log1p(high_effort_engagement)")
plt.title("RQ1: High-Effort Engagement by Authority Tier (boxplot)")
plt.tight_layout()


# RQ2: Where does marginal effect of reach diminish? (diminishing returns)
# (I) Mean high-effort by reach deciles (shows flattening)
reach_deciles = (
    df.groupby("reach_bin", observed=True)
      .agg(reach_mean=("reach", "mean"),
           he_mean=("high_effort_engagement", "mean"),
           he_log_mean=("log_high_effort", "mean"),
           n=("reach", "size"))
      .reset_index(drop=True)
      .sort_values("reach_mean")
)

plt.figure()
plt.plot(reach_deciles["reach_mean"], reach_deciles["he_mean"], marker="o")
plt.xlabel("Mean reach (per decile)")
plt.ylabel("Mean high_effort_engagement")
plt.title("RQ2: High-Effort Engagement vs Reach (decile means)")
plt.tight_layout()

plt.figure()
plt.plot(reach_deciles["reach_mean"], reach_deciles["he_log_mean"], marker="o")
plt.xlabel("Mean reach (per decile)")
plt.ylabel("Mean log1p(high_effort_engagement)")
plt.title("RQ2: log High-Effort Engagement vs Reach (decile means)")
plt.tight_layout()

# (J) Visibility efficiency vs reach (binned) — shows “conversion per reach” changing with scale
plot_binned_mean(
    x="log_reach",
    y="VE_log",
    bins=25,
    xlabel="log1p(reach)",
    ylabel="VE_log = log1p(high_effort) - log1p(reach)",
    title="RQ2: Visibility Efficiency vs Reach (binned means)"
)


# RQ3: Does high visibility reduce importance of persuasive power?
# (K) PP -> high-effort slopes by visibility regime (very_high_reach)
# We visualize by binning PP and comparing outcomes for tail vs non-tail.
tmp = df[["persuasive_power_index", "log_high_effort", "very_high_reach"]].dropna().copy()
tmp["pp_bin"] = pd.qcut(tmp["persuasive_power_index"], q=10, duplicates="drop")

g = (
    tmp.groupby(["pp_bin", "very_high_reach"], observed=True)
       .agg(pp_mean=("persuasive_power_index", "mean"),
            y_mean=("log_high_effort", "mean"),
            n=("log_high_effort", "size"))
       .reset_index()
)

plt.figure()
for flag in [0, 1]:
    sub = g[g["very_high_reach"] == flag].sort_values("pp_mean")
    plt.plot(sub["pp_mean"], sub["y_mean"], marker="o", label=f"very_high_reach={flag}")
plt.xlabel("Mean persuasive_power_index (per bin)")
plt.ylabel("Mean log1p(high_effort_engagement)")
plt.title("RQ3: PP → High-Effort by Visibility Regime (tail vs non-tail)")
plt.legend()
plt.tight_layout()

# (L) Interaction-style visualization: PP effect by reach tertiles
tmp2 = df[["persuasive_power_index", "log_high_effort", "log_reach"]].dropna().copy()
tmp2["reach_tertile"] = pd.qcut(tmp2["log_reach"], q=3, labels=["Low reach", "Mid reach", "High reach"])
tmp2["pp_bin"] = pd.qcut(tmp2["persuasive_power_index"], q=10, duplicates="drop")

g2 = (
    tmp2.groupby(["reach_tertile", "pp_bin"], observed=True)
        .agg(pp_mean=("persuasive_power_index", "mean"),
             y_mean=("log_high_effort", "mean"))
        .reset_index()
)

plt.figure()
for tier in ["Low reach", "Mid reach", "High reach"]:
    sub = g2[g2["reach_tertile"] == tier].sort_values("pp_mean")
    plt.plot(sub["pp_mean"], sub["y_mean"], marker="o", label=tier)
plt.xlabel("Mean persuasive_power_index (per bin)")
plt.ylabel("Mean log1p(high_effort_engagement)")
plt.title("RQ3: PP → High-Effort Across Reach Levels (tertiles)")
plt.legend()
plt.tight_layout()

# -----------------------------
# 6) Optional: correlation heatmap (numeric columns)
# -----------------------------
num_cols = [
    "persuasive_power_index",
    "algorithmic_amplification_index",
    "engagement_success_index",
    "reach", "impressions",
    "high_effort_engagement",
    "authority_log",
    "ETR_like", "VE_log"
]
corr = df[num_cols].corr(numeric_only=True)

plt.figure(figsize=(9, 7))
plt.imshow(corr.values, aspect="auto")
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Correlation Heatmap (selected numeric features)")
plt.colorbar()
plt.tight_layout()

plt.show()