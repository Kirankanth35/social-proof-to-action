# outliers.py
# Find outliers in your dataset (univariate + multivariate) and export results to Excel.

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = Path(__file__).resolve().parent / "dataset_algorithmic_persuasion_10000.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="data")
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

# -----------------------------
# Choose variables to check (edit as you like)
# -----------------------------
num_cols = [
    "reach", "impressions",
    "early_likes", "early_comments", "early_shares",
    "high_effort_engagement", "link_clicks", "saves", "profile_visits",
    "total_likes", "total_comments", "total_shares",
    "persuasive_power_index", "algorithmic_amplification_index", "engagement_success_index",
    "authority_log", "follower_count"
]

# Keep only those that exist (safe if columns change)
num_cols = [c for c in num_cols if c in df.columns]

# -----------------------------
# Helpers
# -----------------------------
def iqr_outliers(series: pd.Series, k: float = 1.5):
    """Return boolean mask for IQR outliers."""
    s = series.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    mask = (series < lo) | (series > hi)
    return mask, lo, hi

def zscore_outliers(series: pd.Series, z_thresh: float = 3.0):
    """Return boolean mask for z-score outliers."""
    s = series.astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        mask = pd.Series(False, index=series.index)
        return mask, mu, sigma
    z = (s - mu) / sigma
    mask = z.abs() > z_thresh
    return mask, mu, sigma

def mad_outliers(series: pd.Series, thresh: float = 3.5):
    """
    Robust outliers using modified z-score (MAD method).
    Common threshold: 3.5
    """
    s = series.astype(float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        mask = pd.Series(False, index=series.index)
        return mask, med, mad
    mz = 0.6745 * (s - med) / mad
    mask = np.abs(mz) > thresh
    return mask, med, mad

# -----------------------------
# 1) Univariate outliers per variable
# -----------------------------
outlier_rows = []

for col in num_cols:
    # IQR
    iqr_mask, lo, hi = iqr_outliers(df[col], k=1.5)
    iqr_idx = df.index[iqr_mask.fillna(False)].tolist()

    # Z-score
    z_mask, mu, sigma = zscore_outliers(df[col], z_thresh=3.0)
    z_idx = df.index[z_mask.fillna(False)].tolist()

    # MAD
    mad_mask, med, mad = mad_outliers(df[col], thresh=3.5)
    mad_idx = df.index[mad_mask.fillna(False)].tolist()

    outlier_rows.append({
        "column": col,
        "iqr_outliers_count": len(iqr_idx),
        "iqr_lower": lo,
        "iqr_upper": hi,
        "z_outliers_count": len(z_idx),
        "z_mean": mu,
        "z_std": sigma,
        "mad_outliers_count": len(mad_idx),
        "mad_median": med,
        "mad_value": mad,
    })

outlier_summary = pd.DataFrame(outlier_rows).sort_values(
    ["iqr_outliers_count", "z_outliers_count", "mad_outliers_count"],
    ascending=False
)

# -----------------------------
# 2) Top outlier posts per key metrics (simple "most extreme" list)
# -----------------------------
key_metrics = ["reach", "impressions", "high_effort_engagement", "persuasive_power_index",
               "algorithmic_amplification_index", "engagement_success_index"]
key_metrics = [c for c in key_metrics if c in df.columns]

top_posts = {}
for col in key_metrics:
    # Top 20 highest
    top_posts[f"top20_{col}"] = df.sort_values(col, ascending=False).head(20)[
        ["post_id", "influencer_id", "post_datetime", col]
    ]

# -----------------------------
# 3) Multivariate outliers (Mahalanobis distance) on selected features
# -----------------------------
# Choose a small set of key features for multivariate outliers:
mv_cols = [c for c in [
    "persuasive_power_index",
    "algorithmic_amplification_index",
    "engagement_success_index",
    "reach",
    "high_effort_engagement",
    "authority_log"
] if c in df.columns]

mv = df[mv_cols].dropna().copy()

# Log-transform big counts for stability
if "reach" in mv.columns:
    mv["reach"] = np.log1p(mv["reach"])
if "high_effort_engagement" in mv.columns:
    mv["high_effort_engagement"] = np.log1p(mv["high_effort_engagement"])

# Standardize
mv_z = (mv - mv.mean()) / mv.std(ddof=0)

# Mahalanobis distance
X = mv_z.values
cov = np.cov(X, rowvar=False)
inv_cov = np.linalg.pinv(cov)
d2 = np.einsum("ij,jk,ik->i", X, inv_cov, X)  # squared distance

mv["mahalanobis_d2"] = d2
# Flag top 1% as multivariate outliers (you can change)
threshold = np.quantile(d2, 0.99)
mv["mv_outlier_99p"] = mv["mahalanobis_d2"] >= threshold

mv_outliers = mv[mv["mv_outlier_99p"]].copy()
mv_outliers = mv_outliers.merge(
    df[["post_id", "influencer_id", "post_datetime"]],
    left_index=True, right_index=True, how="left"
).sort_values("mahalanobis_d2", ascending=False)

# -----------------------------
# 4) Save results to Excel for easy review
# -----------------------------
OUT_PATH = Path(__file__).resolve().parent / "outliers_report.xlsx"

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    outlier_summary.to_excel(writer, sheet_name="summary", index=False)

    # Top posts per metric
    for name, tdf in top_posts.items():
        tdf.to_excel(writer, sheet_name=name[:31], index=False)  # Excel sheet name max=31 chars

    # Multivariate outliers
    mv_outliers.to_excel(writer, sheet_name="multivariate_outliers", index=False)

print(f"Saved outlier report to: {OUT_PATH}")
print("\nQuick view (top 10 columns by IQR outliers):")
print(outlier_summary.head(10).to_string(index=False))