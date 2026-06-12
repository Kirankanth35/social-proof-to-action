from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import argparse
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = "dataset_algorithmic_persuasion_10000.xlsx"
DEFAULT_SHEET = "data"
DEFAULT_OUTPUT = "dataset_feature_engineered.csv"


# ============================================================
# Helpers
# ============================================================

def normalize_name(name: str) -> str:
    """
    Normalize column names so matching works even if the dataset uses
    spaces, capitals, hyphens, or slightly different punctuation.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("__", "_")
        .replace(" ", "_")
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


def safe_log1p(series: pd.Series) -> pd.Series:
    """
    Log transform with protection for negative values.
    Negative values are clipped to 0 because counts should not be negative.
    """
    s = safe_numeric(series).fillna(0)
    s = s.clip(lower=0)
    return np.log1p(s)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safe division that avoids divide-by-zero and returns NaN
    when denominator is 0 or missing.
    """
    num = safe_numeric(numerator)
    den = safe_numeric(denominator)
    den = den.replace(0, np.nan)
    return num / den


def zscore(series: pd.Series) -> pd.Series:
    """
    Standardize safely. If variance is zero, return zeros.
    """
    s = safe_numeric(series)
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
    return (s - mean) / std


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """
    Optional winsorization for extreme ratios/efficiency variables.
    """
    s = safe_numeric(series)
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def average_available(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Row-wise average of available columns. Requires at least one non-missing value.
    """
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[valid_cols].mean(axis=1, skipna=True)


# ============================================================
# Column matching
# ============================================================

CANONICAL_CANDIDATES: Dict[str, List[str]] = {
    # time / ids
    "post_datetime": [
        "post_datetime", "post_date", "datetime", "date", "created_at", "timestamp"
    ],
    "influencer_id": [
        "influencer_id", "creator_id", "account_id", "user_id", "influencer"
    ],

    # totals
    "likes": ["likes", "total_likes", "like_count"],
    "comments": ["comments", "total_comments", "comment_count"],
    "shares": ["shares", "total_shares", "share_count"],
    "reach": ["reach", "post_reach"],
    "impressions": ["impressions", "post_impressions"],
    "saves": ["saves", "saved", "save_count"],
    "profile_visits": ["profile_visits", "profile_visit", "visits", "profile_clicks"],
    "link_clicks": ["link_clicks", "clicks", "website_clicks", "url_clicks"],
    "followers": ["followers", "follower_count", "num_followers", "audience_size"],

    # early metrics
    "early_likes": ["early_likes", "first_hour_likes", "hour1_likes", "likes_1h"],
    "early_comments": ["early_comments", "first_hour_comments", "hour1_comments", "comments_1h"],
    "early_shares": ["early_shares", "first_hour_shares", "hour1_shares", "shares_1h"],

    # pre-existing modeled/derived
    "persuasive_power_index": ["persuasive_power_index", "persuasive_power"],
    "authority_log": ["authority_log"],
    "high_effort_engagement": ["high_effort_engagement"],
    "engagement_success_index": ["engagement_success_index", "success_index"],

    # controls / categories
    "content_type": ["content_type", "post_type"],
    "post_format": ["post_format", "format", "media_type"],
    "verified": ["verified", "is_verified"],
    "account_age_years": ["account_age_years", "account_age", "years_active"],
}


def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return a mapping from canonical column names to actual dataset column names.
    """
    normalized_actual = {normalize_name(c): c for c in df.columns}
    found: Dict[str, str] = {}

    for canonical, candidates in CANONICAL_CANDIDATES.items():
        for candidate in candidates:
            candidate_norm = normalize_name(candidate)
            if candidate_norm in normalized_actual:
                found[canonical] = normalized_actual[candidate_norm]
                break

    return found


def rename_to_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Rename discovered columns to canonical names.
    """
    col_map = build_column_map(df)
    rename_map = {actual: canonical for canonical, actual in col_map.items()}
    df = df.rename(columns=rename_map)
    return df, col_map


# ============================================================
# Loading
# ============================================================

def load_dataset(path: Path, sheet_name: str = DEFAULT_SHEET) -> pd.DataFrame:
    """
    Load CSV or Excel. Prefer sheet_name='data' for your workbook.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except ValueError:
            warnings.warn(
                f"Sheet '{sheet_name}' not found. Loading the first sheet instead.",
                stacklevel=2,
            )
            return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


# ============================================================
# Feature engineering
# ============================================================

def engineer_features(
    df: pd.DataFrame,
    winsorize_ratios: bool = False,
    ratio_lower_q: float = 0.01,
    ratio_upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    """
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    df, discovered = rename_to_canonical(df)

    # ------------------------------
    # Required minimum fields
    # ------------------------------
    if "reach" not in df.columns:
        raise ValueError(
            "A reach column is required for this project. "
            "Please make sure your dataset has a column like 'reach' or 'post_reach'."
        )

    # ------------------------------
    # Parse datetime and time features
    # ------------------------------
    if "post_datetime" in df.columns:
        df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")
        df["post_year"] = df["post_datetime"].dt.year
        df["post_month"] = df["post_datetime"].dt.month
        df["post_day"] = df["post_datetime"].dt.day
        df["post_hour"] = df["post_datetime"].dt.hour
        df["year_month"] = df["post_datetime"].dt.to_period("M").astype("string")
    else:
        warnings.warn("No post_datetime column found. Time features were not created.", stacklevel=2)

    # ------------------------------
    # Standardize numerics
    # ------------------------------
    numeric_candidates = [
        "likes", "comments", "shares", "reach", "impressions", "saves",
        "profile_visits", "link_clicks", "followers",
        "early_likes", "early_comments", "early_shares",
        "persuasive_power_index", "authority_log", "high_effort_engagement",
        "engagement_success_index", "account_age_years"
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # ------------------------------
    # Standardize categories / booleans
    # ------------------------------
    for col in ["content_type", "post_format"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().astype("category")

    if "verified" in df.columns:
        if df["verified"].dtype == "object" or pd.api.types.is_string_dtype(df["verified"]):
            df["verified"] = (
                df["verified"]
                .astype("string")
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0})
            )
        df["verified"] = safe_numeric(df["verified"]).fillna(0).astype("int64")

    # ------------------------------
    # Core totals
    # ------------------------------
    engagement_total_components = [c for c in ["likes", "comments", "shares"] if c in df.columns]
    if engagement_total_components:
        df["total_basic_engagement"] = df[engagement_total_components].fillna(0).sum(axis=1)

    early_components = [c for c in ["early_likes", "early_comments", "early_shares"] if c in df.columns]
    if early_components:
        df["early_engagement_total"] = df[early_components].fillna(0).sum(axis=1)

    # ------------------------------
    # High-effort engagement
    # ------------------------------
    if "high_effort_engagement" not in df.columns:
        high_effort_parts = [c for c in ["link_clicks", "saves", "profile_visits"] if c in df.columns]
        if high_effort_parts:
            df["high_effort_engagement"] = df[high_effort_parts].fillna(0).sum(axis=1)
        else:
            df["high_effort_engagement"] = np.nan
            warnings.warn(
                "No link_clicks / saves / profile_visits columns found. "
                "high_effort_engagement could not be created.",
                stacklevel=2,
            )

    # ------------------------------
    # Log transforms for heavy-tailed counts
    # ------------------------------
    heavy_tail_vars = [
        "likes", "comments", "shares", "reach", "impressions", "saves",
        "profile_visits", "link_clicks", "followers",
        "early_likes", "early_comments", "early_shares",
        "total_basic_engagement", "early_engagement_total",
        "high_effort_engagement",
    ]

    for col in heavy_tail_vars:
        if col in df.columns:
            df[f"log_{col}"] = safe_log1p(df[col])

    # ------------------------------
    # Authority
    # ------------------------------
    if "authority_log" not in df.columns:
        if "followers" in df.columns:
            df["authority_log"] = safe_log1p(df["followers"])
        else:
            df["authority_log"] = np.nan
            warnings.warn(
                "No followers column found. authority_log could not be created.",
                stacklevel=2,
            )

    if "authority_log" in df.columns:
        df["authority_z"] = zscore(df["authority_log"])
        try:
            df["authority_tier"] = pd.qcut(
                df["authority_log"],
                q=3,
                labels=["Low", "Mid", "High"],
                duplicates="drop",
            )
        except ValueError:
            df["authority_tier"] = pd.Series(["Mid"] * len(df), index=df.index, dtype="category")

    # ------------------------------
    # Early-to-total ratios
    # Important quirk from your project:
    # early_comments / total_comments and early_shares / total_shares
    # are not reliable in this synthetic dataset.
    # Use early_like_ratio only.
    # ------------------------------
    if "early_likes" in df.columns and "likes" in df.columns:
        df["early_like_ratio"] = safe_divide(df["early_likes"], df["likes"])

    # ------------------------------
    # Persuasive power index
    # If already present, keep it and also standardize.
    # Otherwise build it from available early measures.
    # ------------------------------
    if "persuasive_power_index" in df.columns:
        df["persuasive_power_index_z"] = zscore(df["persuasive_power_index"])
    else:
        pp_parts: List[str] = []

        if "log_early_likes" in df.columns:
            df["pp_early_likes_z"] = zscore(df["log_early_likes"])
            pp_parts.append("pp_early_likes_z")

        if "log_early_comments" in df.columns:
            df["pp_early_comments_z"] = zscore(df["log_early_comments"])
            pp_parts.append("pp_early_comments_z")

        if "log_early_shares" in df.columns:
            df["pp_early_shares_z"] = zscore(df["log_early_shares"])
            pp_parts.append("pp_early_shares_z")

        if "early_like_ratio" in df.columns:
            ratio = df["early_like_ratio"]
            if winsorize_ratios:
                ratio = winsorize_series(ratio, ratio_lower_q, ratio_upper_q)
            df["pp_early_like_ratio_z"] = zscore(ratio)
            pp_parts.append("pp_early_like_ratio_z")

        # Fallback if true early metrics are not available
        if not pp_parts:
            warnings.warn(
                "No first-hour variables found. Falling back to total engagement-based persuasive power.",
                stacklevel=2,
            )
            for fallback_col in ["log_likes", "log_comments", "log_shares"]:
                if fallback_col in df.columns:
                    z_col = f"{fallback_col}_z"
                    df[z_col] = zscore(df[fallback_col])
                    pp_parts.append(z_col)

        df["persuasive_power_index"] = average_available(df, pp_parts)
        df["persuasive_power_index_z"] = zscore(df["persuasive_power_index"])

    # ------------------------------
    # Amplification index
    # ------------------------------
    amp_parts: List[str] = []

    if "log_reach" in df.columns:
        df["amp_reach_z"] = zscore(df["log_reach"])
        amp_parts.append("amp_reach_z")

    if "log_impressions" in df.columns:
        df["amp_impressions_z"] = zscore(df["log_impressions"])
        amp_parts.append("amp_impressions_z")

    if amp_parts:
        df["amplification_index"] = average_available(df, amp_parts)
    else:
        df["amplification_index"] = np.nan
        warnings.warn(
            "No reach/impressions logs available to compute amplification_index.",
            stacklevel=2,
        )

    # ------------------------------
    # Engagement success index
    # ------------------------------
    if "engagement_success_index" not in df.columns:
        es_parts: List[str] = []

        for base in ["link_clicks", "saves", "profile_visits", "high_effort_engagement"]:
            log_col = f"log_{base}"
            if log_col in df.columns:
                z_col = f"{log_col}_z"
                df[z_col] = zscore(df[log_col])
                es_parts.append(z_col)

        if es_parts:
            df["engagement_success_index"] = average_available(df, es_parts)
        else:
            df["engagement_success_index"] = np.nan
            warnings.warn(
                "No high-effort components found to compute engagement_success_index.",
                stacklevel=2,
            )

    # ------------------------------
    # Efficiency variables
    # ------------------------------
    if "high_effort_engagement" in df.columns and "reach" in df.columns:
        df["reach_efficiency"] = safe_divide(df["high_effort_engagement"], df["reach"])

    if "high_effort_engagement" in df.columns and "impressions" in df.columns:
        df["impression_efficiency"] = safe_divide(df["high_effort_engagement"], df["impressions"])

    if winsorize_ratios:
        for col in ["reach_efficiency", "impression_efficiency"]:
            if col in df.columns:
                df[col] = winsorize_series(df[col], ratio_lower_q, ratio_upper_q)

    # ------------------------------
    # Reach saturation and high-visibility regime
    # ------------------------------
    if "reach" in df.columns:
        df["reach_sq"] = safe_numeric(df["reach"]).pow(2)

        if "log_reach" in df.columns:
            df["log_reach_sq"] = df["log_reach"].pow(2)

        p95 = df["reach"].quantile(0.95)
        df["very_high_reach"] = (df["reach"] >= p95).astype("int64")
        df["high_visibility_regime"] = df["very_high_reach"]

    # ------------------------------
    # Useful interaction-ready variables
    # ------------------------------
    if "persuasive_power_index_z" in df.columns and "authority_z" in df.columns:
        df["pp_x_authority"] = df["persuasive_power_index_z"] * df["authority_z"]

    if "amplification_index" in df.columns and "authority_z" in df.columns:
        df["amplification_x_authority"] = df["amplification_index"] * df["authority_z"]

    if "persuasive_power_index_z" in df.columns and "very_high_reach" in df.columns:
        df["pp_x_very_high_reach"] = df["persuasive_power_index_z"] * df["very_high_reach"]

    # ------------------------------
    # Missing-data summary flags (useful for diagnostics)
    # ------------------------------
    df["missing_any_core"] = 0
    core_cols = [
        c for c in [
            "persuasive_power_index", "reach", "high_effort_engagement",
            "authority_log", "engagement_success_index"
        ] if c in df.columns
    ]
    if core_cols:
        df["missing_any_core"] = df[core_cols].isna().any(axis=1).astype("int64")

    # ------------------------------
    # Final cleanup
    # ------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# Quality checks / reporting
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    """
    Print a compact summary of what was created.
    """
    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    key_cols = [
        "persuasive_power_index",
        "amplification_index",
        "high_effort_engagement",
        "engagement_success_index",
        "authority_log",
        "very_high_reach",
        "reach_efficiency",
    ]

    present = [c for c in key_cols if c in df.columns]
    if present:
        print("\nKey engineered columns found:")
        for c in present:
            print(f"  - {c}")

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        print("\nKey engineered columns NOT found:")
        for c in missing:
            print(f"  - {c}")

    if "very_high_reach" in df.columns:
        pct = df["very_high_reach"].mean() * 100
        print(f"\nHigh-visibility regime share: {pct:.2f}%")

    if "authority_tier" in df.columns:
        print("\nAuthority tiers:")
        print(df["authority_tier"].value_counts(dropna=False))


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering for the algorithmic amplification project.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to dataset (.xlsx/.xls/.csv)")
    parser.add_argument("--sheet", type=str, default=DEFAULT_SHEET, help="Excel sheet name")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output file path (.csv or .xlsx)")
    parser.add_argument(
        "--winsorize-ratios",
        action="store_true",
        help="Winsorize ratio/efficiency variables at 1% and 99%",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df_raw = load_dataset(input_path, sheet_name=args.sheet)
    df_feat = engineer_features(
        df_raw,
        winsorize_ratios=args.winsorize_ratios,
        ratio_lower_q=0.01,
        ratio_upper_q=0.99,
    )

    # Save
    if output_path.suffix.lower() == ".xlsx":
        df_feat.to_excel(output_path, index=False)
    else:
        df_feat.to_csv(output_path, index=False)

    print_summary(df_feat)
    print(f"\nSaved feature-engineered dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    main()