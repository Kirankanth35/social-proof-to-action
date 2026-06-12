from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from matplotlib.ticker import MaxNLocator


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = "dataset_feature_engineered.csv"
DEFAULT_OUTPUT_XLSX = "descriptive_statistics_output.xlsx"
DEFAULT_OUTPUT_DIR = "descriptive_statistics_outputs"


# ============================================================
# Helpers
# ============================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        for c in df.columns
    ]
    return df


def choose_existing(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def save_plot(series: pd.Series, title: str, outpath: Path, bins: int = 30) -> None:
    s = safe_numeric(series).dropna()
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(series: pd.Series, title: str, outpath: Path) -> None:
    s = safe_numeric(series).dropna()
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot(s, vert=False)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_detailed_distribution_plot(series: pd.Series, title: str, outpath: Path, bins: int = 30) -> None:
    s = safe_numeric(series).dropna()
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(series.name, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.axvline(s.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {s.mean():.2f}')
    ax.axvline(s.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {s.median():.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index.name = "variable"
    return df.reset_index()


# ============================================================
# Core descriptive statistics
# ============================================================

def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows, cols = df.shape

    overview = {
        "n_rows": [rows],
        "n_columns": [cols],
        "n_duplicate_rows": [int(df.duplicated().sum())],
        "n_missing_cells": [int(df.isna().sum().sum())],
        "percent_missing_cells": [round(df.isna().sum().sum() / (rows * cols) * 100, 4)],
    }

    if "influencer_id" in df.columns:
        overview["n_unique_influencers"] = [int(df["influencer_id"].nunique(dropna=True))]
        overview["avg_posts_per_influencer"] = [
            round(rows / max(df["influencer_id"].nunique(dropna=True), 1), 4)
        ]

    if "post_datetime" in df.columns:
        dt = pd.to_datetime(df["post_datetime"], errors="coerce")
        overview["min_post_datetime"] = [dt.min()]
        overview["max_post_datetime"] = [dt.max()]

    return pd.DataFrame(overview)


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    out = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_pct.round(4),
        "dtype": df.dtypes.astype(str)
    })

    out = out.sort_values(["missing_count", "missing_percent"], ascending=False)
    return format_index(out)


def numeric_descriptive_table(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()

    stats = pd.DataFrame(index=numeric_cols)

    for col in numeric_cols:
        s = safe_numeric(df[col])

        stats.loc[col, "n_non_missing"] = int(s.notna().sum())
        stats.loc[col, "n_missing"] = int(s.isna().sum())
        stats.loc[col, "mean"] = s.mean()
        stats.loc[col, "std"] = s.std()
        stats.loc[col, "min"] = s.min()
        stats.loc[col, "p1"] = s.quantile(0.01)
        stats.loc[col, "p5"] = s.quantile(0.05)
        stats.loc[col, "p25"] = s.quantile(0.25)
        stats.loc[col, "median"] = s.median()
        stats.loc[col, "p75"] = s.quantile(0.75)
        stats.loc[col, "p95"] = s.quantile(0.95)
        stats.loc[col, "p99"] = s.quantile(0.99)
        stats.loc[col, "max"] = s.max()
        stats.loc[col, "skewness"] = s.skew()
        stats.loc[col, "kurtosis"] = s.kurt()
        stats.loc[col, "n_zeros"] = int((s.fillna(np.nan) == 0).sum())

    return format_index(stats.round(6))


def categorical_frequency_tables(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    for col in categorical_cols:
        freq = (
            df[col]
            .astype("string")
            .fillna("Missing")
            .value_counts(dropna=False)
            .rename_axis(col)
            .reset_index(name="count")
        )
        freq["percent"] = (freq["count"] / len(df) * 100).round(4)
        tables[col] = freq

    return tables


def correlation_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame()

    corr = df[cols].apply(safe_numeric).corr(method="pearson")
    return corr.round(4)


def grouped_numeric_stats(df: pd.DataFrame, group_col: str, vars_to_summarize: List[str]) -> pd.DataFrame:
    if group_col not in df.columns or not vars_to_summarize:
        return pd.DataFrame()

    valid = choose_existing(df, vars_to_summarize)
    if not valid:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_col, dropna=False)[valid]
        .agg(["count", "mean", "std", "median"])
    )

    grouped.columns = [f"{var}_{stat}" for var, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped.round(6)


def create_key_variable_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Tailored to your project.
    """
    core_numeric = choose_existing(df, [
        "persuasive_power_index",
        "persuasive_power_index_z",
        "amplification_index",
        "reach",
        "log_reach",
        "impressions",
        "log_impressions",
        "high_effort_engagement",
        "log_high_effort_engagement",
        "engagement_success_index",
        "authority_log",
        "authority_z",
        "reach_efficiency",
        "impression_efficiency",
        "likes",
        "comments",
        "shares",
        "saves",
        "profile_visits",
        "link_clicks",
        "early_likes",
        "early_comments",
        "early_shares",
        "early_engagement_total",
        "early_like_ratio",
        "very_high_reach",
    ])

    model_numeric = choose_existing(df, [
        "persuasive_power_index",
        "amplification_index",
        "high_effort_engagement",
        "engagement_success_index",
        "authority_log",
        "reach",
        "log_reach",
        "log_reach_sq",
        "very_high_reach",
        "reach_efficiency",
    ])

    categorical_cols = choose_existing(df, [
        "authority_tier",
        "very_high_reach",
        "content_type",
        "post_format",
        "verified",
        "year_month",
    ])

    plot_cols = choose_existing(df, [
        "persuasive_power_index",
        "amplification_index",
        "reach",
        "log_reach",
        "high_effort_engagement",
        "log_high_effort_engagement",
        "engagement_success_index",
        "authority_log",
        "reach_efficiency",
    ])

    return {
        "core_numeric": core_numeric,
        "model_numeric": model_numeric,
        "categorical_cols": categorical_cols,
        "plot_cols": plot_cols,
    }


# ============================================================
# Main runner
# ============================================================

def run_descriptive_statistics(input_path: Path, output_xlsx: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input must be .csv, .xlsx, or .xls")

    df = normalize_colnames(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    variable_lists = create_key_variable_lists(df)

    # ------------------------------
    # Tables
    # ------------------------------
    overview_df = dataset_overview(df)
    missing_df = missing_values_table(df)
    numeric_stats_df = numeric_descriptive_table(df, variable_lists["core_numeric"])
    corr_df = correlation_table(df, variable_lists["model_numeric"])

    authority_group_df = grouped_numeric_stats(
        df,
        group_col="authority_tier",
        vars_to_summarize=variable_lists["model_numeric"],
    )

    visibility_group_df = grouped_numeric_stats(
        df,
        group_col="very_high_reach",
        vars_to_summarize=variable_lists["model_numeric"],
    )

    categorical_tables = categorical_frequency_tables(df, variable_lists["categorical_cols"])

    # ------------------------------
    # Plots
    # ------------------------------
    for col in variable_lists["plot_cols"]:
        save_detailed_distribution_plot(
        df[col],
        title=f"Detailed Distribution of {col}",
        outpath=plots_dir / f"{col}_detailed_distribution.png"
    )

    # ------------------------------
    # Save workbook
    # ------------------------------
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="dataset_overview", index=False)
        missing_df.to_excel(writer, sheet_name="missing_values", index=False)
        numeric_stats_df.to_excel(writer, sheet_name="numeric_descriptives", index=False)

        if not corr_df.empty:
            corr_df.to_excel(writer, sheet_name="correlations")

        if not authority_group_df.empty:
            authority_group_df.to_excel(writer, sheet_name="grouped_authority", index=False)

        if not visibility_group_df.empty:
            visibility_group_df.to_excel(writer, sheet_name="grouped_visibility", index=False)

        for sheet_name, freq_df in categorical_tables.items():
            safe_sheet = f"freq_{sheet_name}"[:31]
            freq_df.to_excel(writer, sheet_name=safe_sheet, index=False)

    # ------------------------------
    # Save plain-text summary
    # ------------------------------
    summary_path = output_dir / "descriptive_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== DESCRIPTIVE STATISTICS SUMMARY ===\n\n")
        f.write(overview_df.to_string(index=False))
        f.write("\n\n=== KEY NUMERIC VARIABLES ===\n\n")
        if not numeric_stats_df.empty:
            f.write(numeric_stats_df.to_string(index=False))
        else:
            f.write("No numeric descriptive statistics were produced.\n")

        f.write("\n\n=== CORRELATIONS ===\n\n")
        if not corr_df.empty:
            f.write(corr_df.to_string())
        else:
            f.write("Not enough variables for a correlation matrix.\n")

    # ------------------------------
    # Console output
    # ------------------------------
    print("\n=== DESCRIPTIVE STATISTICS COMPLETED ===")
    print(f"Input file: {input_path.resolve()}")
    print(f"Workbook saved to: {output_xlsx.resolve()}")
    print(f"Summary text saved to: {summary_path.resolve()}")
    print(f"Plots saved to: {plots_dir.resolve()}")

    print("\nCore numeric variables summarized:")
    for col in variable_lists["core_numeric"]:
        print(f"  - {col}")

    print("\nCategorical variables summarized:")
    for col in variable_lists["categorical_cols"]:
        print(f"  - {col}")

    if "very_high_reach" in df.columns:
        pct = safe_numeric(df["very_high_reach"]).mean() * 100
        print(f"\nHigh-visibility regime share: {pct:.2f}%")

    if "authority_tier" in df.columns:
        print("\nAuthority tier counts:")
        print(df["authority_tier"].value_counts(dropna=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptive statistics for the feature-engineered project dataset.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to feature-engineered dataset")
    parser.add_argument("--output-xlsx", type=str, default=DEFAULT_OUTPUT_XLSX, help="Excel workbook output path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Folder for text summary and plots")
    args = parser.parse_args()

    run_descriptive_statistics(
        input_path=Path(args.input),
        output_xlsx=Path(args.output_xlsx),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()