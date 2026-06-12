from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = "dataset_feature_engineered.csv"
DEFAULT_OUTPUT_DIR = "rq1_outputs"

DEFAULT_BOOTSTRAPS = 1000
DEFAULT_RANDOM_SEED = 42


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


def zscore(series: pd.Series) -> pd.Series:
    s = safe_numeric(series)
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
    return (s - mean) / std


def choose_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_coef(result, candidates: List[str]) -> float:
    for c in candidates:
        if c in result.params.index:
            return float(result.params[c])
    return 0.0


def get_se(result, candidates: List[str]) -> float:
    for c in candidates:
        if c in result.bse.index:
            return float(result.bse[c])
    return np.nan


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build missing core columns if needed.
    """
    df = df.copy()

    if "persuasive_power_index_z" not in df.columns:
        if "persuasive_power_index" in df.columns:
            df["persuasive_power_index_z"] = zscore(df["persuasive_power_index"])
        else:
            raise ValueError("persuasive_power_index_z or persuasive_power_index is required.")

    if "authority_z" not in df.columns:
        if "authority_log" in df.columns:
            df["authority_z"] = zscore(df["authority_log"])
        else:
            raise ValueError("authority_z or authority_log is required.")

    if "log_reach" not in df.columns:
        if "reach" in df.columns:
            df["log_reach"] = np.log1p(safe_numeric(df["reach"]).clip(lower=0))
        else:
            raise ValueError("log_reach or reach is required.")

    if "log_high_effort_engagement" not in df.columns:
        if "high_effort_engagement" in df.columns:
            df["log_high_effort_engagement"] = np.log1p(
                safe_numeric(df["high_effort_engagement"]).clip(lower=0)
            )
        else:
            warnings.warn(
                "log_high_effort_engagement and high_effort_engagement were not found. "
                "Will fall back to engagement_success_index if available.",
                stacklevel=2,
            )

    return df


def choose_core_variables(df: pd.DataFrame) -> Dict[str, str]:
    """
    Choose focal variables for the RQ1 models.
    """
    x = choose_first_existing(df, ["persuasive_power_index_z", "persuasive_power_index"])
    w = choose_first_existing(df, ["authority_z", "authority_log"])
    m = choose_first_existing(df, ["log_reach", "amplification_index", "reach"])
    y = choose_first_existing(
        df,
        ["log_high_effort_engagement", "high_effort_engagement", "engagement_success_index"]
    )
    cluster = choose_first_existing(df, ["influencer_id", "creator_id", "account_id", "user_id"])

    if x is None:
        raise ValueError("Could not find persuasive power variable.")
    if w is None:
        raise ValueError("Could not find authority variable.")
    if m is None:
        raise ValueError("Could not find amplification / mediator variable.")
    if y is None:
        raise ValueError("Could not find high-effort outcome variable.")
    if cluster is None:
        raise ValueError("Could not find influencer cluster ID column.")

    return {"x": x, "w": w, "m": m, "y": y, "cluster": cluster}


def prepare_analysis_df(df: pd.DataFrame, vars_map: Dict[str, str]) -> pd.DataFrame:
    """
    Keep only needed columns and drop rows missing focal variables.
    """
    df = df.copy()

    # Coerce key numerics
    for col in [vars_map["x"], vars_map["w"], vars_map["m"], vars_map["y"]]:
        df[col] = safe_numeric(df[col])

    # Candidate controls
    numeric_controls = [
        c for c in [
            "verified",
            "account_age_years",
            "followers",
            "likes",
            "comments",
            "shares",
            "saves",
            "profile_visits",
            "link_clicks",
            "early_likes",
            "early_comments",
            "early_shares",
            "early_like_ratio",
        ] if c in df.columns
    ]

    categorical_controls = [
        c for c in [
            "content_type",
            "post_format",
            "year_month",
            "post_month",
            "post_hour",
        ] if c in df.columns
    ]

    # Convert categories
    for c in categorical_controls:
        df[c] = df[c].astype("string")

    keep_cols = list(dict.fromkeys(
        [vars_map["cluster"], vars_map["x"], vars_map["w"], vars_map["m"], vars_map["y"]]
        + numeric_controls
        + categorical_controls
    ))

    analysis_df = df[keep_cols].copy()
    analysis_df = analysis_df.dropna(subset=[vars_map["cluster"], vars_map["x"], vars_map["w"], vars_map["m"], vars_map["y"]])

    # If outcome is raw counts, log it for the main model
    if vars_map["y"] == "high_effort_engagement":
        analysis_df["log_high_effort_engagement_model"] = np.log1p(
            safe_numeric(analysis_df["high_effort_engagement"]).clip(lower=0)
        )
        vars_map["y"] = "log_high_effort_engagement_model"

    # If mediator is raw reach, log it for the main model
    if vars_map["m"] == "reach":
        analysis_df["log_reach_model"] = np.log1p(
            safe_numeric(analysis_df["reach"]).clip(lower=0)
        )
        vars_map["m"] = "log_reach_model"

    return analysis_df


def build_formula_a(x: str, w: str, m: str, controls_numeric: List[str], controls_categorical: List[str]) -> str:
    rhs = [x, w, f"{x}:{w}"]
    rhs += controls_numeric
    rhs += [f"C({c})" for c in controls_categorical]
    return f"{m} ~ " + " + ".join(rhs)


def build_formula_b(x: str, w: str, m: str, y: str, controls_numeric: List[str], controls_categorical: List[str]) -> str:
    rhs = [m, x, w, f"{m}:{w}", f"{x}:{w}"]
    rhs += controls_numeric
    rhs += [f"C({c})" for c in controls_categorical]
    return f"{y} ~ " + " + ".join(rhs)


def fit_clustered_ols(formula: str, df: pd.DataFrame, cluster_col: str):
    """
    OLS with cluster-robust SE by influencer.
    """
    model = smf.ols(formula=formula, data=df)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col], "use_correction": True}
    )
    return result


def result_table(result, model_name: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "t": result.tvalues.values,
        "p_value": result.pvalues.values,
        "ci_low": result.conf_int().iloc[:, 0].values,
        "ci_high": result.conf_int().iloc[:, 1].values,
    })
    out["model"] = model_name
    out["r_squared"] = result.rsquared
    out["adj_r_squared"] = result.rsquared_adj
    out["n_obs"] = int(result.nobs)
    return out


def conditional_indirect_effects(a_result, b_result, x: str, w: str, m: str, authority_values: Dict[str, float]) -> pd.DataFrame:
    """
    Conditional indirect effect:
    (a1 + a3*w) * (b1 + b3*w)
    where:
      a1 = x -> m
      a3 = x:w in a-path
      b1 = m -> y
      b3 = m:w in b-path
    """
    a1 = get_coef(a_result, [x])
    a3 = get_coef(a_result, [f"{x}:{w}", f"{w}:{x}"])

    b1 = get_coef(b_result, [m])
    b3 = get_coef(b_result, [f"{m}:{w}", f"{w}:{m}"])

    rows = []
    for label, authority_value in authority_values.items():
        indirect = (a1 + a3 * authority_value) * (b1 + b3 * authority_value)
        rows.append({
            "authority_level": label,
            "authority_value": authority_value,
            "a_path_conditional": a1 + a3 * authority_value,
            "b_path_conditional": b1 + b3 * authority_value,
            "indirect_effect": indirect,
        })
    return pd.DataFrame(rows)


def cluster_bootstrap_conditional_indirect(
    df: pd.DataFrame,
    cluster_col: str,
    formula_a: str,
    formula_b: str,
    x: str,
    w: str,
    m: str,
    authority_values: Dict[str, float],
    n_boot: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap conditional indirect effects by resampling influencers (clusters).
    """
    rng = np.random.default_rng(random_seed)
    unique_clusters = df[cluster_col].dropna().unique()

    boot_records = []

    for i in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)

        sampled_parts = []
        for new_id, cid in enumerate(sampled_clusters):
            part = df[df[cluster_col] == cid].copy()
            # new bootstrap cluster id to preserve within-cluster structure
            part["_boot_cluster_id"] = f"boot_{i}_{new_id}"
            sampled_parts.append(part)

        boot_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

        try:
            a_fit = smf.ols(formula=formula_a, data=boot_df).fit()
            b_fit = smf.ols(formula=formula_b, data=boot_df).fit()

            a1 = get_coef(a_fit, [x])
            a3 = get_coef(a_fit, [f"{x}:{w}", f"{w}:{x}"])
            b1 = get_coef(b_fit, [m])
            b3 = get_coef(b_fit, [f"{m}:{w}", f"{w}:{m}"])

            row = {"bootstrap_iter": i}
            for label, authority_value in authority_values.items():
                row[f"indirect_{label}"] = (a1 + a3 * authority_value) * (b1 + b3 * authority_value)
            boot_records.append(row)

        except Exception:
            continue

    boot_df = pd.DataFrame(boot_records)
    if boot_df.empty:
        raise RuntimeError("Bootstrap failed: no successful bootstrap iterations.")

    summary_rows = []
    for label, authority_value in authority_values.items():
        col = f"indirect_{label}"
        vals = boot_df[col].dropna()
        summary_rows.append({
            "authority_level": label,
            "authority_value": authority_value,
            "bootstrap_n": int(vals.shape[0]),
            "bootstrap_mean": vals.mean(),
            "bootstrap_std": vals.std(),
            "ci_2_5": vals.quantile(0.025),
            "ci_50": vals.quantile(0.50),
            "ci_97_5": vals.quantile(0.975),
        })

    return pd.DataFrame(summary_rows), boot_df


def make_prediction_frame_a(df: pd.DataFrame, x: str, w: str, controls_numeric: List[str], controls_categorical: List[str]) -> pd.DataFrame:
    """
    Prediction grid for the a-path moderation plot.
    """
    x_min = df[x].quantile(0.05)
    x_max = df[x].quantile(0.95)
    x_grid = np.linspace(x_min, x_max, 100)

    authority_levels = {
        "Low Authority (-1 SD)": -1.0,
        "Mean Authority (0 SD)": 0.0,
        "High Authority (+1 SD)": 1.0,
    }

    base_numeric = {}
    for c in controls_numeric:
        base_numeric[c] = safe_numeric(df[c]).median()

    base_categorical = {}
    for c in controls_categorical:
        mode_series = df[c].mode(dropna=True)
        base_categorical[c] = mode_series.iloc[0] if not mode_series.empty else "Missing"

    rows = []
    for label, w_val in authority_levels.items():
        for xv in x_grid:
            row = {x: xv, w: w_val, "_authority_label": label}
            row.update(base_numeric)
            row.update(base_categorical)
            rows.append(row)

    return pd.DataFrame(rows)


def make_prediction_frame_b(df: pd.DataFrame, m: str, x: str, w: str, controls_numeric: List[str], controls_categorical: List[str]) -> pd.DataFrame:
    """
    Prediction grid for the b-path moderation plot.
    """
    m_min = df[m].quantile(0.05)
    m_max = df[m].quantile(0.95)
    m_grid = np.linspace(m_min, m_max, 100)

    authority_levels = {
        "Low Authority (-1 SD)": -1.0,
        "Mean Authority (0 SD)": 0.0,
        "High Authority (+1 SD)": 1.0,
    }

    base_numeric = {}
    for c in controls_numeric:
        base_numeric[c] = safe_numeric(df[c]).median()

    base_categorical = {}
    for c in controls_categorical:
        mode_series = df[c].mode(dropna=True)
        base_categorical[c] = mode_series.iloc[0] if not mode_series.empty else "Missing"

    x_mean = safe_numeric(df[x]).mean()

    rows = []
    for label, w_val in authority_levels.items():
        for mv in m_grid:
            row = {m: mv, x: x_mean, w: w_val, "_authority_label": label}
            row.update(base_numeric)
            row.update(base_categorical)
            rows.append(row)

    return pd.DataFrame(rows)


def save_moderation_plot(pred_df: pd.DataFrame, x_col: str, yhat_col: str, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, g in pred_df.groupby("_authority_label"):
        ax.plot(g[x_col], g[yhat_col], label=label)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Predicted value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main RQ1 runner
# ============================================================

def run_rq1(
    input_path: Path,
    output_dir: Path,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load data
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input must be .csv, .xlsx, or .xls")

    df = normalize_colnames(df)
    df = ensure_required_columns(df)

    vars_map = choose_core_variables(df)
    df = prepare_analysis_df(df, vars_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = vars_map["x"]
    w = vars_map["w"]
    m = vars_map["m"]
    y = vars_map["y"]
    cluster = vars_map["cluster"]

    numeric_controls = [
        c for c in [
            "verified",
            "account_age_years",
        ] if c in df.columns
    ]

    categorical_controls = [
        c for c in [
            "content_type",
            "post_format",
            "year_month",
        ] if c in df.columns
    ]

    formula_a = build_formula_a(x, w, m, numeric_controls, categorical_controls)
    formula_b = build_formula_b(x, w, m, y, numeric_controls, categorical_controls)

    print("\n=== RQ1 MODEL SETUP ===")
    print(f"X (persuasive power): {x}")
    print(f"W (authority): {w}")
    print(f"M (amplification): {m}")
    print(f"Y (high-effort outcome): {y}")
    print(f"Cluster ID: {cluster}")
    print(f"n = {len(df):,}")
    print(f"Unique influencers = {df[cluster].nunique():,}")

    print("\nA-path formula:")
    print(formula_a)
    print("\nB-path formula:")
    print(formula_b)

    # Main clustered models
    a_result = fit_clustered_ols(formula_a, df, cluster)
    b_result = fit_clustered_ols(formula_b, df, cluster)

    # Save summaries
    a_table = result_table(a_result, "A_path_mediator_model")
    b_table = result_table(b_result, "B_path_outcome_model")

    a_table.to_csv(output_dir / "rq1_a_path_results.csv", index=False)
    b_table.to_csv(output_dir / "rq1_b_path_results.csv", index=False)

    with open(output_dir / "rq1_model_summaries.txt", "w", encoding="utf-8") as f:
        f.write("=== A-PATH MODEL SUMMARY ===\n")
        f.write(a_result.summary().as_text())
        f.write("\n\n=== B-PATH MODEL SUMMARY ===\n")
        f.write(b_result.summary().as_text())

    # Conditional indirect effects
    authority_values = {
        "low_minus_1sd": -1.0,
        "mean_0sd": 0.0,
        "high_plus_1sd": 1.0,
    }

    cond_indirect = conditional_indirect_effects(a_result, b_result, x, w, m, authority_values)
    cond_indirect.to_csv(output_dir / "rq1_conditional_indirect_effects_point_estimates.csv", index=False)

    # Bootstrap CIs
    boot_summary, boot_raw = cluster_bootstrap_conditional_indirect(
        df=df,
        cluster_col=cluster,
        formula_a=formula_a,
        formula_b=formula_b,
        x=x,
        w=w,
        m=m,
        authority_values=authority_values,
        n_boot=n_boot,
        random_seed=random_seed,
    )

    boot_summary.to_csv(output_dir / "rq1_conditional_indirect_effects_bootstrap_summary.csv", index=False)
    boot_raw.to_csv(output_dir / "rq1_conditional_indirect_effects_bootstrap_raw.csv", index=False)

    # Prediction plots
    pred_a = make_prediction_frame_a(df, x, w, numeric_controls, categorical_controls)
    pred_a["_yhat"] = a_result.predict(pred_a)
    pred_a.to_csv(output_dir / "rq1_a_path_predictions.csv", index=False)

    pred_b = make_prediction_frame_b(df, m, x, w, numeric_controls, categorical_controls)
    pred_b["_yhat"] = b_result.predict(pred_b)
    pred_b.to_csv(output_dir / "rq1_b_path_predictions.csv", index=False)

    save_moderation_plot(
        pred_df=pred_a,
        x_col=x,
        yhat_col="_yhat",
        title="RQ1 A-Path Moderation: Persuasive Power -> Amplification by Authority",
        outpath=plots_dir / "rq1_a_path_moderation.png",
    )

    save_moderation_plot(
        pred_df=pred_b,
        x_col=m,
        yhat_col="_yhat",
        title="RQ1 B-Path Moderation: Amplification -> High-Effort Engagement by Authority",
        outpath=plots_dir / "rq1_b_path_moderation.png",
    )

    # Simple hypothesis support helper
    a_x_p = a_result.pvalues.get(x, np.nan)
    a_int_p = min(
        a_result.pvalues.get(f"{x}:{w}", np.nan),
        a_result.pvalues.get(f"{w}:{x}", np.nan)
    ) if (f"{x}:{w}" in a_result.pvalues.index or f"{w}:{x}" in a_result.pvalues.index) else np.nan
    b_m_p = b_result.pvalues.get(m, np.nan)
    b_mw_p = min(
        b_result.pvalues.get(f"{m}:{w}", np.nan),
        b_result.pvalues.get(f"{w}:{m}", np.nan)
    ) if (f"{m}:{w}" in b_result.pvalues.index or f"{w}:{m}" in b_result.pvalues.index) else np.nan

    hypothesis_summary = pd.DataFrame([
        {
            "hypothesis": "H1",
            "statement": "Persuasive power positively predicts algorithmic amplification.",
            "p_value": a_x_p,
            "supported": "Yes" if pd.notna(a_x_p) and a_x_p < 0.05 else "No",
        },
        {
            "hypothesis": "H2",
            "statement": "Algorithmic amplification positively predicts high-effort engagement.",
            "p_value": b_m_p,
            "supported": "Yes" if pd.notna(b_m_p) and b_m_p < 0.05 else "No",
        },
        {
            "hypothesis": "H3",
            "statement": "Authority moderates persuasive power -> amplification.",
            "p_value": a_int_p,
            "supported": "Yes" if pd.notna(a_int_p) and a_int_p < 0.05 else "No",
        },
        {
            "hypothesis": "H4",
            "statement": "Authority moderates amplification -> high-effort engagement.",
            "p_value": b_mw_p,
            "supported": "Yes" if pd.notna(b_mw_p) and b_mw_p < 0.05 else "No",
        },
    ])
    hypothesis_summary.to_csv(output_dir / "rq1_hypothesis_summary.csv", index=False)

    # Console summary
    print("\n=== RQ1 ANALYSIS COMPLETE ===")
    print(f"Results folder: {output_dir.resolve()}")
    print(f"A-path R^2: {a_result.rsquared:.4f}")
    print(f"B-path R^2: {b_result.rsquared:.4f}")

    print("\nConditional indirect effects (point estimates):")
    print(cond_indirect.to_string(index=False))

    print("\nBootstrap conditional indirect effects:")
    print(boot_summary.to_string(index=False))

    print("\nHypothesis summary:")
    print(hypothesis_summary.to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1 mediation/moderation models for the social media project.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to feature-engineered dataset")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save RQ1 outputs")
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS, help="Number of cluster bootstrap iterations")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for bootstrap")
    args = parser.parse_args()

    run_rq1(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        n_boot=args.bootstraps,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()