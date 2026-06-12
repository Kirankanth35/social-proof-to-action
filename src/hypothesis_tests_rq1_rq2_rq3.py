from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DEFAULT_INPUT = "dataset_feature_engineered.csv"
DEFAULT_OUTPUT_DIR = "hypothesis_test_outputs"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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


def fit_clustered_ols(formula: str, df: pd.DataFrame, cluster_col: str):
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col], "use_correction": True})


def result_table(result, model_name: str) -> pd.DataFrame:
    conf = result.conf_int()
    out = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "t": result.tvalues.values,
        "p_value": result.pvalues.values,
        "ci_low": conf.iloc[:, 0].values,
        "ci_high": conf.iloc[:, 1].values,
    })
    out["model"] = model_name
    out["r_squared"] = result.rsquared
    out["adj_r_squared"] = result.rsquared_adj
    out["n_obs"] = int(result.nobs)
    return out


def get_coef(result, terms: List[str]) -> float:
    for t in terms:
        if t in result.params.index:
            return float(result.params[t])
    return 0.0


def get_pvalue(result, terms: List[str]) -> float:
    for t in terms:
        if t in result.pvalues.index:
            return float(result.pvalues[t])
    return np.nan


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def prepare_df(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = normalize_colnames(df)

    # Required variables
    cluster = choose_first_existing(df, ["influencer_id", "creator_id", "account_id", "user_id"])
    if cluster is None:
        raise ValueError("Need an influencer/creator/account id column.")

    if "persuasive_power_index_z" not in df.columns:
        if "persuasive_power_index" in df.columns:
            df["persuasive_power_index_z"] = zscore(df["persuasive_power_index"])
        else:
            raise ValueError("Need persuasive_power_index or persuasive_power_index_z.")

    if "authority_z" not in df.columns:
        if "authority_log" in df.columns:
            df["authority_z"] = zscore(df["authority_log"])
        else:
            raise ValueError("Need authority_log or authority_z.")

    if "log_reach" not in df.columns:
        if "reach" in df.columns:
            df["log_reach"] = np.log1p(safe_numeric(df["reach"]).clip(lower=0))
        else:
            raise ValueError("Need reach or log_reach.")

    if "high_effort_engagement" not in df.columns:
        pieces = [c for c in ["link_clicks", "saves", "profile_visits"] if c in df.columns]
        if pieces:
            df["high_effort_engagement"] = df[pieces].apply(safe_numeric).fillna(0).sum(axis=1)
        else:
            raise ValueError("Need high_effort_engagement or link_clicks/saves/profile_visits.")

    if "log_high_effort_engagement" not in df.columns:
        df["log_high_effort_engagement"] = np.log1p(safe_numeric(df["high_effort_engagement"]).clip(lower=0))

    if "very_high_reach" not in df.columns:
        p95 = safe_numeric(df["reach"]).quantile(0.95)
        df["very_high_reach"] = (safe_numeric(df["reach"]) >= p95).astype(int)

    if "log_reach_sq" not in df.columns:
        df["log_reach_sq"] = safe_numeric(df["log_reach"]) ** 2

    # Optional controls
    controls_num = [c for c in ["verified", "account_age_years"] if c in df.columns]
    controls_cat = [c for c in ["content_type", "post_format", "year_month"] if c in df.columns]

    # Final keep
    keep = list(dict.fromkeys(
        [cluster, "persuasive_power_index_z", "authority_z", "log_reach",
         "log_high_effort_engagement", "very_high_reach", "log_reach_sq"]
        + controls_num + controls_cat
    ))
    if "engagement_success_index" in df.columns:
        keep.append("engagement_success_index")

    df = df[keep].copy()

    for c in ["persuasive_power_index_z", "authority_z", "log_reach",
              "log_high_effort_engagement", "log_reach_sq"]:
        df[c] = safe_numeric(df[c])

    if "engagement_success_index" in df.columns:
        df["engagement_success_index"] = safe_numeric(df["engagement_success_index"])

    for c in controls_num:
        df[c] = safe_numeric(df[c])

    for c in controls_cat:
        df[c] = df[c].astype("string")

    df = df.dropna(subset=["persuasive_power_index_z", "authority_z", "log_reach", "log_high_effort_engagement", cluster])

    vars_map = {
        "cluster": cluster,
        "x": "persuasive_power_index_z",
        "w": "authority_z",
        "m": "log_reach",
        "y": "log_high_effort_engagement",
        "y_alt": "engagement_success_index" if "engagement_success_index" in df.columns else "log_high_effort_engagement",
        "vh": "very_high_reach",
        "reach_sq": "log_reach_sq",
        "controls_num": controls_num,
        "controls_cat": controls_cat,
    }
    return df, vars_map


def formula_rhs(base_terms: List[str], controls_num: List[str], controls_cat: List[str]) -> str:
    rhs = base_terms + controls_num + [f"C({c})" for c in controls_cat]
    return " + ".join(rhs)


# ------------------------------------------------------------
# RQ1: Mediation + moderation
# H1, H2, H3, H4, H5
# ------------------------------------------------------------

def run_rq1(df: pd.DataFrame, v: Dict[str, str], outdir: Path) -> Dict[str, str]:
    x, w, m, y, cluster = v["x"], v["w"], v["m"], v["y"], v["cluster"]
    controls_num, controls_cat = v["controls_num"], v["controls_cat"]

    # H1 + H4: x -> m, moderated by authority
    formula_a = f"{m} ~ " + formula_rhs([x, w, f"{x}:{w}"], controls_num, controls_cat)
    res_a = fit_clustered_ols(formula_a, df, cluster)
    result_table(res_a, "RQ1_A_path").to_csv(outdir / "rq1_a_path.csv", index=False)

    # H2 + H5 context: m -> y, with x retained
    formula_b = f"{y} ~ " + formula_rhs([m, x, w, f"{m}:{w}", f"{x}:{w}"], controls_num, controls_cat)
    res_b = fit_clustered_ols(formula_b, df, cluster)
    result_table(res_b, "RQ1_B_path").to_csv(outdir / "rq1_b_path.csv", index=False)

    # Point-estimate indirect effects at authority = -1, 0, +1
    a1 = get_coef(res_a, [x])
    a3 = get_coef(res_a, [f"{x}:{w}", f"{w}:{x}"])
    b1 = get_coef(res_b, [m])
    b3 = get_coef(res_b, [f"{m}:{w}", f"{w}:{m}"])

    indirect_rows = []
    for label, aval in {"low_-1sd": -1.0, "mean_0sd": 0.0, "high_+1sd": 1.0}.items():
        indirect_rows.append({
            "authority_level": label,
            "authority_value": aval,
            "conditional_a_path": a1 + a3 * aval,
            "conditional_b_path": b1 + b3 * aval,
            "indirect_effect": (a1 + a3 * aval) * (b1 + b3 * aval),
        })
    pd.DataFrame(indirect_rows).to_csv(outdir / "rq1_conditional_indirect_effects.csv", index=False)

    summary = {
        "H1_supported": "Yes" if get_pvalue(res_a, [x]) < 0.05 and get_coef(res_a, [x]) > 0 else "No",
        "H2_supported": "Yes" if get_pvalue(res_b, [m]) < 0.05 and get_coef(res_b, [m]) > 0 else "No",
        "H3_supported": "Check indirect effect table",
        "H4_supported": "Yes" if get_pvalue(res_a, [f"{x}:{w}", f"{w}:{x}"]) < 0.05 and get_coef(res_a, [f"{x}:{w}", f"{w}:{x}"]) > 0 else "No",
        "H5_supported": "Check conditional indirect effects by authority",
    }

    with open(outdir / "rq1_summary.txt", "w", encoding="utf-8") as f:
        f.write(res_a.summary().as_text())
        f.write("\n\n")
        f.write(res_b.summary().as_text())
        f.write("\n\n")
        for k, val in summary.items():
            f.write(f"{k}: {val}\n")

    return summary


# ------------------------------------------------------------
# RQ2: Diminishing returns
# H6, H7
# ------------------------------------------------------------

def run_rq2(df: pd.DataFrame, v: Dict[str, str], outdir: Path) -> Dict[str, str]:
    m, y, cluster = v["m"], v["y"], v["cluster"]
    reach_sq = v["reach_sq"]
    controls_num, controls_cat = v["controls_num"], v["controls_cat"]

    formula = f"{y} ~ " + formula_rhs([m, reach_sq], controls_num, controls_cat)
    res = fit_clustered_ols(formula, df, cluster)
    result_table(res, "RQ2_Diminishing_Returns").to_csv(outdir / "rq2_diminishing_returns.csv", index=False)

    beta1 = get_coef(res, [m])
    beta2 = get_coef(res, [reach_sq])
    p1 = get_pvalue(res, [m])
    p2 = get_pvalue(res, [reach_sq])

    turning_point = np.nan
    if beta2 != 0:
        turning_point = -beta1 / (2 * beta2)

    summary = {
        "H6_supported": "Yes" if p1 < 0.05 and beta1 > 0 else "No",
        "H7_supported": "Yes" if p2 < 0.05 and beta2 < 0 else "No",
        "turning_point_log_reach": turning_point,
    }

    with open(outdir / "rq2_summary.txt", "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())
        f.write("\n\n")
        for k, val in summary.items():
            f.write(f"{k}: {val}\n")

    return summary


# ------------------------------------------------------------
# RQ3: High-visibility interaction
# H8, H9
# ------------------------------------------------------------

def run_rq3(df: pd.DataFrame, v: Dict[str, str], outdir: Path) -> Dict[str, str]:
    x, y, vh, cluster = v["x"], v["y"], v["vh"], v["cluster"]
    controls_num, controls_cat = v["controls_num"], v["controls_cat"]

    formula = f"{y} ~ " + formula_rhs([x, vh, f"{x}:{vh}"], controls_num, controls_cat)
    res = fit_clustered_ols(formula, df, cluster)
    result_table(res, "RQ3_High_Visibility").to_csv(outdir / "rq3_high_visibility.csv", index=False)

    beta_x = get_coef(res, [x])
    p_x = get_pvalue(res, [x])
    beta_int = get_coef(res, [f"{x}:{vh}", f"{vh}:{x}"])
    p_int = get_pvalue(res, [f"{x}:{vh}", f"{vh}:{x}"])

    summary = {
        "H8_supported": "Yes" if p_x < 0.05 and beta_x > 0 else "No",
        "H9_supported": "Yes" if p_int < 0.05 and beta_int < 0 else "No",
    }

    with open(outdir / "rq3_summary.txt", "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())
        f.write("\n\n")
        for k, val in summary.items():
            f.write(f"{k}: {val}\n")

    return summary


# ------------------------------------------------------------
# Master runner
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run hypothesis tests for RQ1, RQ2, and RQ3.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Feature-engineered dataset file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, vars_map = prepare_df(input_path)

    rq1_dir = outdir / "rq1"
    rq2_dir = outdir / "rq2"
    rq3_dir = outdir / "rq3"
    rq1_dir.mkdir(exist_ok=True)
    rq2_dir.mkdir(exist_ok=True)
    rq3_dir.mkdir(exist_ok=True)

    rq1_summary = run_rq1(df, vars_map, rq1_dir)
    rq2_summary = run_rq2(df, vars_map, rq2_dir)
    rq3_summary = run_rq3(df, vars_map, rq3_dir)

    overall = pd.DataFrame([
        {"hypothesis": "H1", "result": rq1_summary["H1_supported"]},
        {"hypothesis": "H2", "result": rq1_summary["H2_supported"]},
        {"hypothesis": "H3", "result": rq1_summary["H3_supported"]},
        {"hypothesis": "H4", "result": rq1_summary["H4_supported"]},
        {"hypothesis": "H5", "result": rq1_summary["H5_supported"]},
        {"hypothesis": "H6", "result": rq2_summary["H6_supported"]},
        {"hypothesis": "H7", "result": rq2_summary["H7_supported"]},
        {"hypothesis": "H8", "result": rq3_summary["H8_supported"]},
        {"hypothesis": "H9", "result": rq3_summary["H9_supported"]},
    ])
    overall.to_csv(outdir / "hypothesis_summary_all_rqs.csv", index=False)

    print("\n=== HYPOTHESIS TESTING COMPLETED ===")
    print(f"Input: {input_path.resolve()}")
    print(f"Outputs: {outdir.resolve()}")
    print("\nOverall hypothesis summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()