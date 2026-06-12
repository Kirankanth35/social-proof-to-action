from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np


BASE_DIR = Path("hypothesis_test_outputs")
RQ1_DIR = BASE_DIR / "rq1"
RQ2_DIR = BASE_DIR / "rq2"
RQ3_DIR = BASE_DIR / "rq3"
OUTPUT_FILE = BASE_DIR / "summary_of_hypothesis_testing_results.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def get_row(df: pd.DataFrame, terms: List[str]) -> Optional[pd.Series]:
    if df is None or "term" not in df.columns:
        return None
    for t in terms:
        match = df[df["term"] == t]
        if not match.empty:
            return match.iloc[0]
    return None


def supported_positive(row: Optional[pd.Series]) -> str:
    if row is None:
        return "Could not evaluate"
    if row["p_value"] < 0.05 and row["coef"] > 0:
        return "Supported"
    return "Not supported"


def supported_negative(row: Optional[pd.Series]) -> str:
    if row is None:
        return "Could not evaluate"
    if row["p_value"] < 0.05 and row["coef"] < 0:
        return "Supported"
    return "Not supported"


def supported_ci_positive(ci_low: float, ci_high: float) -> str:
    if pd.notna(ci_low) and pd.notna(ci_high) and ci_low > 0 and ci_high > 0:
        return "Supported"
    return "Not supported"


def supported_ci_negative(ci_low: float, ci_high: float) -> str:
    if pd.notna(ci_low) and pd.notna(ci_high) and ci_low < 0 and ci_high < 0:
        return "Supported"
    return "Not supported"


# ------------------------------------------------------------
# Load outputs
# ------------------------------------------------------------

rq1_a = read_csv_if_exists(RQ1_DIR / "rq1_a_path.csv")
rq1_b = read_csv_if_exists(RQ1_DIR / "rq1_b_path.csv")
rq1_indirect = read_csv_if_exists(RQ1_DIR / "rq1_conditional_indirect_effects.csv")
rq1_boot = read_csv_if_exists(RQ1_DIR / "rq1_conditional_indirect_effects_bootstrap_summary.csv")

rq2_model = read_csv_if_exists(RQ2_DIR / "rq2_diminishing_returns.csv")

# support either name
rq3_model = read_csv_if_exists(RQ3_DIR / "rq3_high_visibility.csv")
if rq3_model is None:
    rq3_model = read_csv_if_exists(RQ3_DIR / "table_7_7_interaction_model.csv")

rq3_base = read_csv_if_exists(RQ3_DIR / "table_7_6_baseline_model.csv")


# ------------------------------------------------------------
# Evaluate hypotheses
# ------------------------------------------------------------

rows = []

# H1: Early persuasive signals -> amplification
h1_row = get_row(rq1_a, ["persuasive_power_index_z", "persuasive_power_index"])
rows.append({
    "Hypothesis": "H1",
    "Statement": "Early persuasive signals are positively associated with algorithmic amplification.",
    "Result": supported_positive(h1_row),
    "Evidence": f"coef={h1_row['coef']:.4f}, p={h1_row['p_value']:.4f}" if h1_row is not None else "Missing output"
})

# H2: Amplification -> high-effort engagement
h2_row = get_row(rq1_b, ["log_reach", "amplification_index", "reach"])
rows.append({
    "Hypothesis": "H2",
    "Statement": "Algorithmic amplification is positively associated with high-effort engagement outcomes.",
    "Result": supported_positive(h2_row),
    "Evidence": f"coef={h2_row['coef']:.4f}, p={h2_row['p_value']:.4f}" if h2_row is not None else "Missing output"
})

# H3: Mediation
h3_result = "Could not evaluate"
h3_evidence = "Missing indirect-effect output"
if rq1_boot is not None and {"authority_level", "ci_2_5", "ci_97_5"}.issubset(rq1_boot.columns):
    mean_row = rq1_boot[rq1_boot["authority_level"].astype(str).str.contains("mean", case=False, na=False)]
    if not mean_row.empty:
        r = mean_row.iloc[0]
        h3_result = supported_ci_positive(r["ci_2_5"], r["ci_97_5"])
        h3_evidence = f"95% CI=[{r['ci_2_5']:.4f}, {r['ci_97_5']:.4f}]"
elif rq1_indirect is not None:
    mean_row = rq1_indirect[rq1_indirect["authority_level"].astype(str).str.contains("mean", case=False, na=False)]
    if not mean_row.empty:
        r = mean_row.iloc[0]
        h3_result = "Supported" if r["indirect_effect"] > 0 else "Not supported"
        h3_evidence = f"indirect_effect={r['indirect_effect']:.4f}"

rows.append({
    "Hypothesis": "H3",
    "Statement": "Algorithmic amplification mediates the relationship between early persuasive signals and high-effort engagement outcomes.",
    "Result": h3_result,
    "Evidence": h3_evidence
})

# H4: Authority moderates persuasive power -> amplification
h4_row = get_row(rq1_a, [
    "persuasive_power_index_z:authority_z",
    "authority_z:persuasive_power_index_z",
    "persuasive_power_index:authority_log",
    "authority_log:persuasive_power_index",
])
rows.append({
    "Hypothesis": "H4",
    "Statement": "Influencer authority positively moderates the relationship between early persuasive signals and algorithmic amplification.",
    "Result": supported_positive(h4_row),
    "Evidence": f"coef={h4_row['coef']:.4f}, p={h4_row['p_value']:.4f}" if h4_row is not None else "Missing output"
})

# H5: Indirect effect stronger for high-authority influencers
h5_result = "Could not evaluate"
h5_evidence = "Missing conditional indirect-effect output"
if rq1_boot is not None and {"authority_level", "bootstrap_mean", "ci_2_5", "ci_97_5"}.issubset(rq1_boot.columns):
    low = rq1_boot[rq1_boot["authority_level"].astype(str).str.contains("low", case=False, na=False)]
    high = rq1_boot[rq1_boot["authority_level"].astype(str).str.contains("high", case=False, na=False)]
    if not low.empty and not high.empty:
        low_r = low.iloc[0]
        high_r = high.iloc[0]
        h5_result = "Supported" if high_r["bootstrap_mean"] > low_r["bootstrap_mean"] else "Not supported"
        h5_evidence = (
            f"low={low_r['bootstrap_mean']:.4f}, "
            f"high={high_r['bootstrap_mean']:.4f}"
        )
elif rq1_indirect is not None and {"authority_level", "indirect_effect"}.issubset(rq1_indirect.columns):
    low = rq1_indirect[rq1_indirect["authority_level"].astype(str).str.contains("low", case=False, na=False)]
    high = rq1_indirect[rq1_indirect["authority_level"].astype(str).str.contains("high", case=False, na=False)]
    if not low.empty and not high.empty:
        low_r = low.iloc[0]
        high_r = high.iloc[0]
        h5_result = "Supported" if high_r["indirect_effect"] > low_r["indirect_effect"] else "Not supported"
        h5_evidence = f"low={low_r['indirect_effect']:.4f}, high={high_r['indirect_effect']:.4f}"

rows.append({
    "Hypothesis": "H5",
    "Statement": "The indirect effect of early persuasive signals on high-effort engagement through algorithmic amplification is stronger for high-authority influencers.",
    "Result": h5_result,
    "Evidence": h5_evidence
})

# H6: Reach positively associated with high-effort engagement at lower/moderate levels
h6_row = get_row(rq2_model, ["log_reach", "reach"])
rows.append({
    "Hypothesis": "H6",
    "Statement": "Reach is positively associated with high-effort engagement at low-to-moderate levels of visibility.",
    "Result": supported_positive(h6_row),
    "Evidence": f"coef={h6_row['coef']:.4f}, p={h6_row['p_value']:.4f}" if h6_row is not None else "Missing output"
})

# H7: Diminishing marginal returns
h7_row = get_row(rq2_model, ["I(log_reach ** 2)", "log_reach_sq"])
rows.append({
    "Hypothesis": "H7",
    "Statement": "The positive effect of reach on high-effort engagement exhibits diminishing marginal returns at higher levels of reach.",
    "Result": supported_negative(h7_row),
    "Evidence": f"coef={h7_row['coef']:.4f}, p={h7_row['p_value']:.4f}" if h7_row is not None else "Missing output"
})

# H8: Persuasive power positive under low/moderate visibility
# Prefer baseline model if available; otherwise use interaction model main effect
h8_source = rq3_base if rq3_base is not None else rq3_model
h8_row = get_row(h8_source, ["persuasive_power_index", "persuasive_power_index_z"])
rows.append({
    "Hypothesis": "H8",
    "Statement": "Persuasive power is positively associated with high-effort engagement under low-to-moderate visibility conditions.",
    "Result": supported_positive(h8_row),
    "Evidence": f"coef={h8_row['coef']:.4f}, p={h8_row['p_value']:.4f}" if h8_row is not None else "Missing output"
})

# H9: Effect weakens as visibility increases
h9_row = get_row(rq3_model, [
    "log_reach:persuasive_power_index",
    "persuasive_power_index:log_reach",
    "very_high_reach:persuasive_power_index",
    "persuasive_power_index:very_high_reach",
])
rows.append({
    "Hypothesis": "H9",
    "Statement": "The positive effect of persuasive power on high-effort engagement weakens as algorithmic visibility increases.",
    "Result": supported_negative(h9_row),
    "Evidence": f"coef={h9_row['coef']:.4f}, p={h9_row['p_value']:.4f}" if h9_row is not None else "Missing output"
})

# H10: Under very high visibility, persuasive power is less predictive
# Only evaluate if very_high_reach interaction output exists
h10_row = get_row(rq3_model, [
    "very_high_reach:persuasive_power_index",
    "persuasive_power_index:very_high_reach",
    "very_high_reach:persuasive_power_index_z",
    "persuasive_power_index_z:very_high_reach",
])
h10_result = supported_negative(h10_row) if h10_row is not None else "Not tested / separate model needed"
h10_evidence = f"coef={h10_row['coef']:.4f}, p={h10_row['p_value']:.4f}" if h10_row is not None else "Model with very_high_reach interaction not found"

rows.append({
    "Hypothesis": "H10",
    "Statement": "Under very high visibility conditions, persuasive power is less predictive of high-effort engagement than under lower visibility conditions.",
    "Result": h10_result,
    "Evidence": h10_evidence
})

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTPUT_FILE, index=False)

print("\n=== SUMMARY OF HYPOTHESIS TESTING RESULTS ===")
print(summary_df.to_string(index=False))
print(f"\nSaved to: {OUTPUT_FILE.resolve()}")