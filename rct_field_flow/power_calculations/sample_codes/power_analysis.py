"""
Power analysis for Soft-Skills RCT (Barangay-level CRT)
Author: RA/PI team

Reads PSPS analysis dataset and computes descriptives, ICCs, design effects, and
minimum detectable effects (MDEs) for four outcomes:
 - labor_part (0/1)
 - entrep_engage (0/1)
 - agency_prop (proportion or index)
 - aspirations_gap_w (continuous)

Assumptions reflect the pilot design:
 - K = 8 clusters per arm (16 total)
 - m ≈ 18 endline sample per cluster (≈10% attrition from baseline ~20)
 - alpha = 0.05 (two-sided); power = 0.80

CLI options allow specifying dataset path and id variable names when they differ
from defaults; the script will also attempt to auto-detect common synonyms.

Outputs CSV summaries for transparency and inclusion in the proposal.
"""

import math
import os
import sys
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import argparse

try:
    import statsmodels.api as sm
except Exception:
    sm = None
    warnings.warn("statsmodels not found; ICCs will be approximated via ANOVA components.")


# -----------------------
# User parameters
# -----------------------
K_PER_ARM = 8
M_ENDLINE = 18  # ~10% attrition from ~20 baseline
ALPHA = 0.05
POWER = 0.80
seed = 3061992

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "analysis_data.dta")


def invnorm(p: float) -> float:
    """Inverse normal CDF via statsmodels if available, else scipy/numpy fallback."""
    try:
        from scipy.stats import norm
        return norm.ppf(p)
    except Exception:
        if sm is not None:
            return sm.distributions.norm_sinv(p)
        # Approximation if everything else fails
        return math.sqrt(2) * erfinv(2 * p - 1)


def erfinv(y: float) -> float:
    # Approximate inverse error function (Hastings, 1955) for fallback use only
    a = 0.147  # magic constant
    sign = 1 if y >= 0 else -1
    ln = math.log(1 - y * y)
    first = 2 / (math.pi * a) + ln / 2
    second = ln / a
    return sign * math.sqrt(math.sqrt(first * first - second) - first)


def anova_icc(y: np.ndarray, g: np.ndarray) -> float:
    """One-way ANOVA ICC estimator: (MSB - MSW) / (MSB + (mbar - 1)*MSW)."""
    # Drop missing
    mask = ~np.isnan(y) & ~pd.isna(g)
    y = y[mask]
    g = g[mask]
    if y.size == 0:
        return np.nan
    # Group stats
    df = pd.DataFrame({"y": y, "g": g})
    groups = df.groupby("g")
    n_i = groups.size().values
    k = n_i.size
    ybar_i = groups["y"].mean().values
    ybar = df["y"].mean()
    # Mean squares
    ssb = np.sum(n_i * (ybar_i - ybar) ** 2)
    msb = ssb / (k - 1) if k > 1 else 0.0
    ssw = np.sum(groups.apply(lambda g: ((g["y"] - g["y"].mean()) ** 2).sum()))
    msw = ssw / (df.shape[0] - k) if df.shape[0] > k else 0.0
    mbar = n_i.mean()
    if msb + (mbar - 1) * msw == 0:
        return 0.0
    return max(0.0, (msb - msw) / (msb + (mbar - 1) * msw))


def compute_binary_stats(series: pd.Series) -> Tuple[float, float]:
    p = series.mean()
    sd = math.sqrt(p * (1 - p))
    return p, sd


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="PSPS-based power analysis")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to .dta analysis dataset")
    parser.add_argument("--cluster-var", dest="cluster", default=None, help="Cluster id variable (e.g., brgy_code)")
    parser.add_argument("--id-var", dest="caseid", default=None, help="Individual id variable (e.g., caseid)")
    # Optional overrides for outcome variable names
    parser.add_argument("--var-labor", dest="var_labor", default=None, help="Labor participation variable name")
    parser.add_argument("--var-entre", dest="var_entre", default=None, help="Entrepreneurship engagement variable name")
    parser.add_argument("--var-agency", dest="var_agency", default=None, help="Agency index/proportion variable name")
    parser.add_argument("--var-asp", dest="var_asp", default=None, help="Aspirations gap variable name")
    parser.add_argument("--var-agefirst", dest="var_agefirst", default=None, help="Age at first marriage variable name (optional)")
    args = parser.parse_args(argv)

    data_path = args.data
    if not os.path.exists(data_path):
        print(f"ERROR: analysis dataset not found: {data_path}")
        sys.exit(1)

    try:
        import pyreadstat
        df, meta = pyreadstat.read_dta(data_path)
    except Exception as e:
        print("ERROR: Unable to read dataset:", e)
        sys.exit(1)

    # Resolve ID variables (allow CLI override and auto-detect common variants)
    cluster_col = args.cluster or find_first_existing(df, [
        "brgy_code", "barangay_id", "brgyid", "brgycode", "barangay_code", "cluster", "cluster_id", "psu"
    ])
    if cluster_col is None:
        print("ERROR: Missing cluster variable. Tried: brgy_code, barangay_id, brgyid, brgycode, barangay_code, cluster, cluster_id, psu")
        print("Columns available:", ", ".join(df.columns))
        sys.exit(1)

    id_col = args.caseid or find_first_existing(df, [
        "caseid", "case_id", "id", "respondent_id", "person_id", "uid"
    ])
    if id_col is None:
        print("ERROR: Missing individual id variable. Tried: caseid, case_id, id, respondent_id, person_id, uid")
        print("Columns available:", ", ".join(df.columns))
        sys.exit(1)

    # Resolve outcome variable names (allow CLI override and auto-detect)
    def resolve(var_cli: str | None, candidates: list[str], label: str) -> str:
        if var_cli:
            if var_cli in df.columns:
                return var_cli
            print(f"ERROR: {label} variable '{var_cli}' not found in columns.")
            print("Columns available:", ", ".join(df.columns))
            sys.exit(1)
        found = find_first_existing(df, candidates)
        if found:
            return found
        print(f"ERROR: Missing {label} variable. Tried: {', '.join(candidates)}")
        print("Columns available:", ", ".join(df.columns))
        sys.exit(1)

    labor_col = resolve(args.var_labor, ["labor_part", "labor", "lfp", "any_work30", "paid_work30", "work30"], "labor participation")
    entre_col = resolve(args.var_entre, ["entrep_engage", "entrepreneurship", "self_employed30", "business30"], "entrepreneurial engagement")
    agency_col = resolve(args.var_agency, ["agency_prop", "agency_index", "agency_sd", "agency"], "agency")
    asp_col = resolve(args.var_asp, ["aspirations_gap_w", "aspirations_gap", "asp_gap_w", "asp_gap"], "aspirations gap")

    # Coerce to numeric
    for v in ["labor_part", "entrep_engage", "agency_prop", "aspirations_gap_w"]:
        df[v] = pd.to_numeric(df[v], errors="coerce")

    # Compute stats and ICCs
    rows = []
    # Map canonical outcome keys to actual dataset columns
    outcome_map = {
        "labor_part": (labor_col, True),
        "entrep_engage": (entre_col, True),
        "agency_prop": (agency_col, False),
        "aspirations_gap_w": (asp_col, False),
    }

    for outkey, (var, is_binary) in outcome_map.items():
        y = df[var].values.astype(float)
        g = df[cluster_col].values
        icc = anova_icc(y, g)
        if is_binary:
            mean, sd = compute_binary_stats(pd.Series(y).dropna())
        else:
            s = pd.Series(y).dropna()
            mean = float(s.mean()) if len(s) else float("nan")
            sd = float(s.std(ddof=1)) if len(s) else float("nan")
        rows.append({"outcome": outkey, "mean": mean, "sd": sd, "icc": icc})

    stats = pd.DataFrame(rows)

    # MDEs
    z1 = invnorm(1 - ALPHA / 2)
    z2 = invnorm(POWER)
    mde_rows = []
    for _, r in stats.iterrows():
        de = 1 + (M_ENDLINE - 1) * max(0.0, r["icc"])  # guard against negative
        if r["outcome"] in ["labor_part", "entrep_engage"]:
            p = r["mean"]
            mde = (z1 + z2) * math.sqrt(2 * de * p * (1 - p) / (K_PER_ARM * M_ENDLINE))
        else:
            mde = (z1 + z2) * math.sqrt(2 * de / (K_PER_ARM * M_ENDLINE))
        mde_rows.append({
            "outcome": r["outcome"],
            "mean": r["mean"],
            "sd": r["sd"],
            "icc": r["icc"],
            "DE": de,
            "MDE": mde,
        })

    power_summary = pd.DataFrame(mde_rows).sort_values("outcome")
    power_summary.to_csv(os.path.join(os.path.dirname(__file__), "power_summary.csv"), index=False)

    # For continuous outcomes, also export SD-scale MDEs explicitly
    cont = power_summary[power_summary["outcome"].isin(["agency_prop", "aspirations_gap_w", "aspirations_gap"])].copy()
    cont["MDE_in_SD"] = cont["MDE"]  # already in SD units under the formula
    cont.to_csv(os.path.join(os.path.dirname(__file__), "power_mde_continuous.csv"), index=False)

    print("Saved power_summary.csv and power_mde_continuous.csv")

    # Optional: export early-marriage prevalence by cluster if age at first marriage is available
    if args.var_agefirst and args.var_agefirst in df.columns:
        try:
            age_first = pd.to_numeric(df[args.var_agefirst], errors="coerce")
            early = (age_first <= 18).astype(float)
            grp = (
                pd.DataFrame({"cluster": df[cluster_col], "early": early})
                .dropna()
                .groupby("cluster")
                .agg(n=("early", "size"), early_share=("early", "mean"))
                .reset_index()
            )
            out_csv = os.path.join(os.path.dirname(__file__), "early_marriage_prevalence.csv")
            grp.to_csv(out_csv, index=False)
            overall = early.mean()
            print(f"Saved early_marriage_prevalence.csv (overall early marriage share: {overall:0.3f})")
        except Exception as e:
            print(f"Warning: failed to compute early marriage prevalence: {e}")


if __name__ == "__main__":
    main()
