from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Optional

import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class AnalysisConfig:
    treatment_column: str = "treatment"
    weight_column: Optional[str] = None
    cluster_column: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict | None) -> "AnalysisConfig":
        if not raw:
            return cls()
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        return cls(**filtered)


def estimate_ate(
    df: pd.DataFrame,
    outcome: str,
    covariates: Optional[Iterable[str]] = None,
    config: AnalysisConfig | None = None,
):
    """Run an OLS impact estimate with optional covariates and clustering."""
    cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig.from_dict(config)
    covariates = list(covariates or [])
    formula = f"{outcome} ~ 1 + C({cfg.treatment_column})"
    if covariates:
        cov_terms = " + ".join(covariates)
        formula = f"{formula} + {cov_terms}"

    model = smf.wls if cfg.weight_column else smf.ols
    weights = df[cfg.weight_column] if cfg.weight_column else None
    fitted_model = model(formula, data=df, weights=weights)

    if cfg.cluster_column:
        result = fitted_model.fit(
            cov_type="cluster", cov_kwds={"groups": df[cfg.cluster_column]}
        )
    else:
        result = fitted_model.fit()
    return result


def heterogeneity_analysis(
    df: pd.DataFrame,
    outcome: str,
    moderator: str,
    covariates: Optional[Iterable[str]] = None,
    config: AnalysisConfig | None = None,
):
    """Estimate heterogeneous treatment effects via interaction terms."""
    cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig.from_dict(config)
    covariates = list(covariates or [])
    formula = f"{outcome} ~ C({cfg.treatment_column}) * C({moderator})"
    if covariates:
        cov_terms = " + ".join(covariates)
        formula = f"{formula} + {cov_terms}"

    model = smf.ols(formula, data=df)
    if cfg.cluster_column:
        return model.fit(cov_type="cluster", cov_kwds={"groups": df[cfg.cluster_column]})
    return model.fit()


def attrition_table(
    baseline: pd.DataFrame,
    endline: pd.DataFrame,
    id_column: str,
    treatment_column: str,
) -> pd.DataFrame:
    """Compute attrition by treatment arm."""
    base = baseline[[id_column, treatment_column]].drop_duplicates()
    end = endline[[id_column]].drop_duplicates()
    merged = base.merge(end, on=id_column, how="left", indicator=True)
    merged["attrited"] = merged["_merge"] == "left_only"
    table = (
        merged.groupby(treatment_column)["attrited"]
        .agg(rate="mean", count="size")
        .reset_index()
    )
    return table


def one_click_analysis(
    df: pd.DataFrame,
    outcomes: List[str],
    config: Dict | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run ATEs for a list of outcomes."""
    cfg = AnalysisConfig.from_dict(config)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for outcome in outcomes:
        model = estimate_ate(df, outcome, config=cfg)
        arm_estimates: Dict[str, Dict[str, float]] = {}
        prefix = f"C({cfg.treatment_column})[T."
        for param, value in model.params.items():
            if param.startswith(prefix):
                arm = param[len(prefix) :].rstrip("]")
                arm_estimates[arm] = {
                    "estimate": value,
                    "p_value": model.pvalues.get(param, float("nan")),
                }
        results[outcome] = arm_estimates
    return results
