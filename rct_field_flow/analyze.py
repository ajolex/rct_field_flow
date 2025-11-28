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


# ========================================
# DATA LOADING UTILITIES
# ========================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Stata .dta files
    
    Args:
        file_path: Path to data file (.csv or .dta)
    
    Returns:
        DataFrame with loaded data
    """
    if file_path.endswith('.dta'):
        # Disable convert_categoricals to avoid issues with duplicate category labels
        return pd.read_stata(file_path, convert_categoricals=False)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .csv or .dta files.")


# ========================================
# DATA DIAGNOSTICS
# ========================================

def diagnose_outliers(series: pd.Series, name: str = "variable") -> Dict:
    """
    Diagnose potential outliers and provide recommendations
    
    Args:
        series: Pandas series to analyze
        name: Variable name for display
    
    Returns:
        Dictionary with statistics, outlier info, and recommendations
    """
    # Basic statistics
    stats = {
        'variable': name,
        'n': int(series.count()),
        'n_missing': int(series.isna().sum()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'skewness': float(series.skew()),
        'kurtosis': float(series.kurtosis())
    }
    
    # Count outliers at different levels
    outliers = {}
    for pct in [1, 5, 10]:
        lower = series.quantile(pct/100)
        upper = series.quantile(1 - pct/100)
        mask = (series < lower) | (series > upper)
        outliers[f'{pct}pct'] = {
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'n_outliers': int(mask.sum()),
            'pct_outliers': float(mask.mean() * 100)
        }
    
    # Generate recommendation
    if stats['skewness'] > 2 or stats['kurtosis'] > 7:
        recommendation = "⚠️ High skewness/kurtosis detected. Winsorization recommended."
        severity = 'high'
    elif outliers['1pct']['pct_outliers'] > 5:
        recommendation = "⚠️ Multiple extreme values detected. Consider winsorization or trimming."
        severity = 'medium'
    else:
        recommendation = "✅ Data appears well-behaved. Cleaning may not be necessary."
        severity = 'low'
    
    return {
        'statistics': stats,
        'outliers': outliers,
        'recommendation': recommendation,
        'severity': severity
    }


def run_data_diagnostics(data: pd.DataFrame, outcome_vars: List[str]) -> Dict[str, Dict]:
    """
    Run diagnostics on all outcome variables
    
    Args:
        data: DataFrame containing the data
        outcome_vars: List of outcome variable names to diagnose
    
    Returns:
        Dictionary mapping variable names to diagnostic results
    """
    diagnostics = {}
    for var in outcome_vars:
        if var in data.columns:
            diagnostics[var] = diagnose_outliers(data[var], name=var)
        else:
            diagnostics[var] = {
                'error': f'Variable {var} not found in data'
            }
    return diagnostics


# ========================================
# DATA CLEANING FUNCTIONS
# ========================================

def winsorize_variable(series: pd.Series, lower: float = 1, upper: float = 99) -> pd.Series:
    """
    Winsorize a variable at specified percentiles
    
    Args:
        series: Pandas series to winsorize
        lower: Lower percentile threshold (default 1)
        upper: Upper percentile threshold (default 99)
    
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower/100)
    upper_bound = series.quantile(upper/100)
    return series.clip(lower=lower_bound, upper=upper_bound)


def trim_variable(series: pd.Series, lower: float = 1, upper: float = 99) -> pd.Series:
    """
    Trim extreme values by setting them to NaN
    
    Args:
        series: Pandas series to trim
        lower: Lower percentile threshold (default 1)
        upper: Upper percentile threshold (default 99)
    
    Returns:
        Trimmed series with extremes set to NaN
    """
    lower_bound = series.quantile(lower/100)
    upper_bound = series.quantile(upper/100)
    mask = (series >= lower_bound) & (series <= upper_bound)
    return series.where(mask, pd.NA)


def create_missing_indicators(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Create dummy variables for missing values and fill with 0
    
    Args:
        df: DataFrame to process
        columns: List of column names to create indicators for
    
    Returns:
        DataFrame with missing indicators and filled values
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[f'{col}_missing'] = df_copy[col].isna().astype(int)
            df_copy[col] = df_copy[col].fillna(0)
    return df_copy


# ========================================
# BALANCE TESTS
# ========================================

def check_balance(
    data: pd.DataFrame, 
    treatment_col: str, 
    covariates: List[str]
) -> pd.DataFrame:
    """
    Check balance of covariates across treatment arms
    
    Args:
        data: DataFrame containing the data
        treatment_col: Name of treatment column
        covariates: List of covariate names to check
    
    Returns:
        DataFrame with balance test results
    """
    from scipy import stats as scipy_stats
    
    results = []
    for var in covariates:
        if var not in data.columns:
            continue
            
        # Calculate means by treatment arm
        treat_vals = data[data[treatment_col] == 1][var].dropna()
        control_vals = data[data[treatment_col] == 0][var].dropna()
        
        # t-test
        t_stat, p_val = scipy_stats.ttest_ind(treat_vals, control_vals)
        
        results.append({
            'variable': var,
            'treatment_mean': treat_vals.mean(),
            'treatment_sd': treat_vals.std(),
            'treatment_n': len(treat_vals),
            'control_mean': control_vals.mean(),
            'control_sd': control_vals.std(),
            'control_n': len(control_vals),
            'difference': treat_vals.mean() - control_vals.mean(),
            't_statistic': t_stat,
            'p_value': p_val
        })
    
    return pd.DataFrame(results)

