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


# ========================================
# CODE GENERATION
# ========================================

def generate_python_analysis_code(
    treatment_col: str,
    outcome_vars: List[str],
    baseline_outcome: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    apply_winsorization: bool = False
) -> str:
    """
    Generate complete Python script that reproduces the analysis
    
    Args:
        treatment_col: Name of treatment column
        outcome_vars: List of outcome variable names
        baseline_outcome: Optional baseline outcome for controls
        covariates: Optional list of covariates for balance tests
        apply_winsorization: Whether to include winsorization code
    
    Returns:
        Complete Python analysis script as string
    """
    covariates = covariates or []
    
    code = f"""# RCT Analysis - Generated by RCT Field Flow
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Set random seed for reproducibility
np.random.seed(123456)

# ========================================
# 1. LOAD DATA
# ========================================
# TODO: Update path to your data file
data = pd.read_csv('your_data.csv')  # or pd.read_stata('your_data.dta')

# ========================================
# 2. VARIABLE DEFINITIONS
# ========================================
treatment_col = '{treatment_col}'
outcome_vars = {outcome_vars}
baseline_outcome = '{baseline_outcome or "None"}'
covariates = {covariates}

# ========================================
# 3. DATA DIAGNOSTICS
# ========================================
def diagnose_outliers(series, name="variable"):
    stats_dict = {{
        'variable': name,
        'n': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }}
    
    # Outliers at 1%
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)
    n_outliers = ((series < lower) | (series > upper)).sum()
    
    print(f"\\n{{name}}: N={{stats_dict['n']}}, Mean={{stats_dict['mean']:.2f}}, "
          f"SD={{stats_dict['std']:.2f}}, Outliers={{n_outliers}}")
    
    return stats_dict

print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)
for var in outcome_vars:
    diagnose_outliers(data[var], var)
"""
    
    # Add cleaning code if requested
    if apply_winsorization:
        code += """
# ========================================  
# 4. DATA CLEANING
# ========================================
def winsorize(series, lower=1, upper=99):
    lower_bound = series.quantile(lower/100)
    upper_bound = series.quantile(upper/100)
    return series.clip(lower=lower_bound, upper=upper_bound)

# Create winsorized versions
for var in outcome_vars:
    data[f'{var}_w1'] = winsorize(data[var], 1, 99)
    data[f'{var}_w5'] = winsorize(data[var], 5, 95)
"""
    
    code += """
# ========================================
# 5. BALANCE TESTS (Baseline)
# ========================================
print("\\n" + "=" * 60)
print("BALANCE TESTS")
print("=" * 60)

balance_vars = outcome_vars + covariates if covariates else outcome_vars
for var in balance_vars:
    treat_mean = data[data[treatment_col] == 1][var].mean()
    control_mean = data[data[treatment_col] == 0][var].mean()
    diff = treat_mean - control_mean
    
    # t-test
    t_stat, p_val = stats.ttest_ind(
        data[data[treatment_col] == 1][var].dropna(),
        data[data[treatment_col] == 0][var].dropna()
    )
    
    print(f"{var}: Treat={treat_mean:.3f}, Control={control_mean:.3f}, "
          f"Diff={diff:.3f}, p={p_val:.3f}")

# ========================================
# 6. MAIN TREATMENT EFFECTS
# ========================================
print("\\n" + "=" * 60)
print("MAIN TREATMENT EFFECTS")
print("=" * 60)

results = {}

for outcome in outcome_vars:
    print(f"\\n--- {outcome} ---")
    
    # Specification 1: No controls
    formula1 = f"{outcome} ~ {treatment_col}"
    model1 = smf.ols(formula1, data=data).fit(cov_type='HC1')
    
    print(f"  (1) No controls: β={model1.params[treatment_col]:.3f}, "
          f"SE={model1.bse[treatment_col]:.3f}, p={model1.pvalues[treatment_col]:.3f}")
"""
    
    # Add baseline control specification if provided
    if baseline_outcome and baseline_outcome != "None":
        code += f"""    
    # Specification 2: With baseline control
    formula2 = f"{{outcome}} ~ {{treatment_col}} + {baseline_outcome}"
    model2 = smf.ols(formula2, data=data).fit(cov_type='HC1')
    
    print(f"  (2) With baseline: β={{model2.params[treatment_col]:.3f}}, "
          f"SE={{model2.bse[treatment_col]:.3f}}, p={{model2.pvalues[treatment_col]:.3f}}")
"""
    
    code += """    
    results[outcome] = model1
"""
    
    # Add robustness checks if winsorization was applied
    if apply_winsorization:
        code += """
# ========================================
# 7. ROBUSTNESS CHECKS
# ========================================
print("\\n" + "=" * 60)
print("ROBUSTNESS: WINSORIZED DATA")
print("=" * 60)

for outcome in outcome_vars:
    outcome_w = f'{outcome}_w1'
    if outcome_w in data.columns:
        formula = f"{outcome_w} ~ {treatment_col}"
        model = smf.ols(formula, data=data).fit(cov_type='HC1')
        print(f"{outcome} (1% winsor): β={model.params[treatment_col]:.3f}, "
              f"p={model.pvalues[treatment_col]:.3f}")
"""
    
    code += """
print("\\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
"""
    
    return code


def generate_stata_analysis_code(
    treatment_col: str,
    outcome_vars: List[str],
    baseline_outcome: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    apply_winsorization: bool = False
) -> str:
    """
    Generate complete Stata do-file that reproduces the analysis
    
    Args:
        treatment_col: Name of treatment column
        outcome_vars: List of outcome variable names
        baseline_outcome: Optional baseline outcome for controls
        covariates: Optional list of covariates for balance tests
        apply_winsorization: Whether to include winsorization code
    
    Returns:
        Complete Stata do-file as string
    """
    covariates = covariates or []
    outcomes_str = ' '.join(outcome_vars)
    covariates_str = ' '.join(covariates)
    
    code = f"""* RCT Analysis - Generated by RCT Field Flow
* Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

clear all
set more off
set seed 123456

* ========================================
* 1. LOAD DATA  
* ========================================
* TODO: Update path to your data file
use "your_data.dta", clear
* Or: import delimited "your_data.csv", clear

* ========================================
* 2. VARIABLE DEFINITIONS
* ========================================
local treatment "{treatment_col}"
local outcomes "{outcomes_str}"
local baseline "{baseline_outcome or ''}"
local covariates "{covariates_str}"

* ========================================
* 3. DATA DIAGNOSTICS
* ========================================
di _n "=" * 60
di "DATA DIAGNOSTICS"
di "=" * 60

foreach var of local outcomes {{
    qui sum `var', detail
    di _n "`var': N=" r(N) ", Mean=" r(mean) ", SD=" r(sd)
    di "  Skewness=" r(skewness) ", Kurtosis=" r(kurtosis)
    
    * Count outliers at 1%
    qui sum `var', detail
    local p1 = r(p1)
    local p99 = r(p99)
    qui count if `var' < `p1' | `var' > `p99'
    di "  Outliers (1%): " r(N)
}}
"""
    
    # Add cleaning code if requested
    if apply_winsorization:
        code += """
* ========================================
* 4. DATA CLEANING
* ========================================
* Winsorization (requires winsor command)
foreach var of local outcomes {
    * 1% winsorization
    egen `var'_w1 = winsor2(`var'), cuts(1 99)
    
    * 5% winsorization  
    egen `var'_w5 = winsor2(`var'), cuts(5 95)
}
"""
    
    code += """
* ========================================
* 5. BALANCE TESTS
* ========================================
di _n "=" * 60
di "BALANCE TESTS"
di "=" * 60

local balance_vars "`outcomes' `covariates'"
foreach var of local balance_vars {
    qui reg `var' `treatment', r
    di "`var': Treatment coef = " _b[`treatment'] ", p-value = " 2*ttail(e(df_r),abs(_b[`treatment']/_se[`treatment']))
}

* Joint F-test
if "`balance_vars'" != "" {
    reg `treatment' `balance_vars', r
    test `balance_vars'
}

* ========================================
* 6. MAIN TREATMENT EFFECTS
* ========================================
di _n "=" * 60
di "MAIN TREATMENT EFFECTS"
di "=" * 60

foreach outcome of local outcomes {
    di _n "--- `outcome' ---"
    
    * Specification 1: No controls
    qui reg `outcome' `treatment', r
    di "  (1) No controls: β = " _b[`treatment'] ", SE = " _se[`treatment'] ///
       ", p = " 2*tta il(e(df_r),abs(_b[`treatment']/_se[`treatment']))
"""
    
    # Add baseline control specification if provided
    if baseline_outcome and baseline_outcome != "None":
        code += f"""    
    * Specification 2: With baseline
    if "`baseline'" != "" {{
        qui reg `outcome' `treatment' `baseline', r
        di "  (2) With baseline: β = " _b[`treatment'] ", SE = " _se[`treatment'] ///
           ", p = " 2*ttail(e(df_r),abs(_b[`treatment']/_se[`treatment']))
    }}
"""
    
    code += """
}
"""
    
    # Add robustness checks if winsorization was applied
    if apply_winsorization:
        code += """
* ========================================
* 7. ROBUSTNESS CHECKS
* ========================================
di _n "=" * 60
di "ROBUSTNESS: WINSORIZED DATA"
di "=" * 60

foreach outcome of local outcomes {
    qui reg `outcome'_w1 `treatment', r
    di "`outcome' (1% winsor): β = " _b[`treatment'] ///
       ", p = " 2*ttail(e(df_r),abs(_b[`treatment']/_se[`treatment']))
}
"""
    
    code += """
di _n "=" * 60
di "ANALYSIS COMPLETE"
di "=" * 60

log close
"""
    
    return code

