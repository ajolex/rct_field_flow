from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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

def load_data(file_path) -> pd.DataFrame:
    """
    Load data from CSV or Stata .dta files
    
    Args:
        file_path: Path to data file (.csv or .dta) or UploadedFile object
    
    Returns:
        DataFrame with loaded data
    """
    # Handle UploadedFile objects from Streamlit
    if hasattr(file_path, 'name'):
        # It's an UploadedFile object
        filename = file_path.name
        if filename.endswith('.dta'):
            # Disable convert_categoricals to avoid issues with duplicate category labels
            return pd.read_stata(file_path, convert_categoricals=False)
        elif filename.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .csv or .dta files.")
    else:
        # It's a string file path
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
    Check balance of covariates across treatment arms (legacy function - use generate_balance_table for enhanced version)
    
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


def generate_balance_table(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: List[str],
    cluster_col: Optional[str] = None,
    format_style: str = 'dataframe'
) -> pd.DataFrame:
    """
    Generate comprehensive balance table (Table 1) with joint orthogonality test
    
    Implements methodology from Bruhn, Karlan & Schoar (2018):
    - Individual t-tests for each covariate
    - Joint F-test for orthogonality
    - Significance stars: *p<0.1, **p<0.05, ***p<0.01
    - Cluster-robust standard errors if cluster_col provided
    
    Args:
        data: DataFrame containing the data
        treatment_col: Name of treatment column (should be 0/1)
        covariates: List of covariate names to test
        cluster_col: Optional cluster variable for robust SEs
        format_style: 'dataframe' or 'latex' or 'html'
    
    Returns:
        DataFrame with columns [Variable, Treatment_Mean, Treatment_SD, Control_Mean, Control_SD, Difference, SE, P_Value, Stars, N_Treatment, N_Control]
    """
    from scipy import stats as scipy_stats
    import numpy as np
    import statsmodels.api as sm
    
    # Filter valid covariates
    valid_covs = [v for v in covariates if v in data.columns]
    
    results = []
    for var in valid_covs:
        # Remove missing values
        subset = data[[var, treatment_col]].dropna()
        
        # Calculate means by treatment arm
        treat_vals = subset[subset[treatment_col] == 1][var]
        control_vals = subset[subset[treatment_col] == 0][var]
        
        if len(treat_vals) == 0 or len(control_vals) == 0:
            continue
        
        # Run regression: var ~ treatment (for robust SE option)
        X = sm.add_constant(subset[treatment_col])
        y = subset[var]
        
        if cluster_col and cluster_col in data.columns:
            # Cluster-robust standard errors
            clusters = subset[cluster_col] if cluster_col in subset.columns else data.loc[subset.index, cluster_col]
            try:
                model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': clusters})
                coef = model.params[treatment_col]
                se = model.bse[treatment_col]
                p_val = model.pvalues[treatment_col]
            except:
                # Fallback to regular t-test
                t_stat, p_val = scipy_stats.ttest_ind(treat_vals, control_vals)
                coef = treat_vals.mean() - control_vals.mean()
                se = np.sqrt(treat_vals.var()/len(treat_vals) + control_vals.var()/len(control_vals))
        else:
            # Standard t-test
            model = sm.OLS(y, X).fit(cov_type='HC1')  # Heteroskedasticity-robust
            coef = model.params[treatment_col]
            se = model.bse[treatment_col]
            p_val = model.pvalues[treatment_col]
        
        # Significance stars
        if p_val < 0.01:
            stars = '***'
        elif p_val < 0.05:
            stars = '**'
        elif p_val < 0.1:
            stars = '*'
        else:
            stars = ''
        
        results.append({
            'Variable': var,
            'Treatment_Mean': treat_vals.mean(),
            'Treatment_SD': treat_vals.std(),
            'Control_Mean': control_vals.mean(),
            'Control_SD': control_vals.std(),
            'Difference': coef,
            'SE': se,
            'P_Value': p_val,
            'Stars': stars,
            'N_Treatment': len(treat_vals),
            'N_Control': len(control_vals)
        })
    
    balance_df = pd.DataFrame(results)
    
    # Add joint orthogonality test (F-test)
    # Regress treatment on all covariates: treatment ~ cov1 + cov2 + ...
    if len(valid_covs) > 0:
        subset_all = data[[treatment_col] + valid_covs].dropna()
        if len(subset_all) > 0:
            X = sm.add_constant(subset_all[valid_covs])
            y = subset_all[treatment_col]
            
            if cluster_col and cluster_col in data.columns:
                clusters = data.loc[subset_all.index, cluster_col]
                try:
                    model_joint = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': clusters})
                except:
                    model_joint = sm.OLS(y, X).fit(cov_type='HC1')
            else:
                model_joint = sm.OLS(y, X).fit(cov_type='HC1')
            
            # F-test: all covariate coefficients jointly zero
            try:
                # Test all variables except constant
                hypotheses = [f'{cov} = 0' for cov in valid_covs]
                f_test = model_joint.f_test(hypotheses)
                f_stat = f_test.fvalue[0][0]
                f_pval = f_test.pvalue
                
                # Add joint test row
                joint_row = {
                    'Variable': 'Joint F-test',
                    'Treatment_Mean': np.nan,
                    'Treatment_SD': np.nan,
                    'Control_Mean': np.nan,
                    'Control_SD': np.nan,
                    'Difference': f_stat,
                    'SE': np.nan,
                    'P_Value': f_pval,
                    'Stars': '***' if f_pval < 0.01 else '**' if f_pval < 0.05 else '*' if f_pval < 0.1 else '',
                    'N_Treatment': np.nan,
                    'N_Control': np.nan
                }
                balance_df = pd.concat([balance_df, pd.DataFrame([joint_row])], ignore_index=True)
            except:
                pass  # Skip joint test if it fails
    
    return balance_df


def estimate_itt(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    covariates: Optional[List[str]] = None,
    baseline_outcome: Optional[str] = None,
    cluster_col: Optional[str] = None,
    strata_cols: Optional[List[str]] = None,
    weight_col: Optional[str] = None
) -> Dict:
    """
    Estimate Intent-to-Treat (ITT) effect using multiple specifications
    
    Implements methodology from Bruhn, Karlan & Schoar (2018):
    - Specification 1: Y ~ T (no controls)
    - Specification 2: Y ~ T + X (with covariates and strata)
    - Specification 3: Y ~ T + X + Y_baseline (ANCOVA for precision)
    
    All specifications use heteroskedasticity-robust standard errors (HC1).
    If cluster_col provided, uses cluster-robust standard errors.
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of treatment column (should be 0/1)
        covariates: Optional list of control variables
        baseline_outcome: Optional baseline value of outcome (for ANCOVA)
        cluster_col: Optional cluster variable for robust SEs
        strata_cols: Optional stratification variables (e.g., i.ran_size_sector)
        weight_col: Optional weight variable
    
    Returns:
        Dictionary with results for each specification:
        {
            'spec1': {'coef': ..., 'se': ..., 'pvalue': ..., 'ci_lower': ..., 'ci_upper': ..., 'n': ..., 'r2': ...},
            'spec2': {...},
            'spec3': {...},
            'outcome_var': str,
            'treatment_col': str,
            'control_mean': float,
            'control_sd': float
        }
    """
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.formula.api import ols, wls
    
    # Prepare data: remove missing values for outcome and treatment
    subset = data[[outcome_var, treatment_col]].dropna()
    valid_indices = subset.index
    
    # Calculate control group statistics
    control_data = data[data[treatment_col] == 0][outcome_var].dropna()
    control_mean = control_data.mean()
    control_sd = control_data.std()
    
    results = {
        'outcome_var': outcome_var,
        'treatment_col': treatment_col,
        'control_mean': control_mean,
        'control_sd': control_sd
    }
    
    covariates = covariates or []
    strata_cols = strata_cols or []
    
    # Determine covariance type
    cov_type = 'HC1'  # Heteroskedasticity-robust
    cov_kwds = {}
    if cluster_col and cluster_col in data.columns:
        cov_type = 'cluster'
        cov_kwds = {'groups': data.loc[valid_indices, cluster_col]}
    
    # ============================================
    # Specification 1: Y ~ T (no controls)
    # ============================================
    formula1 = f'{outcome_var} ~ {treatment_col}'
    
    try:
        if weight_col and weight_col in data.columns:
            model1 = wls(formula1, data=data.loc[valid_indices], weights=data.loc[valid_indices, weight_col])
        else:
            model1 = ols(formula1, data=data.loc[valid_indices])
        
        fit1 = model1.fit(cov_type=cov_type, cov_kwds=cov_kwds) if cov_kwds else model1.fit(cov_type=cov_type)
        
        results['spec1'] = {
            'coef': fit1.params[treatment_col],
            'se': fit1.bse[treatment_col],
            'pvalue': fit1.pvalues[treatment_col],
            'ci_lower': fit1.conf_int().loc[treatment_col, 0],
            'ci_upper': fit1.conf_int().loc[treatment_col, 1],
            't_stat': fit1.tvalues[treatment_col],
            'n': int(fit1.nobs),
            'r2': fit1.rsquared,
            'stars': '***' if fit1.pvalues[treatment_col] < 0.01 else '**' if fit1.pvalues[treatment_col] < 0.05 else '*' if fit1.pvalues[treatment_col] < 0.1 else ''
        }
    except Exception as e:
        results['spec1'] = {'error': str(e)}
    
    # ============================================
    # Specification 2: Y ~ T + X + Strata (with controls)
    # ============================================
    if len(covariates) > 0 or len(strata_cols) > 0:
        # Filter valid covariates and strata
        all_controls = covariates + strata_cols
        valid_controls = [c for c in all_controls if c in data.columns]
        
        # Build formula with categorical strata
        formula2_parts = [outcome_var, '~', treatment_col]
        for cov in covariates:
            if cov in data.columns:
                formula2_parts.append(f'+ {cov}')
        for strata in strata_cols:
            if strata in data.columns:
                formula2_parts.append(f'+ C({strata})')  # Categorical
        
        formula2 = ' '.join(formula2_parts)
        
        # Remove rows with missing controls
        subset2_cols = [outcome_var, treatment_col] + valid_controls
        subset2 = data[subset2_cols].dropna()
        valid_indices2 = subset2.index
        
        try:
            if weight_col and weight_col in data.columns:
                model2 = wls(formula2, data=data.loc[valid_indices2], weights=data.loc[valid_indices2, weight_col])
            else:
                model2 = ols(formula2, data=data.loc[valid_indices2])
            
            cov_kwds2 = {'groups': data.loc[valid_indices2, cluster_col]} if cluster_col and cluster_col in data.columns else {}
            fit2 = model2.fit(cov_type=cov_type, cov_kwds=cov_kwds2) if cov_kwds2 else model2.fit(cov_type=cov_type)
            
            results['spec2'] = {
                'coef': fit2.params[treatment_col],
                'se': fit2.bse[treatment_col],
                'pvalue': fit2.pvalues[treatment_col],
                'ci_lower': fit2.conf_int().loc[treatment_col, 0],
                'ci_upper': fit2.conf_int().loc[treatment_col, 1],
                't_stat': fit2.tvalues[treatment_col],
                'n': int(fit2.nobs),
                'r2': fit2.rsquared,
                'stars': '***' if fit2.pvalues[treatment_col] < 0.01 else '**' if fit2.pvalues[treatment_col] < 0.05 else '*' if fit2.pvalues[treatment_col] < 0.1 else ''
            }
        except Exception as e:
            results['spec2'] = {'error': str(e)}
    
    # ============================================
    # Specification 3: Y ~ T + X + Y_baseline (ANCOVA)
    # ============================================
    if baseline_outcome and baseline_outcome in data.columns:
        # Create missing indicator for baseline
        baseline_missing_col = f'{baseline_outcome}_d'
        data_copy = data.copy()
        data_copy[baseline_missing_col] = data_copy[baseline_outcome].isna().astype(int)
        data_copy[baseline_outcome] = data_copy[baseline_outcome].fillna(0)
        
        # Build formula
        formula3_parts = [outcome_var, '~', treatment_col, f'+ {baseline_outcome}', f'+ {baseline_missing_col}']
        for cov in covariates:
            if cov in data.columns:
                formula3_parts.append(f'+ {cov}')
        for strata in strata_cols:
            if strata in data.columns:
                formula3_parts.append(f'+ C({strata})')
        
        formula3 = ' '.join(formula3_parts)
        
        # Remove rows with missing outcome or treatment
        subset3_cols = [outcome_var, treatment_col, baseline_outcome, baseline_missing_col] + [c for c in covariates + strata_cols if c in data.columns]
        subset3 = data_copy[subset3_cols].dropna(subset=[outcome_var, treatment_col])
        valid_indices3 = subset3.index
        
        try:
            if weight_col and weight_col in data.columns:
                model3 = wls(formula3, data=data_copy.loc[valid_indices3], weights=data_copy.loc[valid_indices3, weight_col])
            else:
                model3 = ols(formula3, data=data_copy.loc[valid_indices3])
            
            cov_kwds3 = {'groups': data_copy.loc[valid_indices3, cluster_col]} if cluster_col and cluster_col in data.columns else {}
            fit3 = model3.fit(cov_type=cov_type, cov_kwds=cov_kwds3) if cov_kwds3 else model3.fit(cov_type=cov_type)
            
            results['spec3'] = {
                'coef': fit3.params[treatment_col],
                'se': fit3.bse[treatment_col],
                'pvalue': fit3.pvalues[treatment_col],
                'ci_lower': fit3.conf_int().loc[treatment_col, 0],
                'ci_upper': fit3.conf_int().loc[treatment_col, 1],
                't_stat': fit3.tvalues[treatment_col],
                'n': int(fit3.nobs),
                'r2': fit3.rsquared,
                'baseline_coef': fit3.params[baseline_outcome],
                'stars': '***' if fit3.pvalues[treatment_col] < 0.01 else '**' if fit3.pvalues[treatment_col] < 0.05 else '*' if fit3.pvalues[treatment_col] < 0.1 else ''
            }
        except Exception as e:
            results['spec3'] = {'error': str(e)}
    
    return results


def estimate_tot(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    instrument_col: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None,
    strata_cols: Optional[List[str]] = None
) -> Dict:
    """
    Estimate Treatment-on-Treated (TOT) effect using Two-Stage Least Squares (2SLS/IV)
    
    TOT estimates the effect of actually receiving treatment (for those who took it up),
    using random assignment as an instrument for take-up.
    
    First stage: D = π₀ + π₁Z + γX + u  (take-up on assignment + controls)
    Second stage: Y = β₀ + τD_hat + δX + ε  (outcome on predicted take-up + controls)
    
    Where:
    - D = actual treatment take-up (treatment_col)
    - Z = random assignment (instrument_col)
    - Y = outcome
    - X = covariates
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of actual treatment/take-up variable (endogenous)
        instrument_col: Name of instrument (random assignment)
        covariates: Optional list of control variables
        cluster_col: Optional cluster variable for robust SEs
        strata_cols: Optional stratification variables
    
    Returns:
        Dictionary with:
        {
            'first_stage': {'coef': ..., 'se': ..., 'pvalue': ..., 'f_stat': ..., 'n': ...},
            'second_stage': {'coef': ..., 'se': ..., 'pvalue': ..., 'ci_lower': ..., 'ci_upper': ..., 'n': ..., 'r2': ...},
            'outcome_var': str,
            'treatment_col': str,
            'instrument_col': str,
            'compliance_rate': float,
            'weak_instrument': bool (True if F-stat < 10)
        }
    """
    try:
        from linearmodels.iv import IV2SLS
    except ImportError:
        return {'error': 'linearmodels package not installed. Run: pip install linearmodels'}
    
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    covariates = covariates or []
    strata_cols = strata_cols or []
    
    # Prepare data
    required_cols = [outcome_var, treatment_col, instrument_col]
    all_controls = covariates + strata_cols
    valid_controls = [c for c in all_controls if c in data.columns]
    subset_cols = required_cols + valid_controls
    
    subset = data[subset_cols].dropna()
    
    if len(subset) == 0:
        return {'error': 'No valid observations after removing missing values'}
    
    # Calculate compliance rate (% assigned who took up)
    assigned = subset[subset[instrument_col] == 1]
    if len(assigned) > 0:
        compliance_rate = assigned[treatment_col].mean()
    else:
        compliance_rate = np.nan
    
    # ============================================
    # First Stage: D ~ Z + X
    # ============================================
    formula_fs_parts = [treatment_col, '~', instrument_col]
    for cov in covariates:
        if cov in subset.columns:
            formula_fs_parts.append(f'+ {cov}')
    for strata in strata_cols:
        if strata in subset.columns:
            formula_fs_parts.append(f'+ C({strata})')
    
    formula_fs = ' '.join(formula_fs_parts)
    
    try:
        model_fs = ols(formula_fs, data=subset)
        
        # Cluster-robust SE if specified
        if cluster_col and cluster_col in data.columns:
            fit_fs = model_fs.fit(cov_type='cluster', cov_kwds={'groups': subset[cluster_col]})
        else:
            fit_fs = model_fs.fit(cov_type='HC1')
        
        # F-statistic for instrument strength (should be > 10)
        f_stat = fit_fs.fvalue
        
        first_stage = {
            'coef': fit_fs.params[instrument_col],
            'se': fit_fs.bse[instrument_col],
            'pvalue': fit_fs.pvalues[instrument_col],
            't_stat': fit_fs.tvalues[instrument_col],
            'f_stat': f_stat,
            'n': int(fit_fs.nobs),
            'r2': fit_fs.rsquared,
            'stars': '***' if fit_fs.pvalues[instrument_col] < 0.01 else '**' if fit_fs.pvalues[instrument_col] < 0.05 else '*' if fit_fs.pvalues[instrument_col] < 0.1 else ''
        }
    except Exception as e:
        return {'error': f'First stage failed: {str(e)}'}
    
    # ============================================
    # Second Stage: Y ~ D_hat + X (via 2SLS)
    # ============================================
    try:
        # Build formula for IV2SLS
        # Dependent variable
        y = subset[outcome_var]
        
        # Exogenous variables (controls + constant)
        exog_list = ['1']  # Constant
        for cov in covariates:
            if cov in subset.columns:
                exog_list.append(cov)
        
        # Handle categorical strata
        if len(strata_cols) > 0:
            strata_dummies = []
            for strata in strata_cols:
                if strata in subset.columns:
                    dummies = pd.get_dummies(subset[strata], prefix=strata, drop_first=True)
                    strata_dummies.append(dummies)
            if strata_dummies:
                exog_df = pd.concat([subset[exog_list[1:]]] + strata_dummies, axis=1) if len(exog_list) > 1 else pd.concat(strata_dummies, axis=1)
                exog = sm.add_constant(exog_df)
            else:
                exog = sm.add_constant(subset[exog_list[1:]]) if len(exog_list) > 1 else pd.DataFrame({'const': 1}, index=subset.index)
        else:
            if len(exog_list) > 1:
                exog = sm.add_constant(subset[exog_list[1:]])
            else:
                exog = pd.DataFrame({'const': 1}, index=subset.index)
        
        # Endogenous variable (actual treatment)
        endog = subset[[treatment_col]]
        
        # Instrument
        instruments = subset[[instrument_col]]
        
        # Fit IV2SLS
        model_iv = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instruments)
        
        if cluster_col and cluster_col in subset.columns:
            fit_iv = model_iv.fit(cov_type='clustered', clusters=subset[cluster_col])
        else:
            fit_iv = model_iv.fit(cov_type='robust')
        
        second_stage = {
            'coef': fit_iv.params[treatment_col],
            'se': fit_iv.std_errors[treatment_col],
            'pvalue': fit_iv.pvalues[treatment_col],
            't_stat': fit_iv.tstats[treatment_col],
            'ci_lower': fit_iv.conf_int().loc[treatment_col, 'lower'],
            'ci_upper': fit_iv.conf_int().loc[treatment_col, 'upper'],
            'n': int(fit_iv.nobs),
            'r2': fit_iv.rsquared,
            'stars': '***' if fit_iv.pvalues[treatment_col] < 0.01 else '**' if fit_iv.pvalues[treatment_col] < 0.05 else '*' if fit_iv.pvalues[treatment_col] < 0.1 else ''
        }
    except Exception as e:
        return {'error': f'Second stage failed: {str(e)}', 'first_stage': first_stage}
    
    results = {
        'outcome_var': outcome_var,
        'treatment_col': treatment_col,
        'instrument_col': instrument_col,
        'first_stage': first_stage,
        'second_stage': second_stage,
        'compliance_rate': compliance_rate,
        'weak_instrument': f_stat < 10 if f_stat else True
    }
    
    return results


def estimate_late(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    instrument_col: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None,
    strata_cols: Optional[List[str]] = None
) -> Dict:
    """
    Estimate Local Average Treatment Effect (LATE) for compliers
    
    LATE is the effect for "compliers" - those who take up treatment when assigned
    but would not take it up if not assigned. This is identical to TOT but emphasizes
    the complier interpretation.
    
    LATE = ITT / Compliance Rate (Wald estimator)
    
    Also estimated via 2SLS using random assignment as instrument for take-up.
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of actual treatment/take-up variable
        instrument_col: Name of instrument (random assignment)
        covariates: Optional list of control variables
        cluster_col: Optional cluster variable for robust SEs
        strata_cols: Optional stratification variables
    
    Returns:
        Dictionary with:
        {
            'late': float (LATE estimate from 2SLS),
            'late_se': float,
            'late_pvalue': float,
            'compliance_rate': float,
            'wald_estimate': float (ITT / compliance),
            'itt_effect': float,
            'first_stage': {...},
            'second_stage': {...},
            'interpretation': str
        }
    """
    # Run TOT estimation (which is 2SLS)
    tot_results = estimate_tot(
        data=data,
        outcome_var=outcome_var,
        treatment_col=treatment_col,
        instrument_col=instrument_col,
        covariates=covariates,
        cluster_col=cluster_col,
        strata_cols=strata_cols
    )
    
    if 'error' in tot_results:
        return tot_results
    
    # Calculate ITT (reduced form: Y ~ Z)
    subset = data[[outcome_var, instrument_col]].dropna()
    
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    formula_itt = f'{outcome_var} ~ {instrument_col}'
    model_itt = ols(formula_itt, data=subset)
    
    if cluster_col and cluster_col in data.columns:
        fit_itt = model_itt.fit(cov_type='cluster', cov_kwds={'groups': subset[cluster_col]})
    else:
        fit_itt = model_itt.fit(cov_type='HC1')
    
    itt_effect = fit_itt.params[instrument_col]
    
    # Wald estimator: ITT / First Stage
    compliance_rate = tot_results['compliance_rate']
    wald_estimate = itt_effect / compliance_rate if compliance_rate != 0 else np.nan
    
    # LATE interpretation
    interpretation = f"""
    LATE Interpretation:
    - The treatment effect is {tot_results['second_stage']['coef']:.3f} for compliers.
    - Compliers are the {compliance_rate*100:.1f}% who take up treatment when assigned.
    - This assumes monotonicity (no defiers) and excluder restriction (instrument only affects outcome through treatment).
    - ITT effect: {itt_effect:.3f} (effect of assignment on outcome)
    - Compliance rate: {compliance_rate:.3f} (% assigned who took up)
    - LATE = ITT / Compliance = {wald_estimate:.3f}
    """
    
    results = {
        'late': tot_results['second_stage']['coef'],
        'late_se': tot_results['second_stage']['se'],
        'late_pvalue': tot_results['second_stage']['pvalue'],
        'late_ci_lower': tot_results['second_stage']['ci_lower'],
        'late_ci_upper': tot_results['second_stage']['ci_upper'],
        'stars': tot_results['second_stage']['stars'],
        'compliance_rate': compliance_rate,
        'wald_estimate': wald_estimate,
        'itt_effect': itt_effect,
        'first_stage': tot_results['first_stage'],
        'second_stage': tot_results['second_stage'],
        'weak_instrument': tot_results['weak_instrument'],
        'interpretation': interpretation,
        'outcome_var': outcome_var,
        'treatment_col': treatment_col,
        'instrument_col': instrument_col
    }
    
    return results


def estimate_heterogeneity(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    subgroup_var: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None,
    strata_cols: Optional[List[str]] = None,
    continuous_split: str = 'median'
) -> Dict:
    """
    Estimate heterogeneous treatment effects by subgroup
    
    Runs regression: Y ~ T + S + T*S + X
    Where T = treatment, S = subgroup, T*S = interaction
    
    For continuous moderators, splits at median (or mean) to create binary subgroups.
    
    Tests joint significance of interaction terms.
    Returns subgroup-specific treatment effects.
    
    Pattern from Bruhn, Karlan & Schoar heterogeneity analysis.
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of treatment column
        subgroup_var: Name of subgroup/moderator variable
        covariates: Optional list of control variables
        cluster_col: Optional cluster variable for robust SEs
        strata_cols: Optional stratification variables
        continuous_split: For continuous variables: 'median' or 'mean'
    
    Returns:
        Dictionary with:
        {
            'subgroups': {
                'subgroup1': {'effect': ..., 'se': ..., 'pvalue': ..., 'n': ...},
                'subgroup2': {...},
                ...
            },
            'interaction_test': {'f_stat': ..., 'p_value': ..., 'significant': bool},
            'main_effect': {'coef': ..., 'se': ..., 'pvalue': ...},
            'subgroup_var': str,
            'subgroup_type': 'categorical' or 'continuous',
            'is_heterogeneous': bool
        }
    """
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    covariates = covariates or []
    strata_cols = strata_cols or []
    
    # Prepare data
    required_cols = [outcome_var, treatment_col, subgroup_var]
    valid_controls = [c for c in covariates + strata_cols if c in data.columns]
    subset_cols = required_cols + valid_controls
    subset = data[subset_cols].dropna()
    
    if len(subset) == 0:
        return {'error': 'No valid observations after removing missing values'}
    
    # Determine if subgroup variable is continuous or categorical
    is_continuous = pd.api.types.is_numeric_dtype(subset[subgroup_var]) and subset[subgroup_var].nunique() > 10
    
    subgroup_type = 'continuous' if is_continuous else 'categorical'
    
    # For continuous variables, create binary split
    if is_continuous:
        if continuous_split == 'median':
            split_value = subset[subgroup_var].median()
        else:
            split_value = subset[subgroup_var].mean()
        
        subset[f'{subgroup_var}_binary'] = (subset[subgroup_var] >= split_value).astype(int)
        subgroup_var_formula = f'{subgroup_var}_binary'
    else:
        subgroup_var_formula = f'C({subgroup_var})'
    
    # Build formula with interaction: Y ~ T + S + T*S + X
    formula_parts = [outcome_var, '~', treatment_col, '+', subgroup_var_formula, '+', f'{treatment_col}:{subgroup_var_formula}']
    
    for cov in covariates:
        if cov in subset.columns:
            formula_parts.append(f'+ {cov}')
    
    for strata in strata_cols:
        if strata in subset.columns:
            formula_parts.append(f'+ C({strata})')
    
    formula = ' '.join(formula_parts)
    
    # Fit model
    try:
        model = ols(formula, data=subset)
        
        if cluster_col and cluster_col in data.columns:
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': subset[cluster_col]})
        else:
            fit = model.fit(cov_type='HC1')
    except Exception as e:
        return {'error': f'Model fitting failed: {str(e)}'}
    
    # Extract main treatment effect
    main_effect = {
        'coef': fit.params[treatment_col],
        'se': fit.bse[treatment_col],
        'pvalue': fit.pvalues[treatment_col],
        't_stat': fit.tvalues[treatment_col],
        'stars': '***' if fit.pvalues[treatment_col] < 0.01 else '**' if fit.pvalues[treatment_col] < 0.05 else '*' if fit.pvalues[treatment_col] < 0.1 else ''
    }
    
    # Test joint significance of interaction terms
    # Find all interaction parameters
    interaction_params = [p for p in fit.params.index if ':' in str(p) and treatment_col in str(p)]
    
    if len(interaction_params) > 0:
        try:
            # F-test for joint significance
            hypotheses = [f'{param} = 0' for param in interaction_params]
            f_test = fit.f_test(hypotheses)
            
            interaction_test = {
                'f_stat': f_test.fvalue[0][0],
                'p_value': f_test.pvalue,
                'significant': f_test.pvalue < 0.05,
                'df_num': f_test.df_num,
                'df_denom': f_test.df_denom
            }
        except:
            interaction_test = {'error': 'Could not compute F-test'}
    else:
        interaction_test = {'error': 'No interaction terms found'}
    
    # Calculate subgroup-specific effects
    subgroups = {}
    
    if is_continuous:
        # Binary split: below and above split value
        for subgroup_val, label in [(0, f'Below {continuous_split}'), (1, f'Above {continuous_split}')]:
            subgroup_data = subset[subset[f'{subgroup_var}_binary'] == subgroup_val]
            
            if len(subgroup_data) > 0:
                # Treatment effect = main effect + interaction (if subgroup=1)
                if subgroup_val == 0:
                    effect = fit.params[treatment_col]
                    se = fit.bse[treatment_col]
                else:
                    # Need to compute combined effect
                    interaction_param = [p for p in interaction_params if subgroup_var_formula in str(p)]
                    if interaction_param:
                        effect = fit.params[treatment_col] + fit.params[interaction_param[0]]
                        # Delta method for SE of linear combination
                        cov_matrix = fit.cov_params()
                        var_effect = cov_matrix.loc[treatment_col, treatment_col] + \
                                   cov_matrix.loc[interaction_param[0], interaction_param[0]] + \
                                   2 * cov_matrix.loc[treatment_col, interaction_param[0]]
                        se = np.sqrt(var_effect)
                    else:
                        effect = fit.params[treatment_col]
                        se = fit.bse[treatment_col]
                
                pvalue = 2 * (1 - scipy_stats.t.cdf(abs(effect/se), fit.df_resid))
                
                subgroups[label] = {
                    'effect': effect,
                    'se': se,
                    'pvalue': pvalue,
                    't_stat': effect/se,
                    'n': len(subgroup_data),
                    'stars': '***' if pvalue < 0.01 else '**' if pvalue < 0.05 else '*' if pvalue < 0.1 else ''
                }
    else:
        # Categorical subgroups
        unique_vals = sorted(subset[subgroup_var].unique())
        
        for subgroup_val in unique_vals:
            subgroup_data = subset[subset[subgroup_var] == subgroup_val]
            
            if len(subgroup_data) > 0:
                # Run separate regression for each subgroup
                formula_subgroup = f'{outcome_var} ~ {treatment_col}'
                for cov in covariates:
                    if cov in subgroup_data.columns:
                        formula_subgroup += f' + {cov}'
                
                try:
                    model_sub = ols(formula_subgroup, data=subgroup_data)
                    if cluster_col and cluster_col in subgroup_data.columns:
                        fit_sub = model_sub.fit(cov_type='cluster', cov_kwds={'groups': subgroup_data[cluster_col]})
                    else:
                        fit_sub = model_sub.fit(cov_type='HC1')
                    
                    subgroups[str(subgroup_val)] = {
                        'effect': fit_sub.params[treatment_col],
                        'se': fit_sub.bse[treatment_col],
                        'pvalue': fit_sub.pvalues[treatment_col],
                        't_stat': fit_sub.tvalues[treatment_col],
                        'n': len(subgroup_data),
                        'stars': '***' if fit_sub.pvalues[treatment_col] < 0.01 else '**' if fit_sub.pvalues[treatment_col] < 0.05 else '*' if fit_sub.pvalues[treatment_col] < 0.1 else ''
                    }
                except:
                    subgroups[str(subgroup_val)] = {'error': 'Estimation failed for this subgroup'}
    
    results = {
        'subgroups': subgroups,
        'interaction_test': interaction_test,
        'main_effect': main_effect,
        'subgroup_var': subgroup_var,
        'subgroup_type': subgroup_type,
        'is_heterogeneous': interaction_test.get('significant', False),
        'n_total': len(subset)
    }
    
    return results


def estimate_binary_outcome(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None,
    strata_cols: Optional[List[str]] = None,
    model_type: str = 'logit'
) -> Dict:
    """
    Estimate treatment effects for binary outcomes using Logit or Probit
    
    For binary outcomes (0/1), linear probability model can give predictions
    outside [0,1] and heteroskedastic errors. Logit/Probit are preferred.
    
    Reports both coefficients and marginal effects (average partial effects).
    
    Pattern from Attanasio et al (2011) Risk Pooling study.
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of binary outcome variable (0/1)
        treatment_col: Name of treatment column
        covariates: Optional list of control variables
        cluster_col: Optional cluster variable for robust SEs
        strata_cols: Optional stratification variables
        model_type: 'logit' or 'probit'
    
    Returns:
        Dictionary with:
        {
            'coef': float (coefficient),
            'se': float (standard error of coefficient),
            'pvalue': float,
            'marginal_effect': float (average partial effect),
            'mfx_se': float (SE of marginal effect),
            'mfx_pvalue': float,
            'pseudo_r2': float,
            'n': int,
            'outcome_var': str,
            'treatment_col': str,
            'model_type': str
        }
    """
    import numpy as np
    from statsmodels.discrete.discrete_model import Logit, Probit
    import statsmodels.api as sm
    
    covariates = covariates or []
    strata_cols = strata_cols or []
    
    # Prepare data
    required_cols = [outcome_var, treatment_col]
    valid_controls = [c for c in covariates + strata_cols if c in data.columns]
    subset_cols = required_cols + valid_controls
    subset = data[subset_cols].dropna()
    
    if len(subset) == 0:
        return {'error': 'No valid observations after removing missing values'}
    
    # Check if outcome is binary
    if not subset[outcome_var].isin([0, 1]).all():
        return {'error': f'Outcome variable {outcome_var} must be binary (0/1)'}
    
    # Build design matrix
    exog_cols = [treatment_col] + covariates
    
    # Handle categorical strata
    if len(strata_cols) > 0:
        strata_dummies = []
        for strata in strata_cols:
            if strata in subset.columns:
                dummies = pd.get_dummies(subset[strata], prefix=strata, drop_first=True)
                strata_dummies.append(dummies)
        if strata_dummies:
            exog_df = pd.concat([subset[exog_cols]] + strata_dummies, axis=1)
            X = sm.add_constant(exog_df)
        else:
            X = sm.add_constant(subset[exog_cols])
    else:
        X = sm.add_constant(subset[exog_cols])
    
    y = subset[outcome_var]
    
    # Fit model
    try:
        if model_type.lower() == 'probit':
            model = Probit(y, X)
        else:
            model = Logit(y, X)
        
        if cluster_col and cluster_col in data.columns:
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': subset[cluster_col]}, disp=0)
        else:
            fit = model.fit(cov_type='HC1', disp=0)
        
        # Get marginal effects (average partial effects)
        mfx = fit.get_margeff(at='overall', method='dydx')
        
        results = {
            'coef': fit.params[treatment_col],
            'se': fit.bse[treatment_col],
            'pvalue': fit.pvalues[treatment_col],
            'ci_lower': fit.conf_int().loc[treatment_col, 0],
            'ci_upper': fit.conf_int().loc[treatment_col, 1],
            't_stat': fit.tvalues[treatment_col],
            'marginal_effect': mfx.margeff[list(X.columns).index(treatment_col) - 1],  # -1 for constant
            'mfx_se': mfx.margeff_se[list(X.columns).index(treatment_col) - 1],
            'mfx_pvalue': mfx.pvalues[list(X.columns).index(treatment_col) - 1],
            'pseudo_r2': fit.prsquared,
            'n': int(fit.nobs),
            'stars': '***' if fit.pvalues[treatment_col] < 0.01 else '**' if fit.pvalues[treatment_col] < 0.05 else '*' if fit.pvalues[treatment_col] < 0.1 else '',
            'outcome_var': outcome_var,
            'treatment_col': treatment_col,
            'model_type': model_type
        }
        
        return results
        
    except Exception as e:
        return {'error': f'{model_type.capitalize()} estimation failed: {str(e)}'}


def format_regression_table(
    results_list: List[Dict],
    decimals: int = 3,
    include_stars: bool = True,
    include_ci: bool = False
) -> pd.DataFrame:
    """
    Format regression results into publication-ready table
    
    Creates Stata-style table with coefficients, standard errors in parentheses,
    and significance stars.
    
    Args:
        results_list: List of result dictionaries from estimation functions
        decimals: Number of decimal places
        include_stars: Whether to include significance stars
        include_ci: Whether to include confidence intervals
    
    Returns:
        DataFrame formatted for display/export
    """
    import pandas as pd
    
    rows = []
    
    for i, res in enumerate(results_list, 1):
        if 'error' in res:
            rows.append({
                'Variable': res.get('outcome_var', f'Model {i}'),
                f'Model {i}': f"Error: {res['error']}"
            })
            continue
        
        # Coefficient row
        coef = res.get('coef', res.get('late', res.get('marginal_effect')))
        stars = res.get('stars', '') if include_stars else ''
        coef_str = f"{coef:.{decimals}f}{stars}"
        
        # SE row (in parentheses)
        se = res.get('se', res.get('late_se', res.get('mfx_se')))
        se_str = f"({se:.{decimals}f})"
        
        # Create row dict
        row_coef = {
            'Variable': res.get('treatment_col', 'Treatment'),
            f'Model {i}': coef_str
        }
        row_se = {
            'Variable': '',
            f'Model {i}': se_str
        }
        
        rows.append(row_coef)
        rows.append(row_se)
        
        # Add confidence interval if requested
        if include_ci and 'ci_lower' in res:
            ci_str = f"[{res['ci_lower']:.{decimals}f}, {res['ci_upper']:.{decimals}f}]"
            row_ci = {
                'Variable': '',
                f'Model {i}': ci_str
            }
            rows.append(row_ci)
        
        # Add summary statistics
        rows.append({
            'Variable': 'N',
            f'Model {i}': str(res.get('n', ''))
        })
        
        if 'r2' in res:
            rows.append({
                'Variable': 'R²',
                f'Model {i}': f"{res['r2']:.{decimals}f}"
            })
        elif 'pseudo_r2' in res:
            rows.append({
                'Variable': 'Pseudo R²',
                f'Model {i}': f"{res['pseudo_r2']:.{decimals}f}"
            })
    
    df = pd.DataFrame(rows)
    
    # Add notes
    if include_stars:
        notes = "Notes: Standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01"
        df = pd.concat([df, pd.DataFrame([{'Variable': notes, **{f'Model {i}': '' for i in range(1, len(results_list)+1)}}])], ignore_index=True)
    
    return df


def estimate_panel_fe(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    panel_id: str,
    time_var: str,
    covariates: Optional[List[str]] = None,
    cluster_col: Optional[str] = None
) -> Dict:
    """
    Estimate treatment effects with panel fixed effects (within estimator)
    
    Useful for panel/longitudinal RCT data where you have multiple observations
    per unit over time. Fixed effects control for time-invariant unobservables.
    
    Model: Y_it = β₀ + τT_it + γX_it + α_i + ε_it
    Where α_i is the unit fixed effect
    
    Pattern from Referrals study: xtreg Y treatment X, fe i(panel_id)
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of treatment column
        panel_id: Name of panel identifier (e.g., individual, household, firm ID)
        time_var: Name of time variable
        covariates: Optional list of time-varying control variables
        cluster_col: Optional cluster variable for robust SEs
    
    Returns:
        Dictionary with:
        {
            'coef': float,
            'se': float,
            'pvalue': float,
            'ci_lower': float,
            'ci_upper': float,
            't_stat': float,
            'n': int,
            'n_groups': int,
            'r2_within': float,
            'r2_between': float,
            'r2_overall': float,
            'outcome_var': str,
            'treatment_col': str
        }
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        return {'error': 'linearmodels package not installed. Run: pip install linearmodels'}
    
    import numpy as np
    
    covariates = covariates or []
    
    # Prepare data
    required_cols = [outcome_var, treatment_col, panel_id, time_var]
    valid_controls = [c for c in covariates if c in data.columns]
    subset_cols = required_cols + valid_controls
    
    subset = data[subset_cols].dropna()
    
    if len(subset) == 0:
        return {'error': 'No valid observations after removing missing values'}
    
    # Set multi-index for panel data
    subset = subset.set_index([panel_id, time_var])
    
    # Dependent variable
    y = subset[outcome_var]
    
    # Independent variables
    exog_cols = [treatment_col] + valid_controls
    X = subset[exog_cols]
    
    # Fit panel FE model
    try:
        model = PanelOLS(y, X, entity_effects=True)
        
        if cluster_col and cluster_col in data.columns:
            # Cluster at entity level (common for panel data)
            fit = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            fit = model.fit(cov_type='robust')
        
        results = {
            'coef': fit.params[treatment_col],
            'se': fit.std_errors[treatment_col],
            'pvalue': fit.pvalues[treatment_col],
            'ci_lower': fit.conf_int().loc[treatment_col, 'lower'],
            'ci_upper': fit.conf_int().loc[treatment_col, 'upper'],
            't_stat': fit.tstats[treatment_col],
            'n': int(fit.nobs),
            'n_groups': int(fit.entity_info['total']),
            'r2_within': fit.rsquared_within,
            'r2_between': fit.rsquared_between,
            'r2_overall': fit.rsquared_overall,
            'stars': '***' if fit.pvalues[treatment_col] < 0.01 else '**' if fit.pvalues[treatment_col] < 0.05 else '*' if fit.pvalues[treatment_col] < 0.1 else '',
            'outcome_var': outcome_var,
            'treatment_col': treatment_col,
            'panel_id': panel_id
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Panel FE estimation failed: {str(e)}'}


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

