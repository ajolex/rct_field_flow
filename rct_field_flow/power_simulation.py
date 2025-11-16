"""
Power Calculation via Simulation for RCT Field Flow
Monte Carlo simulation approach based on J-PAL methodologies
Supports individual and cluster randomization with flexible error distributions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SimulationAssumptions:
    """Container for simulation-based power calculation assumptions"""
    
    # Core parameters
    alpha: float = 0.05  # Significance level
    test_type: Literal["two", "left", "right"] = "two"  # Test type
    effect_size: float = 0.2  # Hypothesized treatment effect (absolute)
    
    # Outcome parameters
    outcome_type: Literal["continuous", "binary"] = "continuous"
    control_mean: float = 10.0  # Mean outcome for control group
    control_sd: float | None = None  # SD for continuous outcomes (auto-calculated for binary)
    
    # Study design
    design_type: Literal["individual", "cluster"] = "individual"
    treatment_share: float = 0.5  # Proportion assigned to treatment
    
    # Sample size
    sample_size: int | None = None  # Total individuals (for individual design)
    num_clusters: int | None = None  # Number of clusters (for cluster design)
    cluster_size: int | None = None  # Individuals per cluster (for cluster design)
    
    # Cluster parameters (if applicable)
    icc: float = 0.0  # Intracluster correlation
    within_cluster_var: float = 1.0  # Within-cluster variance
    
    # Simulation settings
    num_simulations: int = 1000  # Number of Monte Carlo iterations
    seed: int = 123456  # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate and calculate derived parameters"""
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        if self.treatment_share <= 0 or self.treatment_share >= 1:
            raise ValueError("Treatment share must be between 0 and 1")
        
        # Binary outcome specific validations
        if self.outcome_type == "binary":
            if not 0 < self.control_mean < 1:
                raise ValueError("For binary outcomes, control mean must be a proportion between 0 and 1")
            # Auto-calculate SD for binary outcome
            self.control_sd = math.sqrt(self.control_mean * (1 - self.control_mean))
        else:
            if self.control_sd is None:
                raise ValueError("SD required for continuous outcomes")
            if self.control_sd <= 0:
                raise ValueError("SD must be positive")
        
        # Design-specific validations
        if self.design_type == "individual":
            if self.sample_size is None or self.sample_size <= 0:
                raise ValueError("Sample size required for individual design")
        else:  # cluster
            if self.num_clusters is None or self.cluster_size is None:
                raise ValueError("Number of clusters and cluster size required for cluster design")
            if self.num_clusters <= 0 or self.cluster_size <= 0:
                raise ValueError("Clusters and cluster size must be positive")
            if self.icc < 0 or self.icc > 1:
                raise ValueError("ICC must be between 0 and 1")
            # Calculate between-cluster variance from ICC
            self.between_cluster_var = (self.icc * self.within_cluster_var) / (1 - self.icc) if self.icc < 1 else 0


def run_individual_simulation(assumptions: SimulationAssumptions) -> dict:
    """
    Run Monte Carlo simulation for individual randomization
    
    Returns:
        dict with power estimate and simulation statistics
    """
    np.random.seed(assumptions.seed)
    
    n = assumptions.sample_size
    effect = assumptions.effect_size
    prop = assumptions.treatment_share
    alpha = assumptions.alpha
    test_type = assumptions.test_type
    num_sims = assumptions.num_simulations
    
    # Storage for results
    reject_count = 0
    p_values = []
    t_values = []
    effect_estimates = []
    
    for sim in range(num_sims):
        # Generate treatment assignment
        treatment = np.random.uniform(0, 1, n) < prop
        
        # Generate individual errors
        if assumptions.outcome_type == "continuous":
            errors = np.random.normal(0, assumptions.control_sd, n)
            outcome = assumptions.control_mean + errors
        else:  # binary
            # For binary, simulate from Bernoulli
            outcome = np.random.binomial(1, assumptions.control_mean, n).astype(float)
        
        # Apply treatment effect
        outcome[treatment] += effect
        
        # For binary outcomes, clip to [0, 1] range
        if assumptions.outcome_type == "binary":
            outcome = np.clip(outcome, 0, 1)
        
        # Run regression (OLS)
        X = np.column_stack([np.ones(n), treatment.astype(float)])
        y = outcome
        
        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            
            # Residuals and standard errors
            residuals = y - X @ beta
            mse = np.sum(residuals**2) / (n - 2)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_treatment = np.sqrt(var_beta[1, 1])
            
            # T-statistic
            t_stat = beta[1] / se_treatment
            df = n - 2
            
            # P-value and rejection decision
            if test_type == "two":
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                reject = p_val < alpha
            elif test_type == "left":
                p_val = stats.t.cdf(t_stat, df)
                reject = p_val < alpha
            else:  # right
                p_val = 1 - stats.t.cdf(t_stat, df)
                reject = p_val < alpha
            
            if reject:
                reject_count += 1
            
            p_values.append(p_val)
            t_values.append(t_stat)
            effect_estimates.append(beta[1])
            
        except np.linalg.LinAlgError:
            # Singular matrix, skip this simulation
            continue
    
    # Calculate power
    power = reject_count / num_sims
    
    return {
        "power": power,
        "num_simulations": num_sims,
        "rejections": reject_count,
        "mean_effect_estimate": np.mean(effect_estimates),
        "sd_effect_estimate": np.std(effect_estimates),
        "mean_t_statistic": np.mean(t_values),
        "mean_p_value": np.mean(p_values),
        "sample_size": n,
        "design_type": "individual",
    }


def run_cluster_simulation(assumptions: SimulationAssumptions) -> dict:
    """
    Run Monte Carlo simulation for cluster randomization
    
    Returns:
        dict with power estimate and simulation statistics
    """
    np.random.seed(assumptions.seed)
    
    k = assumptions.num_clusters
    m = assumptions.cluster_size
    n = k * m
    effect = assumptions.effect_size
    prop = assumptions.treatment_share
    alpha = assumptions.alpha
    test_type = assumptions.test_type
    num_sims = assumptions.num_simulations
    
    # Storage for results
    reject_count = 0
    p_values = []
    t_values = []
    effect_estimates = []
    icc_estimates = []
    
    for sim in range(num_sims):
        # Generate cluster-level treatment assignment
        cluster_treat = np.random.uniform(0, 1, k) < prop
        
        # Generate cluster-specific errors
        cluster_errors = np.random.normal(0, np.sqrt(assumptions.between_cluster_var), k)
        
        # Generate individual data
        cluster_ids = []
        outcomes = []
        treatments = []
        
        for cluster_idx in range(k):
            # Individual errors for this cluster
            individual_errors = np.random.normal(0, np.sqrt(assumptions.within_cluster_var), m)
            
            # Base outcome (control)
            if assumptions.outcome_type == "continuous":
                cluster_outcome = (assumptions.control_mean + 
                                 cluster_errors[cluster_idx] + 
                                 individual_errors)
            else:  # binary
                # For binary with clusters, use cluster error to shift probability
                prob = assumptions.control_mean + cluster_errors[cluster_idx] * 0.1  # Scale cluster effect
                prob = np.clip(prob, 0.01, 0.99)  # Keep in valid range
                cluster_outcome = np.random.binomial(1, prob, m).astype(float)
            
            # Apply treatment effect
            if cluster_treat[cluster_idx]:
                cluster_outcome += effect
            
            # For binary, clip to [0, 1]
            if assumptions.outcome_type == "binary":
                cluster_outcome = np.clip(cluster_outcome, 0, 1)
            
            # Store data
            cluster_ids.extend([cluster_idx] * m)
            outcomes.extend(cluster_outcome)
            treatments.extend([int(cluster_treat[cluster_idx])] * m)
        
        # Convert to arrays
        cluster_ids = np.array(cluster_ids)
        outcomes = np.array(outcomes)
        treatments = np.array(treatments)
        
        # Estimate ICC on control group
        control_mask = treatments == 0
        if np.sum(control_mask) > k:  # Need sufficient data
            try:
                # Calculate ICC using one-way ANOVA components
                control_outcomes = outcomes[control_mask]
                control_clusters = cluster_ids[control_mask]
                
                # Grand mean
                grand_mean = np.mean(control_outcomes)
                
                # Between-cluster variance
                cluster_means = []
                cluster_sizes = []
                for c in np.unique(control_clusters):
                    c_data = control_outcomes[control_clusters == c]
                    cluster_means.append(np.mean(c_data))
                    cluster_sizes.append(len(c_data))
                
                cluster_means = np.array(cluster_means)
                cluster_sizes = np.array(cluster_sizes)
                
                ms_between = np.sum(cluster_sizes * (cluster_means - grand_mean)**2) / (len(cluster_means) - 1)
                
                # Within-cluster variance
                ms_within = 0
                for c in np.unique(control_clusters):
                    c_data = control_outcomes[control_clusters == c]
                    c_mean = np.mean(c_data)
                    ms_within += np.sum((c_data - c_mean)**2)
                ms_within = ms_within / (len(control_outcomes) - len(cluster_means))
                
                # ICC estimate
                avg_cluster_size = np.mean(cluster_sizes)
                icc_est = (ms_between - ms_within) / (ms_between + (avg_cluster_size - 1) * ms_within)
                icc_est = max(0, min(1, icc_est))  # Bound to [0, 1]
                icc_estimates.append(icc_est)
            except Exception:
                icc_estimates.append(np.nan)
        
        # Run regression with clustered standard errors
        X = np.column_stack([np.ones(n), treatments])
        y = outcomes
        
        try:
            # OLS coefficient
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            
            # Clustered standard errors
            residuals = y - X @ beta
            
            # Cluster-robust variance
            bread = np.linalg.inv(X.T @ X)
            
            # Meat: sum of cluster-level outer products
            meat = np.zeros((2, 2))
            for c in range(k):
                cluster_mask = cluster_ids == c
                X_c = X[cluster_mask]
                resid_c = residuals[cluster_mask]
                meat += (X_c.T @ resid_c).reshape(-1, 1) @ (X_c.T @ resid_c).reshape(1, -1)
            
            # Finite sample adjustment
            vcov = (k / (k - 1)) * bread @ meat @ bread
            se_treatment = np.sqrt(vcov[1, 1])
            
            # T-statistic with cluster-based df
            t_stat = beta[1] / se_treatment
            df = 2 * (k - 1)  # Conservative df for clustered design
            
            # P-value and rejection
            if test_type == "two":
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                reject = p_val < alpha
            elif test_type == "left":
                p_val = stats.t.cdf(t_stat, df)
                reject = p_val < alpha
            else:  # right
                p_val = 1 - stats.t.cdf(t_stat, df)
                reject = p_val < alpha
            
            if reject:
                reject_count += 1
            
            p_values.append(p_val)
            t_values.append(t_stat)
            effect_estimates.append(beta[1])
            
        except np.linalg.LinAlgError:
            continue
    
    # Calculate power
    power = reject_count / num_sims
    
    # Design effect
    deff = 1 + (m - 1) * assumptions.icc
    
    return {
        "power": power,
        "num_simulations": num_sims,
        "rejections": reject_count,
        "mean_effect_estimate": np.mean(effect_estimates),
        "sd_effect_estimate": np.std(effect_estimates),
        "mean_t_statistic": np.mean(t_values),
        "mean_p_value": np.mean(p_values),
        "num_clusters": k,
        "cluster_size": m,
        "total_sample": n,
        "design_effect": deff,
        "specified_icc": assumptions.icc,
        "mean_estimated_icc": np.nanmean(icc_estimates) if icc_estimates else np.nan,
        "design_type": "cluster",
    }


def run_power_simulation(assumptions: SimulationAssumptions) -> dict:
    """
    Run power calculation via Monte Carlo simulation
    
    Main entry point that dispatches to appropriate simulation type
    """
    if assumptions.design_type == "individual":
        return run_individual_simulation(assumptions)
    else:
        return run_cluster_simulation(assumptions)


def generate_simulation_power_curve(
    assumptions: SimulationAssumptions,
    sample_sizes: list[int] | None = None,
    num_points: int = 8
) -> pd.DataFrame:
    """
    Generate power curve using simulation across different sample sizes
    
    Args:
        assumptions: Base simulation assumptions
        sample_sizes: List of sample sizes to test (auto-generated if None)
        num_points: Number of points in curve if auto-generating
    
    Returns:
        DataFrame with sample_size and simulated power
    """
    if sample_sizes is None:
        if assumptions.design_type == "cluster":
            base = assumptions.num_clusters or 20
            sample_sizes = np.linspace(max(10, base // 2), base * 2, num_points, dtype=int).tolist()
        else:
            base = assumptions.sample_size or 500
            sample_sizes = np.linspace(max(50, base // 2), base * 2, num_points, dtype=int).tolist()
    
    results = []
    
    for n in sample_sizes:
        if assumptions.design_type == "cluster":
            temp_assumptions = SimulationAssumptions(
                alpha=assumptions.alpha,
                test_type=assumptions.test_type,
                effect_size=assumptions.effect_size,
                outcome_type=assumptions.outcome_type,
                control_mean=assumptions.control_mean,
                control_sd=assumptions.control_sd,
                design_type="cluster",
                treatment_share=assumptions.treatment_share,
                num_clusters=n,
                cluster_size=assumptions.cluster_size,
                icc=assumptions.icc,
                within_cluster_var=assumptions.within_cluster_var,
                num_simulations=assumptions.num_simulations,
                seed=assumptions.seed + n,  # Different seed for each point
            )
            sim_result = run_cluster_simulation(temp_assumptions)
            total_n = n * assumptions.cluster_size
            clusters = n
        else:
            temp_assumptions = SimulationAssumptions(
                alpha=assumptions.alpha,
                test_type=assumptions.test_type,
                effect_size=assumptions.effect_size,
                outcome_type=assumptions.outcome_type,
                control_mean=assumptions.control_mean,
                control_sd=assumptions.control_sd,
                design_type="individual",
                treatment_share=assumptions.treatment_share,
                sample_size=n,
                num_simulations=assumptions.num_simulations,
                seed=assumptions.seed + n,
            )
            sim_result = run_individual_simulation(temp_assumptions)
            total_n = n
            clusters = None
        
        results.append({
            "sample_size": total_n,
            "clusters": clusters,
            "power": sim_result["power"],
            "mean_effect": sim_result["mean_effect_estimate"],
        })
    
    return pd.DataFrame(results)


def generate_simulation_stata_code(assumptions: SimulationAssumptions, results: dict) -> str:
    """Generate Stata code for simulation-based power calculation"""
    
    code = f"""* Power by Simulation - Generated by RCT Field Flow
* Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
* Based on J-PAL Power Calculation Methodologies

clear all
set more off

* ========================================
* SPECIFY DESIGN PARAMETERS
* ========================================
local effect = {assumptions.effect_size}          // Hypothesized treatment effect
local prop = {assumptions.treatment_share}        // Treatment allocation ratio
local alpha = {assumptions.alpha}                 // Significance level
local side "{assumptions.test_type}"             // Test type: two, left, or right
local sims = {assumptions.num_simulations}        // Number of simulations

"""
    
    if assumptions.design_type == "individual":
        code += f"""local sample_size = {assumptions.sample_size}    // Total sample size
local control_mean = {assumptions.control_mean}  // Control group mean
local control_sd = {assumptions.control_sd}      // Control group SD

* ========================================
* SIMULATION SETUP
* ========================================
tempname sim_name
tempfile sim_results
postfile `sim_name' reject_t using `sim_results'

clear
set seed {assumptions.seed}

* ========================================
* RUN SIMULATIONS
* ========================================
local it = 1
while `it' <= `sims' {{
    clear
    quietly set obs `sample_size'
    
    * Random treatment assignment
    quietly gen treat = runiform() < `prop'
    
    * Generate outcome
    """
        
        if assumptions.outcome_type == "continuous":
            code += """drawnorm error, m(0) sd(`control_sd')
    quietly gen outcome = `control_mean' + error
    """
        else:
            code += """quietly gen outcome = rbinomial(1, `control_mean')
    """
        
        code += """
    * Apply treatment effect
    quietly replace outcome = outcome + `effect' if treat
    
    * Regression
    quietly regress outcome treat
    
    * Calculate t-statistic and test
    local t_value = _b[treat] / _se[treat]
    local df = `sample_size' - 2
    
    if "`side'" == "two" {
        local critical_l = invt(`df', `alpha'/2)
        local critical_u = invt(`df', 1 - `alpha'/2)
        local reject_t = (`t_value' > `critical_u') | (`t_value' < `critical_l')
    }
    else if "`side'" == "left" {
        local critical_l = invt(`df', `alpha')
        local reject_t = (`t_value' < `critical_l')
    }
    else if "`side'" == "right" {
        local critical_u = invt(`df', 1 - `alpha')
        local reject_t = (`t_value' > `critical_u')
    }
    
    post `sim_name' (`reject_t')
    local it = `it' + 1
}

* ========================================
* CALCULATE POWER
* ========================================
postclose `sim_name'
use `sim_results', clear

sum reject_t

di ""
di _dup(70) "="
di "SIMULATION RESULTS"
di _dup(70) "="
di "Sample size: `sample_size'"
di "Effect size: `effect'"
di "Number of simulations: `sims'"
di "Estimated power: " %5.3f r(mean)
di _dup(70) "="
"""
    
    else:  # cluster design
        code += f"""local num_clusters = {assumptions.num_clusters}    // Number of clusters
local cluster_size = {assumptions.cluster_size}  // Individuals per cluster
local sample_size = `num_clusters' * `cluster_size'
local control_mean = {assumptions.control_mean}  // Control intercept
local icc = {assumptions.icc}                    // Intracluster correlation
local within_var = {assumptions.within_cluster_var}  // Within-cluster variance
local between_var = (`icc' * `within_var') / (1 - `icc')  // Between-cluster variance

* ========================================
* SIMULATION SETUP
* ========================================
tempname sim_name
tempfile sim_results
postfile `sim_name' reject_t rho_calculated using `sim_results'

clear
set seed {assumptions.seed}

* ========================================
* RUN SIMULATIONS
* ========================================
local it = 1
while `it' <= `sims' {{
    clear
    
    * Generate cluster-level data
    quietly set obs `num_clusters'
    quietly gen cluster_group = _n
    quietly gen cluster_error = rnormal(0, sqrt(`between_var'))
    quietly gen treat = runiform() < `prop'
    
    sort cluster_group
    tempfile cluster_data
    quietly save `cluster_data', replace
    
    * Generate individual data
    clear
    quietly set obs `sample_size'
    quietly gen u = invnormal(uniform())
    quietly egen cluster_group = cut(u), group(`num_clusters')
    quietly replace cluster_group = cluster_group + 1
    sort cluster_group
    
    * Merge cluster characteristics
    quietly merge m:1 cluster_group using `cluster_data'
    
    * Individual errors
    quietly gen indiv_error = rnormal(0, sqrt(`within_var'))
    
    * Generate outcome
    """
        
        if assumptions.outcome_type == "continuous":
            code += """quietly gen outcome = `control_mean' + cluster_error + indiv_error
    """
        else:
            code += """quietly gen prob = `control_mean' + cluster_error * 0.1
    quietly replace prob = max(0.01, min(0.99, prob))
    quietly gen outcome = rbinomial(1, prob)
    """
        
        code += """
    * Apply treatment effect
    quietly replace outcome = outcome + `effect' if treat == 1
    
    * Calculate ICC on control group
    quietly loneway outcome cluster_group if treat == 0
    local rho_calculated = `r(rho)'
    
    * Regression with clustered SE
    quietly regress outcome treat, vce(cluster cluster_group)
    
    * T-test
    local t_value = _b[treat] / _se[treat]
    local df = 2 * (`num_clusters' - 1)
    
    if "`side'" == "two" {
        local critical_l = invt(`df', `alpha'/2)
        local critical_u = invt(`df', 1 - `alpha'/2)
        local reject_t = (`t_value' > `critical_u') | (`t_value' < `critical_l')
    }
    else if "`side'" == "left" {
        local critical_l = invt(`df', `alpha')
        local reject_t = (`t_value' < `critical_l')
    }
    else if "`side'" == "right" {
        local critical_u = invt(`df', 1 - `alpha')
        local reject_t = (`t_value' > `critical_u')
    }
    
    post `sim_name' (`reject_t') (`rho_calculated')
    local it = `it' + 1
    clear
}

* ========================================
* CALCULATE POWER
* ========================================
postclose `sim_name'
use `sim_results', clear

sum reject_t
local power = r(mean)

sum rho_calculated
local rho_calc = round(r(mean), 0.01)

di ""
di _dup(70) "="
di "SIMULATION RESULTS"
di _dup(70) "="
di "Number of clusters: `num_clusters'"
di "Cluster size: `cluster_size'"
di "Total sample: `sample_size'"
di "Effect size: `effect'"
di "Specified ICC: `icc'"
di "Calculated ICC: " %5.3f `rho_calc'
di "Number of simulations: `sims'"
di "Estimated power: " %5.3f `power'
di _dup(70) "="
"""
    
    return code


def generate_simulation_python_code(assumptions: SimulationAssumptions, results: dict) -> str:
    """Generate Python code for simulation-based power calculation"""
    
    code = f"""# Power by Simulation - Generated by RCT Field Flow
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
# Based on J-PAL Power Calculation Methodologies

import numpy as np
from scipy import stats

# ========================================
# DESIGN PARAMETERS
# ========================================
effect_size = {assumptions.effect_size}         # Hypothesized treatment effect
treatment_share = {assumptions.treatment_share}  # Proportion in treatment
alpha = {assumptions.alpha}                      # Significance level
test_type = "{assumptions.test_type}"           # two-sided, left, or right
num_simulations = {assumptions.num_simulations} # Monte Carlo iterations
seed = {assumptions.seed}                        # Random seed

"""
    
    if assumptions.design_type == "individual":
        code += f"""sample_size = {assumptions.sample_size}             # Total individuals
control_mean = {assumptions.control_mean}          # Control group mean
control_sd = {assumptions.control_sd}              # Control group SD
outcome_type = "{assumptions.outcome_type}"        # continuous or binary

# ========================================
# RUN SIMULATION
# ========================================
np.random.seed(seed)

reject_count = 0
t_statistics = []

for sim in range(num_simulations):
    # Random treatment assignment
    treatment = np.random.uniform(0, 1, sample_size) < treatment_share
    
    # Generate outcome
    """
        
        if assumptions.outcome_type == "continuous":
            code += """errors = np.random.normal(0, control_sd, sample_size)
    outcome = control_mean + errors
    """
        else:
            code += """outcome = np.random.binomial(1, control_mean, sample_size).astype(float)
    """
        
        code += """
    # Apply treatment effect
    outcome[treatment] += effect_size
    
    # OLS regression
    X = np.column_stack([np.ones(sample_size), treatment.astype(float)])
    y = outcome
    
    # Coefficients
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    
    # Standard errors
    residuals = y - X @ beta
    mse = np.sum(residuals**2) / (sample_size - 2)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_treatment = np.sqrt(var_beta[1, 1])
    
    # T-statistic
    t_stat = beta[1] / se_treatment
    t_statistics.append(t_stat)
    df = sample_size - 2
    
    # Test rejection
    if test_type == "two":
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif test_type == "left":
        p_val = stats.t.cdf(t_stat, df)
    else:  # right
        p_val = 1 - stats.t.cdf(t_stat, df)
    
    if p_val < alpha:
        reject_count += 1

# ========================================
# RESULTS
# ========================================
power = reject_count / num_simulations

print("="*70)
print("SIMULATION RESULTS")
print("="*70)
print(f"Sample size: {{sample_size}}")
print(f"Effect size: {{effect_size}}")
print(f"Number of simulations: {{num_simulations}}")
print(f"Estimated power: {{power:.3f}}")
print(f"Mean t-statistic: {{np.mean(t_statistics):.3f}}")
print("="*70)
"""
    
    else:  # cluster design
        code += f"""num_clusters = {assumptions.num_clusters}           # Number of clusters
cluster_size = {assumptions.cluster_size}          # Individuals per cluster
sample_size = num_clusters * cluster_size
control_mean = {assumptions.control_mean}          # Control intercept
icc = {assumptions.icc}                            # Intracluster correlation
within_var = {assumptions.within_cluster_var}      # Within-cluster variance
between_var = (icc * within_var) / (1 - icc) if icc < 1 else 0
outcome_type = "{assumptions.outcome_type}"        # continuous or binary

# ========================================
# RUN SIMULATION
# ========================================
np.random.seed(seed)

reject_count = 0
icc_estimates = []

for sim in range(num_simulations):
    # Cluster-level treatment assignment
    cluster_treat = np.random.uniform(0, 1, num_clusters) < treatment_share
    
    # Cluster errors
    cluster_errors = np.random.normal(0, np.sqrt(between_var), num_clusters)
    
    # Generate individual data
    cluster_ids = np.repeat(np.arange(num_clusters), cluster_size)
    treatments = np.repeat(cluster_treat, cluster_size).astype(float)
    
    # Individual errors
    indiv_errors = np.random.normal(0, np.sqrt(within_var), sample_size)
    
    # Outcomes
    """
        
        if assumptions.outcome_type == "continuous":
            code += """cluster_errors_expanded = np.repeat(cluster_errors, cluster_size)
    outcome = control_mean + cluster_errors_expanded + indiv_errors
    """
        else:
            code += """cluster_errors_expanded = np.repeat(cluster_errors, cluster_size)
    prob = np.clip(control_mean + cluster_errors_expanded * 0.1, 0.01, 0.99)
    outcome = np.random.binomial(1, prob).astype(float)
    """
        
        code += """
    # Apply treatment effect
    outcome[treatments == 1] += effect_size
    
    # OLS coefficient
    X = np.column_stack([np.ones(sample_size), treatments])
    y = outcome
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    
    # Clustered standard errors
    residuals = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    
    meat = np.zeros((2, 2))
    for c in range(num_clusters):
        cluster_mask = cluster_ids == c
        X_c = X[cluster_mask]
        resid_c = residuals[cluster_mask]
        score = X_c.T @ resid_c
        meat += np.outer(score, score)
    
    vcov = (num_clusters / (num_clusters - 1)) * bread @ meat @ bread
    se_treatment = np.sqrt(vcov[1, 1])
    
    # T-statistic
    t_stat = beta[1] / se_treatment
    df = 2 * (num_clusters - 1)
    
    # Test rejection
    if test_type == "two":
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif test_type == "left":
        p_val = stats.t.cdf(t_stat, df)
    else:
        p_val = 1 - stats.t.cdf(t_stat, df)
    
    if p_val < alpha:
        reject_count += 1

# ========================================
# RESULTS
# ========================================
power = reject_count / num_simulations
design_effect = 1 + (cluster_size - 1) * icc

print("="*70)
print("SIMULATION RESULTS")
print("="*70)
print(f"Number of clusters: {{num_clusters}}")
print(f"Cluster size: {{cluster_size}}")
print(f"Total sample: {{sample_size}}")
print(f"Effect size: {{effect_size}}")
print(f"ICC: {{icc}}")
print(f"Design effect: {{design_effect:.3f}}")
print(f"Number of simulations: {{num_simulations}}")
print(f"Estimated power: {{power:.3f}}")
print("="*70)
"""
    
    return code
