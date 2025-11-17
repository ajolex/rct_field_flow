"""
Power Calculations Module for RCT Field Flow
Supports individual and cluster randomization with/without covariates and compliance
Based on J-PAL power calculation guide and standard statistical methods
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from scipy import stats


@dataclass
class PowerAssumptions:
    """Container for power calculation assumptions"""
    
    # Core parameters
    alpha: float = 0.05  # Significance level (two-sided)
    power: float = 0.80  # Statistical power
    baseline_mean: float = 0.5  # Baseline outcome mean
    baseline_sd: float = 0.5  # Baseline outcome SD
    
    # Outcome type
    outcome_type: Literal["continuous", "binary"] = "continuous"  # Type of outcome variable
    
    # Treatment effect
    effect_size: float | None = None  # Absolute effect size (for MDE calc, leave None)
    mde: float | None = None  # Minimum detectable effect (for sample size calc, leave None)
    
    # Study design
    design_type: Literal["individual", "cluster"] = "individual"
    treatment_share: float = 0.5  # Proportion assigned to treatment
    
    # Cluster parameters (if cluster randomization)
    num_clusters: int | None = None  # Total clusters
    cluster_size: int | None = None  # Average individuals per cluster
    icc: float = 0.0  # Intracluster correlation coefficient
    
    # Covariates
    r_squared: float = 0.0  # R² from baseline covariates (0-1)
    
    # Compliance
    compliance_rate: float = 1.0  # Proportion of treated who actually receive treatment
    
    # Attrition
    attrition_rate: float = 0.0  # Expected attrition rate (0-0.3). For clusters, applied to cluster size not num clusters
    
    def __post_init__(self):
        """Validate assumptions"""
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        if not 0 < self.power < 1:
            raise ValueError("Power must be between 0 and 1")
        if self.treatment_share <= 0 or self.treatment_share >= 1:
            raise ValueError("Treatment share must be between 0 and 1")
        if self.r_squared < 0 or self.r_squared > 1:
            raise ValueError("R² must be between 0 and 1")
        if self.compliance_rate <= 0 or self.compliance_rate > 1:
            raise ValueError("Compliance rate must be between 0 and 1")
        if self.attrition_rate < 0 or self.attrition_rate >= 1:
            raise ValueError("Attrition rate must be between 0 and 1")
        if self.design_type == "cluster":
            # For cluster designs, cluster_size is always required; num_clusters is only
            # required for MDE calculations (not for sample size calculations).
            if self.cluster_size is None:
                raise ValueError("Cluster design requires cluster_size")
            if self.num_clusters is not None and self.num_clusters <= 0:
                raise ValueError("num_clusters must be positive when provided")
            if self.icc < 0 or self.icc > 1:
                raise ValueError("ICC must be between 0 and 1")
        
        # Binary outcome validations
        if self.outcome_type == "binary":
            if not 0 < self.baseline_mean < 1:
                raise ValueError("For binary outcomes, baseline mean must be a proportion between 0 and 1")
            # For binary outcomes, SD is calculated from proportion
            self.baseline_sd = math.sqrt(self.baseline_mean * (1 - self.baseline_mean))
            # Note: Covariates (R²) not applicable for binary outcomes per J-PAL guidelines
            if self.r_squared > 0:
                raise ValueError("Covariates adjustment (R²) not applicable for binary outcomes")


def calculate_design_effect(cluster_size: int, icc: float) -> float:
    """
    Calculate design effect (DEFF) for cluster randomization
    DEFF = 1 + (m - 1) * ICC
    where m is average cluster size
    """
    return 1 + (cluster_size - 1) * icc


def calculate_mde(
    assumptions: PowerAssumptions,
    sample_size: int | None = None
) -> dict:
    """
    Calculate Minimum Detectable Effect given sample size and assumptions
    
    Returns dict with MDE in absolute and standardized units, including adjustments for attrition
    """
    # Critical values
    z_alpha = stats.norm.ppf(1 - assumptions.alpha / 2)
    z_beta = stats.norm.ppf(assumptions.power)
    
    # Effective sample size adjustment for covariates
    variance_inflation = 1 - assumptions.r_squared
    
    if assumptions.design_type == "individual":
        # Individual randomization
        if sample_size is None:
            raise ValueError("Sample size required for MDE calculation")
        
        # Adjust for treatment allocation ratio
        p = assumptions.treatment_share
        allocation_factor = 1 / (p * (1 - p))
        
        # Base MDE in SD units
        mde_sd = (z_alpha + z_beta) * math.sqrt(
            allocation_factor * variance_inflation / sample_size
        )
        
        # Adjust for imperfect compliance (ITT estimand)
        if assumptions.compliance_rate < 1.0:
            mde_sd = mde_sd / assumptions.compliance_rate
        
        # Convert to absolute units
        mde_abs = mde_sd * assumptions.baseline_sd
        
        # Calculate with attrition adjustment
        attrition_factor = 1 / (1 - assumptions.attrition_rate) if assumptions.attrition_rate > 0 else 1.0
        sample_with_attrition = int(math.ceil(sample_size * attrition_factor))
        
        # Recalculate MDE with inflated sample
        if assumptions.attrition_rate > 0:
            mde_sd_with_attrition = (z_alpha + z_beta) * math.sqrt(
                allocation_factor * variance_inflation / sample_with_attrition
            )
            if assumptions.compliance_rate < 1.0:
                mde_sd_with_attrition = mde_sd_with_attrition / assumptions.compliance_rate
            mde_abs_with_attrition = mde_sd_with_attrition * assumptions.baseline_sd
        else:
            mde_sd_with_attrition = mde_sd
            mde_abs_with_attrition = mde_abs
        
        return {
            "mde_sd": mde_sd,
            "mde_absolute": mde_abs,
            "sample_size": sample_size,
            "design_effect": 1.0,
            "effective_n": sample_size,
            "attrition_rate": assumptions.attrition_rate,
            "sample_with_attrition": sample_with_attrition,
            "mde_sd_with_attrition": mde_sd_with_attrition,
            "mde_absolute_with_attrition": mde_abs_with_attrition,
        }
    
    else:  # cluster randomization
        if assumptions.num_clusters is None or assumptions.cluster_size is None:
            raise ValueError("Cluster design requires num_clusters and cluster_size")
        
        # Design effect
        deff = calculate_design_effect(assumptions.cluster_size, assumptions.icc)
        
        # Effective sample size
        total_individuals = assumptions.num_clusters * assumptions.cluster_size
        effective_n = total_individuals / deff
        
        # Allocation factor
        p = assumptions.treatment_share
        allocation_factor = 1 / (p * (1 - p))
        
        # MDE in SD units
        mde_sd = (z_alpha + z_beta) * math.sqrt(
            allocation_factor * variance_inflation * deff / total_individuals
        )
        
        # Adjust for imperfect compliance
        if assumptions.compliance_rate < 1.0:
            mde_sd = mde_sd / assumptions.compliance_rate
        
        # Convert to absolute units
        mde_abs = mde_sd * assumptions.baseline_sd
        
        # Calculate with attrition adjustment - applied to cluster SIZE not number of clusters
        attrition_factor = 1 / (1 - assumptions.attrition_rate) if assumptions.attrition_rate > 0 else 1.0
        cluster_size_with_attrition = int(math.ceil(assumptions.cluster_size * attrition_factor))
        total_with_attrition = assumptions.num_clusters * cluster_size_with_attrition
        
        # Recalculate design effect with adjusted cluster size
        deff_with_attrition = calculate_design_effect(cluster_size_with_attrition, assumptions.icc)
        effective_n_with_attrition = total_with_attrition / deff_with_attrition
        
        # Recalculate MDE with inflated cluster size
        if assumptions.attrition_rate > 0:
            mde_sd_with_attrition = (z_alpha + z_beta) * math.sqrt(
                allocation_factor * variance_inflation * deff_with_attrition / total_with_attrition
            )
            if assumptions.compliance_rate < 1.0:
                mde_sd_with_attrition = mde_sd_with_attrition / assumptions.compliance_rate
            mde_abs_with_attrition = mde_sd_with_attrition * assumptions.baseline_sd
        else:
            mde_sd_with_attrition = mde_sd
            mde_abs_with_attrition = mde_abs
            cluster_size_with_attrition = assumptions.cluster_size
            deff_with_attrition = deff
        
        return {
            "mde_sd": mde_sd,
            "mde_absolute": mde_abs,
            "num_clusters": assumptions.num_clusters,
            "cluster_size": assumptions.cluster_size,
            "total_individuals": total_individuals,
            "design_effect": deff,
            "effective_n": effective_n,
            "icc": assumptions.icc,
            "attrition_rate": assumptions.attrition_rate,
            "cluster_size_with_attrition": cluster_size_with_attrition,
            "total_with_attrition": total_with_attrition,
            "design_effect_with_attrition": deff_with_attrition,
            "effective_n_with_attrition": effective_n_with_attrition,
            "mde_sd_with_attrition": mde_sd_with_attrition,
            "mde_absolute_with_attrition": mde_abs_with_attrition,
        }


def calculate_sample_size(assumptions: PowerAssumptions) -> dict:
    """
    Calculate required sample size given MDE and assumptions
    
    Returns dict with sample size and related statistics, including adjustments for attrition
    """
    if assumptions.mde is None:
        raise ValueError("MDE required for sample size calculation")
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - assumptions.alpha / 2)
    z_beta = stats.norm.ppf(assumptions.power)
    
    # Convert MDE to SD units if needed
    mde_sd = assumptions.mde / assumptions.baseline_sd
    
    # Adjust for compliance
    if assumptions.compliance_rate < 1.0:
        mde_sd = mde_sd * assumptions.compliance_rate
    
    # Variance inflation from covariates
    variance_inflation = 1 - assumptions.r_squared
    
    # Allocation factor
    p = assumptions.treatment_share
    allocation_factor = 1 / (p * (1 - p))
    
    if assumptions.design_type == "individual":
        # Individual randomization
        n_required = math.ceil(
            allocation_factor * variance_inflation * ((z_alpha + z_beta) / mde_sd) ** 2
        )
        
        # Adjust for attrition
        attrition_factor = 1 / (1 - assumptions.attrition_rate) if assumptions.attrition_rate > 0 else 1.0
        n_with_attrition = int(math.ceil(n_required * attrition_factor))
        
        return {
            "sample_size": n_required,
            "design_effect": 1.0,
            "effective_n": n_required,
            "mde_sd": mde_sd,
            "attrition_rate": assumptions.attrition_rate,
            "sample_size_with_attrition": n_with_attrition,
        }
    
    else:  # cluster randomization
        if assumptions.cluster_size is None:
            raise ValueError("Cluster size required for cluster design")
        
        # Design effect
        deff = calculate_design_effect(assumptions.cluster_size, assumptions.icc)
        
        # Required total individuals
        n_individuals = math.ceil(
            allocation_factor * variance_inflation * deff * ((z_alpha + z_beta) / mde_sd) ** 2
        )
        
        # Required clusters
        n_clusters = math.ceil(n_individuals / assumptions.cluster_size)
        
        # Actual total individuals (rounded up to whole clusters)
        actual_individuals = n_clusters * assumptions.cluster_size
        
        # Adjust for attrition - applied to cluster SIZE not number of clusters
        attrition_factor = 1 / (1 - assumptions.attrition_rate) if assumptions.attrition_rate > 0 else 1.0
        cluster_size_with_attrition = int(math.ceil(assumptions.cluster_size * attrition_factor))
        total_with_attrition = n_clusters * cluster_size_with_attrition
        
        return {
            "num_clusters": n_clusters,
            "cluster_size": assumptions.cluster_size,
            "total_individuals": actual_individuals,
            "design_effect": deff,
            "effective_n": actual_individuals / deff,
            "mde_sd": mde_sd,
            "icc": assumptions.icc,
            "attrition_rate": assumptions.attrition_rate,
            "cluster_size_with_attrition": cluster_size_with_attrition,
            "total_with_attrition": total_with_attrition,
        }


def generate_power_curve(
    assumptions: PowerAssumptions,
    sample_sizes: list[int] | None = None
) -> pd.DataFrame:
    """
    Generate power curve data for different sample sizes
    
    Returns DataFrame with sample_size and corresponding power
    """
    if assumptions.effect_size is None:
        raise ValueError("Effect size required for power curve")
    
    # Default sample size range
    if sample_sizes is None:
        if assumptions.design_type == "cluster":
            # For clusters, vary number of clusters
            base = assumptions.num_clusters or 20
            sample_sizes = list(range(max(10, base // 2), base * 3, 2))
        else:
            # For individual, vary total sample
            sample_sizes = list(range(50, 2000, 50))
    
    results = []
    
    for n in sample_sizes:
        if assumptions.design_type == "cluster":
            # Treat n as number of clusters
            temp_assumptions = PowerAssumptions(
                alpha=assumptions.alpha,
                power=assumptions.power,
                baseline_mean=assumptions.baseline_mean,
                baseline_sd=assumptions.baseline_sd,
                effect_size=assumptions.effect_size,
                design_type="cluster",
                treatment_share=assumptions.treatment_share,
                num_clusters=n,
                cluster_size=assumptions.cluster_size,
                icc=assumptions.icc,
                r_squared=assumptions.r_squared,
                compliance_rate=assumptions.compliance_rate,
            )
            assert assumptions.cluster_size is not None  # Type guard
            total_n = n * assumptions.cluster_size
        else:
            temp_assumptions = PowerAssumptions(
                alpha=assumptions.alpha,
                power=assumptions.power,
                baseline_mean=assumptions.baseline_mean,
                baseline_sd=assumptions.baseline_sd,
                effect_size=assumptions.effect_size,
                design_type="individual",
                treatment_share=assumptions.treatment_share,
                r_squared=assumptions.r_squared,
                compliance_rate=assumptions.compliance_rate,
            )
            total_n = n
        
        # Calculate power for this sample size
        power_val = calculate_power_for_effect(temp_assumptions, n)
        
        results.append({
            "sample_size": total_n,
            "clusters": n if assumptions.design_type == "cluster" else None,
            "power": power_val,
        })
    
    return pd.DataFrame(results)


def calculate_power_for_effect(assumptions: PowerAssumptions, sample_size: int) -> float:
    """Calculate statistical power for given effect size and sample size"""
    # Critical value
    z_alpha = stats.norm.ppf(1 - assumptions.alpha / 2)
    
    # Effect size in SD units
    assert assumptions.effect_size is not None  # Type guard
    effect_sd = assumptions.effect_size / assumptions.baseline_sd
    
    # Adjust for compliance
    if assumptions.compliance_rate < 1.0:
        effect_sd = effect_sd * assumptions.compliance_rate
    
    # Variance inflation
    variance_inflation = 1 - assumptions.r_squared
    
    # Allocation factor
    p = assumptions.treatment_share
    allocation_factor = 1 / (p * (1 - p))
    
    if assumptions.design_type == "cluster":
        assert assumptions.cluster_size is not None  # Type guard
        deff = calculate_design_effect(assumptions.cluster_size, assumptions.icc)
        total_n = sample_size * assumptions.cluster_size
        ncp = effect_sd * math.sqrt(total_n / (allocation_factor * variance_inflation * deff))
    else:
        ncp = effect_sd * math.sqrt(sample_size / (allocation_factor * variance_inflation))
    
    # Non-centrality parameter for power
    power = stats.norm.cdf(ncp - z_alpha)
    
    return float(power)


def generate_cluster_size_table(
    assumptions: PowerAssumptions,
    cluster_sizes: list[int] | None = None,
    num_clusters_options: list[int] | None = None
) -> pd.DataFrame:
    """
    Generate MDE table for different cluster sizes and number of clusters
    Table structure: rows = cluster_size (with 1-2 increments), columns = num_clusters
    MDE values are adjusted for attrition (final numbers after attrition)
    
    Returns DataFrame with cluster_size, num_clusters, and MDE
    """
    if assumptions.design_type != "cluster":
        raise ValueError("Cluster size table only applicable for cluster designs")
    
    # Calculate dynamic ranges based on assumed cluster_size and num_clusters (±25%)
    if cluster_sizes is None:
        m_base = assumptions.cluster_size
        m_min = max(2, int(m_base * 0.75))  # Min 75% of base
        m_max = int(m_base * 1.25)  # Max 125% of base
        
        # Generate cluster sizes with 1-2 increments
        if m_max - m_min <= 10:
            # Use increment of 1 for small ranges
            cluster_sizes = list(range(m_min, m_max + 1))
        else:
            # Use increment of 2 for larger ranges
            cluster_sizes = list(range(m_min, m_max + 1, 2))
            # Always include the exact base value
            if m_base not in cluster_sizes:
                cluster_sizes.append(m_base)
                cluster_sizes.sort()
    
    if num_clusters_options is None:
        k_base = assumptions.num_clusters
        k_min = max(5, int(k_base * 0.75))  # Min 75% of base, at least 5
        k_max = int(k_base * 1.25)  # Max 125% of base
        
        # Generate num_clusters options with reasonable increments
        if k_max - k_min <= 20:
            # Use increment of 2-3 for small ranges
            step = 2 if (k_max - k_min) <= 10 else 3
            num_clusters_options = list(range(k_min, k_max + 1, step))
        else:
            # Use increment of 5 for larger ranges
            num_clusters_options = list(range(k_min, k_max + 1, 5))
        
        # Always include the exact base value
        if k_base not in num_clusters_options:
            num_clusters_options.append(k_base)
            num_clusters_options.sort()
    
    results = []
    
    for m in cluster_sizes:
        for k in num_clusters_options:
            temp_assumptions = PowerAssumptions(
                alpha=assumptions.alpha,
                power=assumptions.power,
                baseline_mean=assumptions.baseline_mean,
                baseline_sd=assumptions.baseline_sd,
                design_type="cluster",
                treatment_share=assumptions.treatment_share,
                num_clusters=k,
                cluster_size=m,
                icc=assumptions.icc,
                r_squared=assumptions.r_squared,
                compliance_rate=assumptions.compliance_rate,
            )
            
            mde_result = calculate_mde(temp_assumptions)
            
            # Use attrition-adjusted MDE if attrition > 0
            if assumptions.attrition_rate > 0:
                mde_to_display = mde_result["mde_absolute_with_attrition"]
                total_sample = mde_result["total_with_attrition"]
                mde_sd_display = mde_result["mde_sd_with_attrition"]
                m_display = mde_result["cluster_size_with_attrition"]
            else:
                mde_to_display = mde_result["mde_absolute"]
                total_sample = m * k
                mde_sd_display = mde_result["mde_sd"]
                m_display = m
            
            results.append({
                "cluster_size": m_display,
                "num_clusters": k,
                "total_sample": total_sample,
                "design_effect": mde_result.get("design_effect_with_attrition", mde_result["design_effect"]),
                "mde_sd": mde_sd_display,
                "mde_absolute": mde_to_display,
            })
    
    return pd.DataFrame(results)


def generate_python_code(assumptions: PowerAssumptions, results: dict) -> str:
    """Generate reproducible Python code for the power calculation"""
    
    code = f"""# Power Calculation - Generated by RCT Field Flow
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

import numpy as np
from scipy import stats
import math

# ========================================
# ASSUMPTIONS
# ========================================
alpha = {assumptions.alpha}  # Significance level (two-sided)
power = {assumptions.power}  # Statistical power
baseline_mean = {assumptions.baseline_mean}  # Baseline outcome mean
baseline_sd = {assumptions.baseline_sd}  # Baseline outcome SD
design_type = "{assumptions.design_type}"  # Individual or cluster randomization
treatment_share = {assumptions.treatment_share}  # Proportion in treatment arm

# Effect and sample
"""
    
    if assumptions.mde is not None:
        code += f"mde = {assumptions.mde}  # Minimum detectable effect (absolute)\n"
    if assumptions.effect_size is not None:
        code += f"effect_size = {assumptions.effect_size}  # Expected effect size (absolute)\n"
    
    code += f"""
# Covariates and compliance
r_squared = {assumptions.r_squared}  # R² from baseline covariates
compliance_rate = {assumptions.compliance_rate}  # Treatment compliance rate
attrition_rate = {assumptions.attrition_rate}  # Expected attrition rate

"""
    
    if assumptions.design_type == "cluster":
        code += f"""# Cluster parameters
num_clusters = {assumptions.num_clusters}  # Total number of clusters
cluster_size = {assumptions.cluster_size}  # Average cluster size
icc = {assumptions.icc}  # Intracluster correlation coefficient

# ========================================
# CALCULATIONS
# ========================================

# Design effect
design_effect = 1 + (cluster_size - 1) * icc
print(f"Design Effect (DEFF): {{design_effect:.3f}}")

# Critical values
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)

# Variance adjustments
variance_inflation = 1 - r_squared  # Covariate adjustment
allocation_factor = 1 / (treatment_share * (1 - treatment_share))

# Total sample
total_individuals = num_clusters * cluster_size
effective_n = total_individuals / design_effect

print(f"Total individuals: {{total_individuals}}")
print(f"Effective sample size: {{effective_n:.1f}}")

"""
        
        if assumptions.mde is not None:
            code += """# Calculate required sample size
mde_sd = mde / baseline_sd
if compliance_rate < 1.0:
    mde_sd = mde_sd * compliance_rate

n_individuals_required = math.ceil(
    allocation_factor * variance_inflation * design_effect * 
    ((z_alpha + z_beta) / mde_sd) ** 2
)
n_clusters_required = math.ceil(n_individuals_required / cluster_size)

print(f"\\nRequired number of clusters (without attrition): {n_clusters_required}")
print(f"Required total sample (without attrition): {n_clusters_required * cluster_size}")

# Adjust for attrition - applied to cluster SIZE not number of clusters
if attrition_rate > 0:
    attrition_factor = 1 / (1 - attrition_rate)
    cluster_size_with_attrition = math.ceil(cluster_size * attrition_factor)
    total_with_attrition = n_clusters_required * cluster_size_with_attrition
    print(f"\\nInflated cluster size (with {attrition_rate*100:.0f}% attrition): {cluster_size_with_attrition}")
    print(f"Required total sample (with attrition): {n_clusters_required} clusters × {cluster_size_with_attrition} = {total_with_attrition}")
"""
        else:
            code += """# Calculate MDE
mde_sd = (z_alpha + z_beta) * math.sqrt(
    allocation_factor * variance_inflation * design_effect / total_individuals
)
if compliance_rate < 1.0:
    mde_sd = mde_sd / compliance_rate

mde_absolute = mde_sd * baseline_sd

print(f"\\nMinimum Detectable Effect (without attrition):")
print(f"  In SD units: {mde_sd:.3f}")
print(f"  Absolute: {mde_absolute:.3f}")

# Adjust for attrition - applied to cluster SIZE not number of clusters
if attrition_rate > 0:
    attrition_factor = 1 / (1 - attrition_rate)
    cluster_size_with_attrition = math.ceil(cluster_size * attrition_factor)
    total_with_attrition = num_clusters * cluster_size_with_attrition
    
    # Recalculate design effect with inflated cluster size
    deff_with_attrition = 1 + (cluster_size_with_attrition - 1) * icc
    
    mde_sd_with_attrition = (z_alpha + z_beta) * math.sqrt(
        allocation_factor * variance_inflation * deff_with_attrition / total_with_attrition
    )
    if compliance_rate < 1.0:
        mde_sd_with_attrition = mde_sd_with_attrition / compliance_rate
    mde_absolute_with_attrition = mde_sd_with_attrition * baseline_sd
    
    print(f"\\nWith {attrition_rate*100:.0f}% attrition:")
    print(f"  Inflated cluster size: {cluster_size_with_attrition}")
    print(f"  Total N: {num_clusters} clusters × {cluster_size_with_attrition} = {total_with_attrition}")
    print(f"  MDE (SD units): {mde_sd_with_attrition:.3f}")
    print(f"  MDE (absolute): {mde_absolute_with_attrition:.3f}")
"""
    
    else:  # individual randomization
        code += """# ========================================
# CALCULATIONS
# ========================================

# Critical values
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)

# Variance adjustments
variance_inflation = 1 - r_squared
allocation_factor = 1 / (treatment_share * (1 - treatment_share))

"""
        
        if assumptions.mde is not None:
            code += f"""# Calculate required sample size
sample_size = {results.get('sample_size', 'N/A')}

mde_sd = mde / baseline_sd
if compliance_rate < 1.0:
    mde_sd = mde_sd * compliance_rate

n_required = math.ceil(
    allocation_factor * variance_inflation * ((z_alpha + z_beta) / mde_sd) ** 2
)

print(f"Required sample size (without attrition): {{n_required}}")

# Adjust for attrition
if attrition_rate > 0:
    attrition_factor = 1 / (1 - attrition_rate)
    n_with_attrition = math.ceil(n_required * attrition_factor)
    print(f"Required sample size (with {{attrition_rate*100:.0f}}% attrition): {{n_with_attrition}}")
"""
        else:
            code += f"""# Calculate MDE
sample_size = {results.get('sample_size', 'N/A')}

mde_sd = (z_alpha + z_beta) * math.sqrt(
    allocation_factor * variance_inflation / sample_size
)
if compliance_rate < 1.0:
    mde_sd = mde_sd / compliance_rate

mde_absolute = mde_sd * baseline_sd

print(f"\\nMinimum Detectable Effect (without attrition):")
print(f"  In SD units: {{mde_sd:.3f}}")
print(f"  Absolute: {{mde_absolute:.3f}}")

# Adjust for attrition
if attrition_rate > 0:
    attrition_factor = 1 / (1 - attrition_rate)
    n_with_attrition = math.ceil(sample_size * attrition_factor)
    
    mde_sd_with_attrition = (z_alpha + z_beta) * math.sqrt(
        allocation_factor * variance_inflation / n_with_attrition
    )
    if compliance_rate < 1.0:
        mde_sd_with_attrition = mde_sd_with_attrition / compliance_rate
    mde_absolute_with_attrition = mde_sd_with_attrition * baseline_sd
    
    print(f"\\nMinimum Detectable Effect (with {{attrition_rate*100:.0f}}% attrition, N={{n_with_attrition}}):")
    print(f"  In SD units: {{mde_sd_with_attrition:.3f}}")
    print(f"  Absolute: {{mde_absolute_with_attrition:.3f}}")
"""
    
    code += """
# ========================================
# KEY INSIGHTS
# ========================================
print("\\n" + "="*50)
print("Power Calculation Summary")
print("="*50)
"""
    
    return code


def generate_stata_code(assumptions: PowerAssumptions, results: dict) -> str:
    """Generate reproducible Stata code for the power calculation"""
    
    outcome_note = "proportion" if assumptions.outcome_type == "binary" else "mean"
    
    code = f"""* Power Calculation - Generated by RCT Field Flow
* Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

clear all
set more off

* ========================================
* ASSUMPTIONS
* ========================================
* Outcome type: {assumptions.outcome_type}
scalar alpha = {assumptions.alpha}  // Significance level (two-sided)
scalar power = {assumptions.power}  // Statistical power
scalar baseline_mean = {assumptions.baseline_mean}  // Baseline outcome {outcome_note}
scalar baseline_sd = {assumptions.baseline_sd}  // Baseline outcome SD
local design_type "{assumptions.design_type}"  // Individual or cluster
scalar p_treat = {assumptions.treatment_share}  // Treatment share

"""
    
    if assumptions.mde is not None:
        code += f"scalar mde = {assumptions.mde}  // Minimum detectable effect\n"
    if assumptions.effect_size is not None:
        code += f"scalar effect = {assumptions.effect_size}  // Effect size\n"
    
    # Covariates only for continuous outcomes
    if assumptions.outcome_type == "continuous":
        code += f"""* Covariates and compliance
scalar r2 = {assumptions.r_squared}  // R² from covariates
scalar compliance = {assumptions.compliance_rate}  // Compliance rate
scalar attrition_rate = {assumptions.attrition_rate}  // Expected attrition rate

"""
    else:
        code += f"""* Compliance
scalar compliance = {assumptions.compliance_rate}  // Compliance rate
scalar attrition_rate = {assumptions.attrition_rate}  // Expected attrition rate
* Note: Covariate adjustment (R²) not applicable for binary outcomes

"""
    
    if assumptions.design_type == "cluster":
        if assumptions.mde is not None:
            code += f"""* Cluster parameters
scalar m = {assumptions.cluster_size}  // Cluster size
scalar icc = {assumptions.icc}  // Intracluster correlation

* ========================================
* CALCULATIONS
* ========================================

* Design effect
scalar deff = 1 + (m - 1) * icc
di "Design Effect: " deff

* Critical values
scalar z_a = invnormal(1 - alpha/2)
scalar z_b = invnormal(power)

* Variance adjustments
scalar var_infl = 1 - r2
scalar alloc_factor = 1 / (p_treat * (1 - p_treat))

* Calculate required clusters and set k accordingly
scalar mde_sd = mde / baseline_sd
if compliance < 1 {{
    scalar mde_sd = mde_sd * compliance
}}

scalar n_req = ceil(alloc_factor * var_infl * deff * ((z_a + z_b)/mde_sd)^2)
scalar k_req = ceil(n_req / m)
scalar k = k_req
scalar n_total = k * m
scalar n_eff = n_total / deff

di ""
di "Required clusters (without attrition): " k
di "Required total sample (without attrition): " n_total
di "Effective sample: " n_eff

* Adjust for attrition - applied to cluster SIZE not number of clusters
if attrition_rate > 0 {{
    scalar attrition_factor = 1 / (1 - attrition_rate)
    scalar m_with_attrition = ceil(m * attrition_factor)
    scalar n_with_attrition = k * m_with_attrition
    di ""
    di "Inflated cluster size (with " attrition_rate*100 "% attrition): " m_with_attrition
    di "Required total sample (with attrition): " k " clusters × " m_with_attrition " = " n_with_attrition
}}
"""
        else:
            code += f"""* Cluster parameters
scalar k = {assumptions.num_clusters}  // Number of clusters
scalar m = {assumptions.cluster_size}  // Cluster size
scalar icc = {assumptions.icc}  // Intracluster correlation

* ========================================
* CALCULATIONS
* ========================================

* Design effect
scalar deff = 1 + (m - 1) * icc
di "Design Effect: " deff

* Critical values
scalar z_a = invnormal(1 - alpha/2)
scalar z_b = invnormal(power)

* Variance adjustments
scalar var_infl = 1 - r2
scalar alloc_factor = 1 / (p_treat * (1 - p_treat))

* Total sample
scalar n_total = k * m
scalar n_eff = n_total / deff

di "Total individuals: " n_total
di "Effective sample: " n_eff

* Calculate MDE
scalar mde_sd = (z_a + z_b) * sqrt(alloc_factor * var_infl * deff / n_total)
if compliance < 1 {{
    scalar mde_sd = mde_sd / compliance
}}
scalar mde_abs = mde_sd * baseline_sd

di ""
di "Minimum Detectable Effect (without attrition):"
di "  SD units: " mde_sd
di "  Absolute: " mde_abs

* Adjust for attrition - applied to cluster SIZE not number of clusters
if attrition_rate > 0 {{
    scalar attrition_factor = 1 / (1 - attrition_rate)
    scalar m_with_attrition = ceil(m * attrition_factor)
    scalar n_with_attrition = k * m_with_attrition
    
    * Recalculate design effect with inflated cluster size
    scalar deff_with_attrition = 1 + (m_with_attrition - 1) * icc
    
    scalar mde_sd_with_attrition = (z_a + z_b) * sqrt(alloc_factor * var_infl * deff_with_attrition / n_with_attrition)
    if compliance < 1 {{
        scalar mde_sd_with_attrition = mde_sd_with_attrition / compliance
    }}
    scalar mde_abs_with_attrition = mde_sd_with_attrition * baseline_sd
    
    di ""
    di "With " attrition_rate*100 "% attrition:"
    di "  Inflated cluster size: " m_with_attrition
    di "  Total N: " k " clusters × " m_with_attrition " = " n_with_attrition
    di "  MDE (SD units): " mde_sd_with_attrition
    di "  MDE (absolute): " mde_abs_with_attrition
}}
"""
    
    else:  # individual
        code += """* ========================================
* CALCULATIONS
* ========================================

* Critical values
scalar z_a = invnormal(1 - alpha/2)
scalar z_b = invnormal(power)

* Variance adjustments
scalar var_infl = 1 - r2
scalar alloc_factor = 1 / (p_treat * (1 - p_treat))

"""
        
        if assumptions.mde is not None:
            code += f"""* Calculate required sample
scalar n = {results.get('sample_size', '.')}

scalar mde_sd = mde / baseline_sd
if compliance < 1 {{
    scalar mde_sd = mde_sd * compliance
}}

scalar n_req = ceil(alloc_factor * var_infl * ((z_a + z_b)/mde_sd)^2)

di "Required sample size (without attrition): " n_req

* Adjust for attrition
if attrition_rate > 0 {{
    scalar attrition_factor = 1 / (1 - attrition_rate)
    scalar n_with_attrition = ceil(n_req * attrition_factor)
    di "Required sample size (with " attrition_rate*100 "% attrition): " n_with_attrition
}}
"""
        else:
            code += f"""* Calculate MDE
scalar n = {results.get('sample_size', '.')}

scalar mde_sd = (z_a + z_b) * sqrt(alloc_factor * var_infl / n)
if compliance < 1 {{
    scalar mde_sd = mde_sd / compliance
}}
scalar mde_abs = mde_sd * baseline_sd

di ""
di "Minimum Detectable Effect (without attrition):"
di "  SD units: " mde_sd
di "  Absolute: " mde_abs

* Adjust for attrition
if attrition_rate > 0 {{
    scalar attrition_factor = 1 / (1 - attrition_rate)
    scalar n_with_attrition = ceil(n * attrition_factor)
    
    scalar mde_sd_with_attrition = (z_a + z_b) * sqrt(alloc_factor * var_infl / n_with_attrition)
    if compliance < 1 {{
        scalar mde_sd_with_attrition = mde_sd_with_attrition / compliance
    }}
    scalar mde_abs_with_attrition = mde_sd_with_attrition * baseline_sd
    
    di ""
    di "Minimum Detectable Effect (with " attrition_rate*100 "% attrition, N=" n_with_attrition "):"
    di "  SD units: " mde_sd_with_attrition
    di "  Absolute: " mde_abs_with_attrition
}}
"""
    
    code += """
* ========================================
* SUMMARY
* ========================================
di ""
di _dup(50) "="
di "Power Calculation Summary"
di _dup(50) "="
"""
    
    return code
