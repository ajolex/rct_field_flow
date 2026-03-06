"""
Auto-narrative generators for RCT Design Wizard
Generate default narrative text based on numeric inputs
"""

from typing import Dict, Any, Optional


def generate_randomization_narrative(numeric: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate default narrative for randomization section
    
    Args:
        numeric: Dictionary with keys: design_type, strata, arms, seed, balance_summary
        
    Returns:
        Dictionary with narrative fields populated
    """
    design_type = numeric.get("design_type", "").lower()
    arms = numeric.get("arms", 0)
    strata = numeric.get("strata", [])
    
    # Rationale
    rationale_parts = []
    
    if design_type and "stratified" in design_type:
        if strata:
            strata_str = ", ".join(strata) if isinstance(strata, list) else str(strata)
            rationale_parts.append(
                f"We employ stratified randomization to ensure balance across key covariates ({strata_str}), "
                "which reduces baseline variance and improves statistical power."
            )
        else:
            rationale_parts.append(
                "We employ stratified randomization to ensure balance across key covariates, "
                "which reduces baseline variance and improves statistical power."
            )
    elif design_type and "cluster" in design_type:
        rationale_parts.append(
            "We use cluster-level randomization to minimize spillover effects and contamination between "
            "treatment and control groups, which is critical given the nature of the intervention."
        )
    else:
        rationale_parts.append(
            "We use simple randomization to assign units to treatment arms, ensuring each unit has an "
            "equal probability of assignment."
        )
    
    if arms and arms > 2:
        rationale_parts.append(
            f"The {arms}-arm design allows us to test multiple intervention variants and identify "
            "the most effective components."
        )
    
    rationale = " ".join(rationale_parts) if rationale_parts else None
    
    # Implementation steps
    implementation = (
        "1. Verify the sampling frame and finalize the list of eligible units\n"
        "2. Collect baseline covariates for stratification (if applicable)\n"
        "3. Execute randomization using a pre-specified seed for reproducibility\n"
        "4. Conduct balance checks on key covariates\n"
        "5. Securely store assignment list and share with field team on implementation date"
    )
    
    # Concealment
    concealment = (
        "Treatment assignment will remain concealed from field staff and participants until the intervention "
        "launch date. The assignment list will be stored in an encrypted file accessible only to the PI and "
        "data manager. Field teams will receive assignments in sealed envelopes or via secure digital channels "
        "immediately before implementation."
    )
    
    # Contamination mitigation
    if design_type and "cluster" in design_type:
        contamination = (
            "We minimize contamination risk through cluster-level randomization with sufficient geographic "
            "separation between clusters. Field teams will be instructed to restrict intervention activities "
            "to assigned clusters and monitor for potential spillovers."
        )
    else:
        contamination = (
            "We will monitor for potential contamination through follow-up surveys asking about exposure to "
            "intervention components. If contamination is detected, we will conduct robustness checks and "
            "report estimates adjusting for non-compliance."
        )
    
    # Clustering justification
    if design_type and "cluster" in design_type:
        clustering = (
            "Cluster randomization is necessary because the intervention is delivered at the group level "
            "(e.g., community, school, or health facility), making individual-level randomization infeasible "
            "or inappropriate due to spillover concerns."
        )
    else:
        clustering = None
    
    return {
        "rationale": rationale,
        "implementation_steps": implementation,
        "concealment": concealment,
        "contamination_mitigation": contamination,
        "clustering_justification": clustering
    }


def generate_power_narrative(numeric: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate default narrative for power section
    
    Args:
        numeric: Dictionary with keys: n_per_arm, mde, icc, alpha, power, variance, attrition, take_up
        
    Returns:
        Dictionary with narrative fields populated
    """
    n_per_arm = numeric.get("n_per_arm")
    mde = numeric.get("mde")
    icc = numeric.get("icc")
    attrition = numeric.get("attrition")
    take_up = numeric.get("take_up")
    power_val = numeric.get("power", 0.80)
    alpha = numeric.get("alpha", 0.05)
    
    # Effect size justification
    if mde:
        effect_size = (
            f"The minimum detectable effect (MDE) of {mde} represents a substantively meaningful impact "
            "based on prior literature and policy relevance. This effect size is considered the smallest "
            "change that would justify the cost and effort of implementing the intervention at scale. "
            "We reviewed similar studies and consulted with policymakers to establish this threshold."
        )
    else:
        effect_size = (
            "The minimum detectable effect (MDE) represents a substantively meaningful impact based on "
            "prior literature and policy relevance. We reviewed similar studies and consulted with "
            "policymakers to establish this threshold."
        )
    
    # Variance source
    variance_source = (
        "Variance estimates are derived from baseline survey data or pilot studies in similar contexts. "
        "We used conservative estimates to account for potential differences between our sample and "
        "reference populations. Standard deviations were calculated from the primary outcome variable "
        "measured in comparable studies."
    )
    
    # Attrition inflation
    if attrition:
        attrition_pct = attrition * 100 if attrition < 1 else attrition
        attrition_inflation = (
            f"We anticipate {attrition_pct:.0f}% attrition based on follow-up rates in similar studies. "
            f"The sample size is inflated accordingly to maintain {power_val*100:.0f}% power. We will implement "
            "tracking protocols including phone contact, community liaisons, and monetary incentives to "
            "minimize attrition and ensure balance across treatment arms."
        )
    else:
        attrition_inflation = (
            "We anticipate some attrition based on follow-up rates in similar studies. The sample size may "
            "need to be inflated accordingly. We will implement tracking protocols to minimize attrition."
        )
    
    # Sensitivity analyses
    sensitivity = (
        "We will conduct sensitivity analyses varying key assumptions:\n"
        "1. Alternative effect sizes (±25% of target MDE)\n"
        "2. Different attrition scenarios (best/worst case)\n"
        "3. Varying take-up rates and compliance levels\n"
        "4. Alternative variance estimates from pilot data\n"
        "These analyses will inform adaptive sample size decisions and help identify data collection priorities."
    )
    
    # Design effect explanation
    if icc:
        design_effect = (
            f"The intracluster correlation (ICC) of {icc} reflects within-cluster homogeneity based on "
            "similar studies. Cluster randomization increases the required sample size through the design "
            "effect (DEFF = 1 + (m-1)×ICC, where m is average cluster size). We account for this by "
            "increasing the total sample size or reducing the number of clusters while maintaining power."
        )
    else:
        design_effect = None
    
    return {
        "effect_size_justification": effect_size,
        "variance_source": variance_source,
        "attrition_inflation": attrition_inflation,
        "sensitivity_analyses": sensitivity,
        "design_effect_explanation": design_effect
    }


def should_generate_narrative(current_narrative: Optional[str]) -> bool:
    """
    Check if narrative should be auto-generated
    
    Args:
        current_narrative: Current narrative text
        
    Returns:
        True if narrative is None or empty, False otherwise
    """
    return not current_narrative or (isinstance(current_narrative, str) and not current_narrative.strip())
