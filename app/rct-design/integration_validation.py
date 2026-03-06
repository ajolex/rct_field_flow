"""
Integration Contract Validation
Tests and validates integration with upstream pages
"""

from typing import Dict, Any, List, Tuple


def validate_randomization_contract(session_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate randomization integration contract
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if "rand" not in session_state:
        return False, ["Missing 'rand' key in session_state"]
    
    rand = session_state["rand"]
    
    # Check required keys
    required_keys = ["design_type", "arms", "seed"]
    for key in required_keys:
        if key not in rand:
            issues.append(f"Missing required key in rand: {key}")
    
    # Check optional keys
    optional_keys = ["strata", "balance_summary"]
    for key in optional_keys:
        if key not in rand:
            issues.append(f"Missing optional key in rand: {key} (will use default)")
    
    # Type validation
    if "design_type" in rand and not isinstance(rand["design_type"], str):
        issues.append(f"rand.design_type should be str, got {type(rand['design_type'])}")
    
    if "arms" in rand and not isinstance(rand["arms"], int):
        issues.append(f"rand.arms should be int, got {type(rand['arms'])}")
    
    if "seed" in rand and not isinstance(rand["seed"], int):
        issues.append(f"rand.seed should be int, got {type(rand['seed'])}")
    
    if "strata" in rand:
        if not (isinstance(rand["strata"], list) or isinstance(rand["strata"], str)):
            issues.append(f"rand.strata should be list or str, got {type(rand['strata'])}")
    
    return len(issues) == 0, issues


def validate_power_contract(session_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate power integration contract
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if "power" not in session_state:
        return False, ["Missing 'power' key in session_state"]
    
    power = session_state["power"]
    
    # Check for n_per_arm or n_each
    if "n_per_arm" not in power and "n_each" not in power:
        issues.append("Missing 'n_per_arm' or 'n_each' in power")
    
    # Check optional keys
    optional_keys = ["mde", "icc", "assumptions", "notes"]
    for key in optional_keys:
        if key not in power:
            issues.append(f"Missing optional key in power: {key} (will use default)")
    
    # Type validation
    if "n_per_arm" in power:
        if not isinstance(power["n_per_arm"], (int, float)):
            issues.append(f"power.n_per_arm should be numeric, got {type(power['n_per_arm'])}")
    
    if "n_each" in power:
        if not isinstance(power["n_each"], (int, float)):
            issues.append(f"power.n_each should be numeric, got {type(power['n_each'])}")
    
    if "mde" in power:
        if not isinstance(power["mde"], (int, float)):
            issues.append(f"power.mde should be numeric, got {type(power['mde'])}")
    
    if "icc" in power:
        if not isinstance(power["icc"], (int, float)):
            issues.append(f"power.icc should be numeric, got {type(power['icc'])}")
    
    if "assumptions" in power:
        if not isinstance(power["assumptions"], dict):
            issues.append(f"power.assumptions should be dict, got {type(power['assumptions'])}")
        else:
            # Check assumption keys
            expected_assumptions = ["alpha", "power", "variance", "attrition", "take_up"]
            for key in expected_assumptions:
                if key not in power["assumptions"]:
                    issues.append(f"Missing assumption key: {key} (will use default)")
    
    return len(issues) == 0, issues


def validate_all_contracts(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all integration contracts
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "randomization": {
            "available": "rand" in session_state,
            "valid": False,
            "issues": []
        },
        "power": {
            "available": "power" in session_state,
            "valid": False,
            "issues": []
        }
    }
    
    # Validate randomization
    if results["randomization"]["available"]:
        is_valid, issues = validate_randomization_contract(session_state)
        results["randomization"]["valid"] = is_valid
        results["randomization"]["issues"] = issues
    else:
        results["randomization"]["issues"] = ["Randomization data not available"]
    
    # Validate power
    if results["power"]["available"]:
        is_valid, issues = validate_power_contract(session_state)
        results["power"]["valid"] = is_valid
        results["power"]["issues"] = issues
    else:
        results["power"]["issues"] = ["Power data not available"]
    
    return results


def get_validation_summary(session_state: Dict[str, Any]) -> str:
    """
    Get a human-readable validation summary
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Formatted summary string
    """
    results = validate_all_contracts(session_state)
    
    lines = ["## Integration Contract Validation\n"]
    
    for page_name, result in results.items():
        lines.append(f"\n### {page_name.title()}")
        
        if not result["available"]:
            lines.append("- ❌ Not available")
        elif result["valid"]:
            lines.append("- ✅ Valid")
        else:
            lines.append("- ⚠️ Available but has issues:")
            for issue in result["issues"]:
                lines.append(f"  - {issue}")
    
    return "\n".join(lines)


# Sample valid states for testing
SAMPLE_VALID_RAND = {
    "design_type": "stratified",
    "arms": 3,
    "seed": 12345,
    "strata": ["region", "gender"],
    "balance_summary": "All covariates balanced at p>0.05"
}

SAMPLE_VALID_POWER = {
    "n_per_arm": 500,
    "mde": 0.15,
    "icc": 0.05,
    "assumptions": {
        "alpha": 0.05,
        "power": 0.80,
        "variance": 0.25,
        "attrition": 0.15,
        "take_up": 0.85
    },
    "notes": "Sample size calculated for primary outcome"
}


if __name__ == "__main__":
    # Test with sample data
    test_session_state = {
        "rand": SAMPLE_VALID_RAND,
        "power": SAMPLE_VALID_POWER
    }
    
    print(get_validation_summary(test_session_state))
    
    # Test with missing data
    print("\n\n--- Testing with empty state ---\n")
    print(get_validation_summary({}))
