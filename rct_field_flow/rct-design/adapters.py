"""
Adapters module for RCT Design Wizard
Handles integration with existing Randomization and Power pages
"""

from typing import Dict, Any


def adapt_randomization_state(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate randomization numeric fields from st.session_state.rand
    
    Integration contract:
    - design_type: str (e.g., "simple", "stratified", "cluster")
    - arms: int (number of treatment arms)
    - seed: int (randomization seed)
    - strata: list or str (strata variables or CSV)
    - balance_summary: str or dict (balance check results)
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Dictionary with randomization numeric values
    """
    rand_state = session_state.get("rand", {})
    
    numeric = {
        "design_type": rand_state.get("design_type"),
        "arms": rand_state.get("arms"),
        "seed": rand_state.get("seed"),
        "strata": rand_state.get("strata", []),
        "balance_summary": rand_state.get("balance_summary")
    }
    
    return numeric


def adapt_power_state(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate power numeric fields from st.session_state.power
    
    Integration contract:
    - n_per_arm (or n_each): int (sample size per arm)
    - mde: float (minimum detectable effect)
    - icc: float (intracluster correlation, if clustered)
    - assumptions: dict (alpha, power, variance, attrition, take_up)
    - notes: str (additional notes)
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Dictionary with power numeric values
    """
    power_state = session_state.get("power", {})
    assumptions = power_state.get("assumptions", {})
    
    numeric = {
        "n_per_arm": power_state.get("n_per_arm") or power_state.get("n_each"),
        "icc": power_state.get("icc"),
        "mde": power_state.get("mde"),
        "alpha": assumptions.get("alpha", 0.05),
        "power": assumptions.get("power", 0.80),
        "variance": assumptions.get("variance"),
        "attrition": assumptions.get("attrition"),
        "take_up": assumptions.get("take_up")
    }
    
    return numeric


def check_integration_contracts(session_state: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that the expected session state keys exist
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Dictionary mapping page names to availability status
    """
    contracts = {
        "randomization": "rand" in session_state,
        "power": "power" in session_state
    }
    
    return contracts


def get_integration_status(session_state: Dict[str, Any]) -> str:
    """
    Get a human-readable status message about integration
    
    Args:
        session_state: Streamlit session state dictionary
        
    Returns:
        Status message string
    """
    contracts = check_integration_contracts(session_state)
    
    available = []
    missing = []
    
    for page, available_flag in contracts.items():
        if available_flag:
            available.append(page)
        else:
            missing.append(page)
    
    status_parts = []
    
    if available:
        status_parts.append(f"✓ Connected to: {', '.join(available)}")
    
    if missing:
        status_parts.append(f"⚠ Not connected: {', '.join(missing)} (numeric fields will be editable)")
    
    return "\n".join(status_parts) if status_parts else "No integrations detected"
