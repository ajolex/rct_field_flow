"""
Storage module for RCT Design Wizard
Handles loading and saving of concept note state
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


# Default data directory
DATA_DIR = Path(__file__).parent / "data"
SCHEMA_PATH = Path(__file__).parent / "schema.json"


def ensure_data_dir() -> Path:
    """Ensure the data directory exists"""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def get_default_state() -> Dict[str, Any]:
    """Load the default state from schema.json"""
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file: {e}")


def load_state(project_name: str = "default") -> Dict[str, Any]:
    """
    Load the state for a project
    
    Args:
        project_name: Name of the project (default: "default")
        
    Returns:
        Dictionary containing the project state
    """
    ensure_data_dir()
    state_file = DATA_DIR / f"{project_name}.json"
    
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Add migration logic here if needed
            return state
        except json.JSONDecodeError as e:
            print(f"Warning: Could not load state from {state_file}: {e}")
            return get_default_state()
    else:
        # Return default state if no saved state exists
        return get_default_state()


def save_state(state: Dict[str, Any], project_name: str = "default") -> bool:
    """
    Save the current state for a project
    
    Args:
        state: Dictionary containing the project state
        project_name: Name of the project (default: "default")
        
    Returns:
        True if successful, False otherwise
    """
    ensure_data_dir()
    state_file = DATA_DIR / f"{project_name}.json"
    
    try:
        # Add timestamp to state
        state["_last_saved"] = datetime.now().isoformat()
        
        # Write to file with pretty formatting
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving state to {state_file}: {e}")
        return False


def list_projects() -> list:
    """
    List all saved projects
    
    Returns:
        List of project names
    """
    ensure_data_dir()
    projects = []
    
    for file in DATA_DIR.glob("*.json"):
        if file.stem != "schema":
            projects.append(file.stem)
    
    return sorted(projects)


def delete_project(project_name: str) -> bool:
    """
    Delete a project's saved state
    
    Args:
        project_name: Name of the project to delete
        
    Returns:
        True if successful, False otherwise
    """
    ensure_data_dir()
    state_file = DATA_DIR / f"{project_name}.json"
    
    try:
        if state_file.exists():
            state_file.unlink()
            return True
        return False
    except Exception as e:
        print(f"Error deleting project {project_name}: {e}")
        return False


def migrate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate state from older versions to current version
    
    Args:
        state: Old state dictionary
        
    Returns:
        Migrated state dictionary
    """
    current_version = "1.0.0"
    state_version = state.get("version", "0.0.0")
    
    if state_version == current_version:
        return state
    
    # Add migration logic for future versions here
    # Example:
    # if state_version == "0.9.0":
    #     state = migrate_0_9_to_1_0(state)
    
    state["version"] = current_version
    return state
