# RCT Design Wizard

**Version:** 1.0.0

A Streamlit-based concept note builder for RCT (Randomized Controlled Trial) design documentation.

## Overview

The RCT Design Wizard is a comprehensive tool for creating structured concept notes for randomized controlled trials. It guides users through 15 sections covering all aspects of RCT design, from problem statement to compliance documentation.

## Features

- **15 Structured Sections**: Complete coverage of RCT design components
- **Auto-narrative Generation**: Automatic generation of technical narratives based on numeric inputs
- **Integration with Existing Pages**: Syncs data from Randomization and Power calculation pages
- **Project Management**: Support for multiple projects with separate saved states
- **Export to Markdown**: Generate formatted concept notes ready for review
- **Validation Warnings**: Soft warnings for common issues (e.g., >3 primary outcomes)

## Architecture

### Modules

#### `storage.py`
Handles data persistence for concept notes.

**Key Functions:**
- `load_state(project_name)`: Load a saved project state
- `save_state(state, project_name)`: Save current project state
- `list_projects()`: List all saved projects
- `delete_project(project_name)`: Delete a project
- `migrate(state)`: Handle schema version migrations

**Data Location:** `data/` directory (JSON files)

#### `adapters.py`
Manages integration with upstream pages (Randomization, Power).

**Key Functions:**
- `adapt_randomization_state(session_state)`: Extract randomization numeric values
- `adapt_power_state(session_state)`: Extract power calculation numeric values
- `check_integration_contracts(session_state)`: Validate integration availability
- `get_integration_status(session_state)`: Get human-readable integration status

**Integration Contracts:**

**Randomization Page (`st.session_state["rand"]`):**
- `design_type`: str (e.g., "simple", "stratified", "cluster")
- `arms`: int (number of treatment arms)
- `seed`: int (randomization seed)
- `strata`: list or str (strata variables)
- `balance_summary`: str or dict (balance check results)

**Power Page (`st.session_state["power"]`):**
- `n_per_arm` or `n_each`: int (sample size per arm)
- `mde`: float (minimum detectable effect)
- `icc`: float (intracluster correlation)
- `assumptions`: dict with keys:
  - `alpha`: float (default 0.05)
  - `power`: float (default 0.80)
  - `variance`: float
  - `attrition`: float
  - `take_up`: float

#### `narratives.py`
Auto-generates narrative text based on numeric inputs.

**Key Functions:**
- `generate_randomization_narrative(numeric)`: Generate randomization section narratives
- `generate_power_narrative(numeric)`: Generate power section narratives
- `should_generate_narrative(current_narrative)`: Check if auto-generation is needed

**Generated Narratives:**

For **Randomization**:
- Rationale (based on design type, strata, arms)
- Implementation steps
- Concealment strategy
- Contamination mitigation
- Clustering justification (if applicable)

For **Power**:
- Effect size justification
- Variance source explanation
- Attrition inflation strategy
- Sensitivity analyses description
- Design effect explanation (if clustered)

#### `export.py`
Handles Markdown export using Jinja2 templates.

**Key Functions:**
- `render_markdown(state)`: Render state to Markdown string
- `export_to_file(state, output_path)`: Save Markdown to file

**Template:** `template.md` (Jinja2 format)

#### `wizard.py`
Main Streamlit page with UI components.

**Key Functions:**
- `initialize_state()`: Initialize session state
- `sync_from_upstream()`: Sync data from integrated pages
- `render_section_X()`: Render each of the 15 sections
- `main()`: Main application entry point

## Schema Structure

The wizard uses a structured JSON schema (`schema.json`) with the following top-level sections:

1. **meta**: Project metadata (title, PIs, affiliations, etc.)
2. **problem_policy_relevance**: Problem statement and evidence gap
3. **theory_of_change**: ToC components (activities, outputs, outcomes)
4. **intervention_implementation**: Intervention details and delivery
5. **study_population_sampling**: Population definition and sampling frame
6. **outcomes_measurement**: Primary/secondary outcomes and measurement
7. **randomization_design**: Numeric parameters and narrative
8. **power_sample_size**: Power calculations and justifications
9. **data_collection_plan**: Survey schedule and QC protocols
10. **analysis_plan**: Estimands, models, and specifications
11. **ethics_risks_approvals**: IRB status, risks, and mitigation
12. **timeline_milestones**: Project timeline and critical path
13. **budget_summary**: Budget breakdown and funding
14. **policy_relevance_scalability**: Policy alignment and scale pathways
15. **compliance**: Pre-registration, DMP, and transparency

## Usage

### As a Standalone Page

Run the wizard directly:

```python
streamlit run rct_field_flow/rct-design/wizard.py
```

### Integrated with Main App

Add to your Streamlit app's navigation:

```python
# In app.py or main navigation
import sys
from pathlib import Path

# Add rct-design to path
sys.path.insert(0, str(Path(__file__).parent / "rct-design"))

# Import wizard
from wizard import main as wizard_main

# Add to page dictionary
pages = {
    "RCT Design Wizard": wizard_main,
    # ... other pages
}
```

### Creating a New Project

1. Enter a project name in the "New project name" field
2. Click "➕ Create New"
3. Fill in the sections
4. Click "💾 Save Progress"

### Syncing from Upstream Pages

1. Complete Randomization and/or Power calculation pages first
2. Navigate to the Wizard
3. Click "🔄 Sync from Pages" or open the wizard (auto-syncs on load)
4. Numeric fields will be auto-populated
5. Auto-narratives will be generated if fields are empty

### Exporting Concept Note

1. Fill in desired sections
2. Click "📥 Export Markdown"
3. Click "⬇️ Download Markdown" in the popup
4. Save the generated `.md` file

## Validation & Warnings

The wizard provides soft warnings for:

- **>3 Primary Outcomes**: Warns if more than 3 primary outcomes are specified
- **N per arm = 0**: Warns if sample size not specified
- **ICC without clustering**: Warns if ICC is specified but design is not clustered
- **Missing ICC for clustered design**: Warns if design is clustered but ICC is not specified

Warnings do not block saving or exporting.

## Data Storage

Projects are stored as JSON files in `rct-design/data/`:

- Format: `{project_name}.json`
- Location: `rct_field_flow/rct-design/data/`
- Encoding: UTF-8
- Includes automatic `_last_saved` timestamp

## Extending the Wizard

### Adding a New Field

1. Update `schema.json` with the new field
2. Add UI component in corresponding `render_section_X()` function
3. Update `template.md` to include the field in export
4. Increment `version` in `schema.json`
5. Add migration logic in `storage.migrate()` if needed

### Adding a New Section

1. Add section structure to `schema.json`
2. Create `render_section_X()` function in `wizard.py`
3. Add section to `render_all_sections()`
4. Add section template to `template.md`
5. Update this README

### Custom Auto-Narratives

Add generation functions to `narratives.py`:

```python
def generate_custom_narrative(numeric: Dict[str, Any]) -> Dict[str, str]:
    # Your logic here
    return {
        "field_name": "Generated narrative text"
    }
```

## Testing Integration

To verify integration with upstream pages:

```python
# In Python console or test script
import streamlit as st

# Mock upstream state
st.session_state["rand"] = {
    "design_type": "stratified",
    "arms": 3,
    "seed": 12345,
    "strata": ["region", "gender"]
}

st.session_state["power"] = {
    "n_per_arm": 500,
    "mde": 0.15,
    "icc": 0.05,
    "assumptions": {
        "alpha": 0.05,
        "power": 0.80,
        "attrition": 0.15
    }
}

# Then run wizard - it should auto-populate numeric fields
```

## Requirements

- Python 3.8+
- streamlit
- jinja2
- Standard library: json, pathlib, datetime, typing

Install dependencies:

```bash
pip install streamlit jinja2
```

## Version History

### v1.0.0 (Current)
- Initial release
- 15 structured sections
- Auto-narrative generation for randomization and power
- Integration with Randomization and Power pages
- Markdown export with Jinja2 templates
- Project management (create, save, load, list)
- Validation warnings

## Future Enhancements (Deferred)

- **LaTeX Export**: Generate LaTeX-formatted concept notes
- **Multiple Project Selection**: Manage multiple projects simultaneously
- **Pre-registration Text Generator**: Auto-generate AEA registry text
- **RAG Suggestions**: Re-integrate LLM-based suggestions for narrative improvement
- **Validation Layer Enhancements**: Hard validation rules and blocking errors
- **Budget Auto-Calculator**: Compute totals and percentages automatically
- **Timeline Visualization**: Gantt chart or timeline view
- **Risk Matrix Component**: Enhanced risk assessment UI

## Support

For issues or questions:
1. Check this README
2. Review `schema.json` for data structure
3. Check integration contracts in `adapters.py`
4. Review validation logic in `wizard.py`

## License

Same as parent project (rct_field_flow)

---

**Maintained by:** RCT Field Flow Team  
**Last Updated:** 2025-11-16
