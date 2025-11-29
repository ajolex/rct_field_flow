from __future__ import annotations

import importlib.util
import io
import math
import sys
import json
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import ModuleType
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml
import streamlit_authenticator as stauth

try:
    from .assign_cases import assign_cases
    from .flag_quality import QualityResults, flag_all
    from .randomize import RandomizationConfig, RandomizationResult, Randomizer, TreatmentArm
    from .power_calc import (
        PowerAssumptions,
        calculate_mde,
        calculate_sample_size,
        generate_power_curve,
        generate_cluster_size_table,
        generate_python_code,
        generate_stata_code,
    )
    from .power_simulation import (
        SimulationAssumptions,
        run_power_simulation,
        generate_simulation_power_curve,
        generate_simulation_stata_code,
        generate_simulation_python_code,
    )
    from .monitor import (
        load_config as mon_load_config,
        load_submissions as mon_load_submissions,
        prepare_data as mon_prepare_data,
        render_dashboard as mon_render_dashboard,
    )
    from .analyze import (
        AnalysisConfig, attrition_table,
        generate_balance_table, estimate_itt, estimate_tot, estimate_late,
        estimate_heterogeneity, estimate_binary_outcome, estimate_panel_fe,
        format_regression_table, load_data, run_data_diagnostics,
        winsorize_variable, check_balance, generate_python_analysis_code,
        generate_stata_analysis_code
    )
    from .visualize import (
        plot_coefficients, plot_distributions, plot_heterogeneity,
        plot_balance, plot_kde_comparison
    )
    from .backcheck import BackcheckConfig, sample_backchecks
    from .report import generate_weekly_report
    from .surveycto import SurveyCTO
    # Persistence (successful import path)
    from .persistence import (
        init_db,
        record_user_login,
        record_activity,
        upsert_design_data,
        upsert_randomization,
        fetch_users_for_auth,
        create_user,
    )
except ImportError:  # pragma: no cover
    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from rct_field_flow.assign_cases import assign_cases  # type: ignore
    from rct_field_flow.flag_quality import QualityResults, flag_all  # type: ignore
    from rct_field_flow.randomize import (  # type: ignore
        RandomizationConfig,
        RandomizationResult,
        Randomizer,
        TreatmentArm,
    )
    from rct_field_flow.power_calc import (  # type: ignore
        PowerAssumptions,
        calculate_mde,
        calculate_sample_size,
        generate_power_curve,
        generate_cluster_size_table,
        generate_python_code,
        generate_stata_code,
    )
    from rct_field_flow.power_simulation import (  # type: ignore
        SimulationAssumptions,
        run_power_simulation,
        generate_simulation_power_curve,
        generate_simulation_stata_code,
        generate_simulation_python_code,
    )
    from rct_field_flow.monitor import (  # type: ignore
        load_config as mon_load_config,
        load_submissions as mon_load_submissions,
        prepare_data as mon_prepare_data,
        render_dashboard as mon_render_dashboard,
    )
    from rct_field_flow.analyze import (  # type: ignore
        AnalysisConfig,
        estimate_ate,
        heterogeneity_analysis,
        attrition_table,
        load_data,
        generate_balance_table,
        estimate_itt,
        estimate_tot,
        estimate_late,
        estimate_heterogeneity,
        estimate_binary_outcome,
        estimate_panel_fe,
        format_regression_table,
        run_data_diagnostics,
        winsorize_variable,
        check_balance,
        generate_python_analysis_code,
        generate_stata_analysis_code,
    )
    from rct_field_flow.backcheck import BackcheckConfig, sample_backchecks  # type: ignore
    from rct_field_flow.report import generate_weekly_report  # type: ignore
    from rct_field_flow.surveycto import SurveyCTO  # type: ignore
    from rct_field_flow.visualize import (  # type: ignore
        plot_coefficients,
        plot_distributions,
        plot_heterogeneity,
        plot_balance,
        plot_kde_comparison,
    )
    # Persistence (fallback import path)
    from rct_field_flow.persistence import (  # type: ignore
        init_db,
        record_user_login,
        record_activity,
        upsert_design_data,
        upsert_randomization,
        fetch_users_for_auth,
        create_user,
    )

# ----------------------------------------------------------------------------- #
# Page configuration & session state                                            #
# ----------------------------------------------------------------------------- #

st.set_page_config(page_title="RCT Field Flow", page_icon=":bar_chart:", layout="wide")

if "baseline_data" not in st.session_state:
    st.session_state.baseline_data: pd.DataFrame | None = None
if "randomization_result" not in st.session_state:
    st.session_state.randomization_result: RandomizationResult | None = None
if "validation_state" not in st.session_state:
    st.session_state.validation_state: Dict[str, Any] | None = None
if "case_data" not in st.session_state:
    st.session_state.case_data: pd.DataFrame | None = None
if "quality_data" not in st.session_state:
    st.session_state.quality_data: pd.DataFrame | None = None

# ===== RCT DESIGN SESSION STATE =====
if "design_data" not in st.session_state:
    st.session_state.design_data: Dict | None = None
if "design_team_name" not in st.session_state:
    st.session_state.design_team_name: str | None = None
if "design_program_card" not in st.session_state:
    st.session_state.design_program_card: str | None = None
if "design_workbook_responses" not in st.session_state:
    st.session_state.design_workbook_responses: Dict = {}

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
RCT_DESIGN_APP_DIR = Path(__file__).parent / "rct-design" / "app"


def load_default_config() -> Dict:
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def yaml_dump(data: Dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def yaml_load(text: str) -> Dict:
    return yaml.safe_load(text) if text.strip() else {}


def load_rct_design_module(module_name: str, relative_path: str) -> ModuleType:
    """Load a module from the embedded rct-design app without polluting sys.path."""
    module_path = RCT_DESIGN_APP_DIR / relative_path
    if not module_path.exists():
        raise FileNotFoundError(f"RCT design module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load spec for {module_name} ({module_path})")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_arm_state(arms_defaults: List[Dict], count: int) -> None:
    """Ensure treatment arm session state entries exist for the requested count."""
    prev = st.session_state.get("_arm_state_count", 0)
    if count < prev:
        for idx in range(count, prev):
            st.session_state.pop(f"arm_name_{idx}", None)
            st.session_state.pop(f"arm_prop_{idx}", None)

    for idx in range(count):
        if idx < len(arms_defaults):
            default_name = arms_defaults[idx].get("name", f"arm_{idx+1}")
            default_prop = float(arms_defaults[idx].get("proportion", 1.0 / count))
        else:
            default_name = "control" if idx == 0 else f"treatment_{idx}"
            default_prop = float(1.0 / count)
        st.session_state.setdefault(f"arm_name_{idx}", default_name)
        st.session_state.setdefault(f"arm_prop_{idx}", round(default_prop, 3))

    st.session_state["_arm_state_count"] = count


def generate_python_randomization_code(config: RandomizationConfig, display_method: str) -> str:
    """Generate Python code that replicates the randomization with exact parameters."""
    arms_code = ",\n        ".join([
        f'TreatmentArm(name="{arm.name}", proportion={arm.proportion})'
        for arm in config.arms
    ])
    
    strata_code = f"{config.strata}" if config.strata else "None"
    cluster_code = f'"{config.cluster}"' if config.cluster else "None"
    balance_code = f"{config.balance_covariates}" if config.balance_covariates else "None"
    
    # Generate detailed documentation
    method_description = {
        "simple": "Simple randomization: Each unit is randomly assigned independently",
        "stratified": f"Stratified randomization: Units randomized separately within each stratum (strata: {', '.join(config.strata) if config.strata else 'None'})",
        "cluster": f"Cluster randomization: Entire clusters assigned to same treatment (cluster: {config.cluster})"
    }.get(config.method, display_method)
    
    rerandom_note = ""
    if config.iterations > 1:
        rerandom_note = f'''
RERANDOMIZATION APPROACH:
- Total iterations: {config.iterations}
- For each iteration, a new random assignment is generated
- Balance is measured using ANOVA F-tests for each covariate
- The p-value represents the probability that group means are equal
- The "best min p-value" is the HIGHEST minimum p-value across all covariates
- Higher p-values indicate better balance (less likely to reject null of equal means)
- The assignment with the best balance (highest min p-value) is selected

This follows Morgan & Rubin (2012) rerandomization methodology. Note that 
rerandomization can affect inference - see Bruhn & McKenzie (2009) for discussion.
'''
    else:
        rerandom_note = '''
SINGLE RANDOMIZATION:
- No rerandomization performed (iterations = 1)
- This is standard randomization without balance optimization
'''
    
    balance_method_note = ""
    if config.balance_covariates:
        balance_method_note = f'''
BALANCE CHECKING METHOD:
- Covariates checked: {', '.join(config.balance_covariates)}
- Method: One-way ANOVA F-test for each covariate across treatment arms
- Null hypothesis: All treatment arms have equal means for the covariate
- Test statistic: F = (between-group variance) / (within-group variance)
- P-value: Probability of observing this F-statistic under null hypothesis
- Uses scipy.stats.f_oneway for computation
- For rerandomization: selects assignment with highest minimum p-value across all covariates
'''
    
    strata_note = ""
    if config.strata and config.method in ["stratified", "cluster"]:
        strata_note = f'''
STRATIFICATION DETAILS:
- Randomization performed separately within each stratum combination
- Strata variables: {', '.join(config.strata)}
- Treatment proportions maintained within each stratum
- This ensures balanced treatment assignment across strata
- Total number of strata: depends on unique combinations in your data
'''
    
    cluster_note = ""
    if config.cluster and config.method == "cluster":
        cluster_note = f'''
CLUSTER RANDOMIZATION DETAILS:
- Cluster variable: {config.cluster}
- All units within the same cluster receive the same treatment
- Randomization occurs at the cluster level, not individual level
- This accounts for intra-cluster correlation in treatment effects
- Number of clusters randomized: depends on unique clusters in your data
'''
    
    code = f'''"""
================================================================================
RANDOMIZATION CODE - RCT Field Flow
================================================================================
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Method: {display_method}

RANDOMIZATION SPECIFICATION:
- ID Column: {config.id_column}
- Treatment Column: {config.treatment_column}
- Method: {method_description}
- Random Seed: {config.seed} (for exact replication)
- Use Existing Assignment: {config.use_existing_assignment}

TREATMENT ARMS:
{chr(10).join([f"- {arm.name}: {arm.proportion*100:.1f}% ({arm.proportion} proportion)" for arm in config.arms])}
{rerandom_note}{balance_method_note}{strata_note}{cluster_note}
IMPORTANT ASSUMPTIONS:
1. Random seed ensures reproducibility - same seed + data = same assignment
2. Treatment proportions are approximate - exact counts depend on sample size
3. For stratified randomization, each stratum maintains treatment proportions
4. For cluster randomization, all units in a cluster get the same treatment
5. Missing data in strata or cluster variables will cause assignment to fail
6. Balance checking uses parametric tests (ANOVA) - assumes approximate normality

REFERENCES:
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate 
  balance in experiments. Annals of Statistics, 40(2), 1263-1282.
- Bruhn, M., & McKenzie, D. (2009). In pursuit of balance: Randomization in 
  practice in development field experiments. American Economic Journal: Applied
  Economics, 1(4), 200-232.

================================================================================
"""

import pandas as pd
from rct_field_flow.randomize import RandomizationConfig, Randomizer, TreatmentArm

# Load your baseline data
df = pd.read_csv("baseline_data.csv")  # Update with your actual file path

# Validate data before randomization
print(f"Loaded {{len(df):,}} observations")
print(f"ID column '{{config.id_column}}' has {{df['{config.id_column}'].nunique()}} unique values")

# Check for missing values in critical columns
critical_cols = ['{config.id_column}']
{f"critical_cols.extend({config.strata})" if config.strata else ""}
{f"critical_cols.append('{config.cluster}')" if config.cluster else ""}
{f"critical_cols.extend({config.balance_covariates})" if config.balance_covariates else ""}

missing_summary = df[critical_cols].isnull().sum()
if missing_summary.sum() > 0:
    print("\\nWARNING: Missing values detected in critical columns:")
    print(missing_summary[missing_summary > 0])
    print("Consider imputing or dropping these observations before randomization.\\n")

# Randomization configuration
config = RandomizationConfig(
    id_column="{config.id_column}",
    treatment_column="{config.treatment_column}",
    method="{config.method}",
    arms=[
        {arms_code}
    ],
    strata={strata_code},
    cluster={cluster_code},
    balance_covariates={balance_code},
    iterations={config.iterations},
    seed={config.seed},
    use_existing_assignment={config.use_existing_assignment}
)

# Run randomization
print("\\n" + "="*80)
print("RUNNING RANDOMIZATION")
print("="*80 + "\\n")

randomizer = Randomizer(config)
result = randomizer.run(df, verbose=True)

# Display results
print("\\n" + "="*80)
print("RANDOMIZATION COMPLETE")
print("="*80)
print(f"Iterations run: {{result.iterations}}")
print(f"Best min p-value achieved: {{result.best_min_pvalue:.4f}}")
{'print("  → Interpretation: Higher is better (less evidence of imbalance)")' if config.balance_covariates else ''}
print(f"Used existing assignment: {{result.used_existing_assignment}}")
print()

# Treatment distribution
print("-" * 80)
print("TREATMENT DISTRIBUTION")
print("-" * 80)
treatment_counts = result.assignments["{config.treatment_column}"].value_counts()
treatment_props = result.assignments["{config.treatment_column}"].value_counts(normalize=True)
for arm in treatment_counts.index:
    count = treatment_counts[arm]
    prop = treatment_props[arm]
    print(f"{{arm:20s}}: {{count:6d}} observations ({{prop*100:5.1f}}%)")
print(f"{'Total':20s}: {{len(result.assignments):6d}} observations")
print()

# Distribution by strata (if applicable)
{f'''strata_vars = {config.strata}
if strata_vars:
    print("-" * 80)
    print("TREATMENT DISTRIBUTION BY STRATA")
    print("-" * 80)
    crosstab = pd.crosstab(
        [result.assignments[col] for col in strata_vars],
        result.assignments["{config.treatment_column}"],
        margins=True
    )
    print(crosstab)
    print()
''' if config.strata else ''}

# Balance table (if covariates specified)
if not result.balance_table.empty:
    print("-" * 80)
    print("BALANCE TABLE")
    print("-" * 80)
    print("Columns: variable name, F-statistic, p-value, mean by treatment arm")
    print("P-value interpretation: >0.05 indicates no significant imbalance")
    print()
    print(result.balance_table)
    print()
    
    # Highlight any concerning imbalances
    min_pval = result.balance_table['p_value'].min()
    if min_pval < 0.05:
        print("⚠ WARNING: Some covariates show significant imbalance (p < 0.05)")
        imbalanced = result.balance_table[result.balance_table['p_value'] < 0.05]
        print("Imbalanced covariates:")
        print(imbalanced[['p_value']])
    else:
        print("✓ All covariates are reasonably balanced (all p-values >= 0.05)")
    print()

# Save assignments
output_file = "randomized_assignments.csv"
result.assignments.to_csv(output_file, index=False)
print("-" * 80)
print(f"Assignments saved to {{output_file}}")
print(f"Total observations: {{len(result.assignments):,}}")
print("-" * 80)
'''
    return code


def generate_python_validation_code(config: RandomizationConfig, n_simulations: int) -> str:
    """Generate Python code that replicates the validation workflow."""

    arms_code = ",\n        ".join(
        [
            f'TreatmentArm(name="{arm.name}", proportion={arm.proportion})'
            for arm in config.arms
        ]
    )
    strata_code = json.dumps(config.strata)
    balance_code = json.dumps(config.balance_covariates)
    cluster_code = f'"{config.cluster}"' if config.cluster else "None"
    seed_literal = "None" if config.seed is None else str(config.seed)

    code = f'''"""
================================================================================
RANDOMIZATION VALIDATION CODE - RCT Field Flow
================================================================================
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Simulations: {int(n_simulations):,}

This script reruns the validation exactly as configured in the app.
Update the data path and re-run to reproduce the results.
================================================================================
"""

import pandas as pd
from rct_field_flow.randomize import RandomizationConfig, Randomizer, TreatmentArm

# Load your baseline data
df = pd.read_csv("baseline_data.csv")  # TODO: replace with your dataset path

# Mirror the validation configuration from the app
config = RandomizationConfig(
    id_column="{config.id_column}",
    treatment_column="{config.treatment_column}",
    method="{config.method}",
    arms=[
        {arms_code}
    ],
    strata={strata_code},
    cluster={cluster_code},
    balance_covariates={balance_code},
    iterations={config.iterations},
    seed={seed_literal},
    use_existing_assignment={config.use_existing_assignment}
)

randomizer = Randomizer(config)

result = randomizer.validate_randomization(
    df=df,
    n_simulations={int(n_simulations)},
    base_seed={seed_literal},
    verbose=False,
)

status = "PASSED" if result.is_valid else "FAILED"
print(f"Validation {{status}}")

summary_df = (
    pd.DataFrame.from_dict(result.summary_stats, orient="index")
    .reset_index()
    .rename(columns=dict(index="Treatment Arm"))
)

print("\\nAssignment probability summary:")
print(summary_df)

if result.warnings:
    print("\\nWarnings detected during validation:")
    for warning in result.warnings:
        print(f" - {{warning}}")
else:
    print("\\nNo warnings detected.")

if result.assignment_probabilities.empty:
    print("\nAssignment probabilities table is empty; no files generated.")
else:
    result.assignment_probabilities.to_csv("validation_results.csv", index=False)
    print("\nSaved detailed probabilities to validation_results.csv")

    first_arm = summary_df.loc[0, "Treatment Arm"]
    prob_col = f"prob_{{first_arm}}"

    if prob_col in result.assignment_probabilities.columns:
        import matplotlib.pyplot as plt

        probs = result.assignment_probabilities[prob_col]
        expected_prob = result.summary_stats[first_arm]["expected"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(probs, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(expected_prob, color="red", linestyle="--", linewidth=2,
                   label=f"Expected: {{expected_prob:.3f}}")
        ax.set_xlabel(f"P(assignment to {{first_arm}})")
        ax.set_ylabel("Number of observations")
        ax.set_title(f"Assignment Probability Distribution - {{first_arm}}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("validation_histogram.png", dpi=300)
        plt.close(fig)
        print("Saved histogram to validation_histogram.png")
    else:
        print(f"Column {{prob_col}} not found; skipping histogram export.")
'''
    return code


def generate_stata_validation_code(config: RandomizationConfig, n_simulations: int) -> str:
    """Generate a pure Stata do-file that validates randomization fairness."""
    
    # Generate treatment arm info
    arms_names = [arm.name for arm in config.arms]
    arms_props = [arm.proportion for arm in config.arms]
    
    # Generate cumulative proportions for assignment
    cumulative_props = []
    cum = 0.0
    for prop in arms_props[:-1]:
        cum += prop
        cumulative_props.append(f"{cum:.6f}")
    
    # Generate treatment assignment code
    treatment_assign = "gen double rand_draw = runiform()\n"
    treatment_assign += f"gen str20 {config.treatment_column}_sim = \"\"\n"
    
    for idx, (name, cutoff) in enumerate(zip(arms_names[:-1], cumulative_props)):
        if idx == 0:
            treatment_assign += f'    replace {config.treatment_column}_sim = "{name}" if rand_draw < {cutoff}\n'
        else:
            prev_cutoff = cumulative_props[idx-1]
            treatment_assign += f'    replace {config.treatment_column}_sim = "{name}" if rand_draw >= {prev_cutoff} & rand_draw < {cutoff}\n'
    
    last_name = arms_names[-1]
    if len(cumulative_props) > 0:
        treatment_assign += f'    replace {config.treatment_column}_sim = "{last_name}" if rand_draw >= {cumulative_props[-1]}\n'
    else:
        treatment_assign += f'    replace {config.treatment_column}_sim = "{last_name}"\n'
    
    # Strata handling
    strata_code = ""
    if config.strata and config.method in ["stratified", "cluster"]:
        strata_vars = " ".join(config.strata)
        strata_code = f'''
* Stratified randomization - randomize within each stratum
bysort {strata_vars} (rand_draw): replace {config.treatment_column}_sim = ""
bysort {strata_vars} (rand_draw): replace rand_draw = runiform()
sort {strata_vars} rand_draw
bysort {strata_vars}: '''
        # Indent the assignment code for stratified case
        treatment_assign = "\n".join(["    " + line if line.strip() else line 
                                       for line in treatment_assign.split("\n")])
    
    # Cluster handling
    cluster_code = ""
    if config.cluster and config.method == "cluster":
        cluster_var = config.cluster
        cluster_code = f'''
* Cluster randomization - assign treatment at cluster level
* First, get unique clusters
preserve
keep {cluster_var}
duplicates drop
gen cluster_id = _n
tempfile clusters
save `clusters'
restore

* Merge back to get cluster assignments
merge m:1 {cluster_var} using `clusters', nogenerate

* Randomize clusters (not individuals)
gen double cluster_rand = runiform()
sort cluster_rand
'''
    
    # Base seed
    base_seed = config.seed if config.seed else 12345
    
    code = f'''********************************************************************************
* RANDOMIZATION VALIDATION DO-FILE - RCT Field Flow (Pure Stata)
********************************************************************************
* Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
* Method: {config.method}
* Simulations: {int(n_simulations):,}
* Base seed: {base_seed}
*
* PURPOSE:
*   Validate randomization fairness by running it {int(n_simulations):,} times with 
*   different seeds. Each observation should have approximately equal probability 
*   of being assigned to each treatment arm across all simulations.
*
* INSTRUCTIONS:
*   1. Update the data import section below with your dataset path
*   2. Run this script to generate validation results
*   3. Check that average assignment probabilities match expected proportions
*   4. Review histogram to ensure binomial-like distribution
*
* REFERENCE:
*   This follows best practices from RANDOMIZATION.md guide:
*   "Run your randomization a few hundred times with different seeds and 
*    compare the outcomes... if some observations are almost always in 
*    treatment or almost always in control – check your randomization code"
********************************************************************************

clear all
set more off
version 14.0

* TODO: Import your baseline data
* Example for CSV:
*   import delimited "baseline_data.csv", clear
* Example for Stata:
*   use "baseline_data.dta", clear

di _n(2) "================================================================================"
di "RANDOMIZATION VALIDATION - RCT Field Flow"
di "================================================================================"
di "Method: {config.method}"
di "Treatment arms: {', '.join([f'{name} ({prop*100:.1f}%)' for name, prop in zip(arms_names, arms_props)])}"
di "ID column: {config.id_column}"
di "Simulations: {int(n_simulations):,}"
di "================================================================================" _n

* Validate data
confirm variable {config.id_column}
quietly count if missing({config.id_column})
if r(N) > 0 {{
    di as error "ERROR: " r(N) " observations have missing ID"
    exit 198
}}

quietly duplicates report {config.id_column}
if r(unique_value) != r(N) {{
    di as error "ERROR: Duplicate IDs found in {config.id_column}"
    exit 198
}}

* Initialize probability tracking variables
gen long obs_id = _n
{' '.join([f'gen double prob_{name} = 0' for name in arms_names])}

di "Running {int(n_simulations):,} simulations..."

* Run validation simulations
forvalues sim = 1/{int(n_simulations)} {{
    
    quietly {{
        * Set seed for this simulation
        local seed = {base_seed} + `sim' - 1
        set seed `seed'
        
        {cluster_code if config.cluster and config.method == "cluster" else ""}
        {strata_code if config.strata and config.method in ["stratified", "cluster"] else ""}
        * Generate random assignment for this simulation
        {treatment_assign}
        
        * Count assignments for each arm
        {' '.join([f'''
        count if {config.treatment_column}_sim == "{name}"
        replace prob_{name} = prob_{name} + 1 if {config.treatment_column}_sim == "{name}"''' 
        for name in arms_names])}
        
        * Clean up simulation variables
        drop rand_draw {config.treatment_column}_sim
        {f'drop cluster_rand cluster_id' if config.cluster and config.method == "cluster" else ""}
    }}
    
    * Progress indicator
    if mod(`sim', {max(1, int(n_simulations) // 10)}) == 0 {{
        di "  Completed `sim'/{int(n_simulations)} simulations"
    }}
}}

di _n "Converting counts to probabilities..."

* Convert counts to probabilities
{' '.join([f'replace prob_{name} = prob_{name} / {int(n_simulations)}' for name in arms_names])}

* Calculate average probability (for histogram)
gen double avg_prob = ({' + '.join([f'prob_{name}' for name in arms_names])}) / {len(arms_names)}

di _n "================================================================================"
di "VALIDATION RESULTS"
di "================================================================================" _n

* Summary statistics for each arm
{chr(10).join([f'''
di "Treatment Arm: {name} (Expected: {prop*100:.1f}%%)"
quietly summarize prob_{name}, detail
di "  Mean probability:  " %%6.4f r(mean) " (expected: {prop:.4f})"
di "  Std deviation:     " %%6.4f r(sd)
di "  Min probability:   " %%6.4f r(min)
di "  Max probability:   " %%6.4f r(max)
local mean_{name.replace(" ", "_")} = r(mean)
local expected_{name.replace(" ", "_")} = {prop}
if abs(`mean_{name.replace(" ", "_")}' - `expected_{name.replace(" ", "_")}') > 0.05 {{
    di as error "  ⚠ WARNING: Mean deviates from expected by more than 5%%!"
}}
di ""''' for name, prop in zip(arms_names, arms_props)])}

* Check for extreme probabilities
local warning_count = 0
{chr(10).join([f'''
quietly count if prob_{name} < {prop} - 0.30 | prob_{name} > {prop} + 0.30
if r(N) > 0 {{
    di as error "⚠ WARNING: " r(N) " observations have extreme probabilities for '{name}'"
    di as error "  (>30%% deviation from expected {prop*100:.1f}%%)"
    local warning_count = `warning_count' + 1
}}''' for name, prop in zip(arms_names, arms_props)])}

di _n "================================================================================"
if `warning_count' == 0 {{
    di as text "✓ VALIDATION PASSED - No issues detected"
    di as text "  Randomization appears fair and unbiased"
}} else {{
    di as error "✗ VALIDATION FAILED - `warning_count' warning(s) detected"
    di as error "  Please review your randomization code"
}}
di "================================================================================" _n

* Export results
preserve
keep {config.id_column} obs_id prob_* avg_prob
export delimited using "validation_results.csv", replace
di "Saved detailed results to: validation_results.csv"
restore

* Generate histogram for first treatment arm
histogram prob_{arms_names[0]}, ///
    width({1.0 / (int(n_simulations) / 20)}) ///
    frequency ///
    title("Assignment Probability Distribution - {arms_names[0]}") ///
    subtitle("{int(n_simulations):,} simulations") ///
    xtitle("Probability of assignment to {arms_names[0]}") ///
    ytitle("Number of observations") ///
    xline({arms_props[0]}, lcolor(red) lwidth(medium) lpattern(dash)) ///
    note("Expected probability: {arms_props[0]:.3f} (red dashed line)" ///
         "Each observation should have approximately equal probability" ///
         "Histogram should resemble binomial distribution", size(small)) ///
    scheme(s2color)

graph export "validation_histogram.png", replace width(2400) height(1600)
di "Saved histogram to: validation_histogram.png"

di _n "================================================================================"
di "INTERPRETATION GUIDE"
di "================================================================================"
di "1. Mean probabilities should match expected proportions (±5%)"
di "2. Standard deviation should be reasonable (not too high)"
di "3. No observations should have extreme probabilities (±30%)"
di "4. Histogram should look like a binomial distribution"
di "5. If validation fails, there may be systematic bias in randomization code"
di "================================================================================" _n
'''
    return code


def generate_stata_randomization_code(config: RandomizationConfig, display_method: str) -> str:
    """Generate Stata do-file that replicates the randomization with exact parameters."""
    
    # Generate treatment arm assignment code
    arms_names = [arm.name for arm in config.arms]
    cumulative_props = []
    cum = 0.0
    for prop in config.arms[:-1]:  # All except last
        cum += prop.proportion
        cumulative_props.append(f"{cum:.6f}")
    
    treatment_code = "gen double random_draw = runiform()\n"
    treatment_code += f"gen {config.treatment_column} = \"\"\n"
    
    # Add detailed comments for treatment assignment logic
    treatment_code += "\n* Treatment assignment logic:\n"
    treatment_code += "* runiform() generates uniform random number between 0 and 1\n"
    treatment_code += "* Each treatment arm assigned based on cumulative probability thresholds\n"
    
    for idx, (name, cutoff) in enumerate(zip(arms_names[:-1], cumulative_props)):
        prop_pct = config.arms[idx].proportion * 100
        if idx == 0:
            treatment_code += f'replace {config.treatment_column} = "{name}" if random_draw < {cutoff}  // {prop_pct:.1f}% of sample\n'
        else:
            prev_cutoff = cumulative_props[idx-1]
            treatment_code += f'replace {config.treatment_column} = "{name}" if random_draw >= {prev_cutoff} & random_draw < {cutoff}  // {prop_pct:.1f}% of sample\n'
    
    # Last arm gets the remainder
    last_name = arms_names[-1]
    last_prop_pct = config.arms[-1].proportion * 100
    if len(cumulative_props) > 0:
        treatment_code += f'replace {config.treatment_column} = "{last_name}" if random_draw >= {cumulative_props[-1]}  // {last_prop_pct:.1f}% of sample\n'
    else:
        treatment_code += f'replace {config.treatment_column} = "{last_name}"  // {last_prop_pct:.1f}% of sample\n'
    
    # Strata handling
    strata_prefix = ""
    if config.strata and config.method in ["stratified", "cluster"]:
        strata_vars = " ".join(config.strata)
        strata_prefix = f"bysort {strata_vars}: "
        strata_comment = f"* Stratified by: {', '.join(config.strata)}"
    else:
        strata_comment = "* No stratification"
    
    # Cluster handling
    cluster_comment = ""
    cluster_code = ""
    if config.cluster and config.method == "cluster":
        cluster_comment = f"\n* Clustered by: {config.cluster}"
        if config.strata:
            cluster_code = f'''
* First, randomize at cluster level within strata
preserve
bysort {" ".join(config.strata)} {config.cluster}: gen cluster_tag = _n == 1
keep if cluster_tag == 1
{strata_prefix}{treatment_code}
keep {" ".join(config.strata)} {config.cluster} {config.treatment_column}
tempfile cluster_assignments
save `cluster_assignments'
restore

* Merge cluster assignments back to individual level
merge m:1 {" ".join(config.strata)} {config.cluster} using `cluster_assignments', nogen
'''
        else:
            cluster_code = f'''
* First, randomize at cluster level
preserve
bysort {config.cluster}: gen cluster_tag = _n == 1
keep if cluster_tag == 1
{treatment_code}
keep {config.cluster} {config.treatment_column}
tempfile cluster_assignments
save `cluster_assignments'
restore

* Merge cluster assignments back to individual level
merge m:1 {config.cluster} using `cluster_assignments', nogen
'''
    
    # Balance check code
    balance_code = ""
    if config.balance_covariates:
        balance_vars = " ".join(config.balance_covariates)
        balance_code = f'''
* Balance checks
foreach var of varlist {balance_vars} {{
    di _n "Balance check for `var':"
    oneway `var' {config.treatment_column}, tabulate
}}
'''
    
    # Generate method description
    method_description = {
        "simple": "Simple randomization: Each unit randomly assigned independently",
        "stratified": f"Stratified randomization: Within-stratum random assignment (strata: {', '.join(config.strata) if config.strata else 'None'})",
        "cluster": f"Cluster randomization: All units in same cluster get same treatment (cluster: {config.cluster})"
    }.get(config.method, display_method)
    
    rerandom_note = ""
    if config.iterations > 1:
        rerandom_note = f'''
* RERANDOMIZATION APPROACH:
*   - Total iterations: {config.iterations}
*   - Each iteration generates a new random assignment
*   - Balance measured using ANOVA F-tests for specified covariates
*   - Assignment with best balance (highest minimum p-value) is selected
*   - Higher p-values indicate better balance across treatment arms
*   - This approach follows Morgan & Rubin (2012) methodology
*   - Note: Rerandomization affects inference (see Bruhn & McKenzie 2009)
'''
    else:
        rerandom_note = '''
* SINGLE RANDOMIZATION (no rerandomization performed)
'''
    
    balance_note = ""
    if config.balance_covariates:
        balance_note = f'''
* BALANCE CHECKING:
*   - Covariates: {', '.join(config.balance_covariates)}
*   - Method: One-way ANOVA F-test (oneway command in Stata)
*   - Null hypothesis: All treatment arms have equal means
*   - P-value interpretation: >0.05 indicates acceptable balance
'''
    
    code = f'''/*
================================================================================
RANDOMIZATION CODE - RCT Field Flow (Stata Implementation)
================================================================================
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Method: {display_method}

RANDOMIZATION SPECIFICATION:
  - ID Column: {config.id_column}
  - Treatment Column: {config.treatment_column}
  - Method: {method_description}
  - Random Seed: {config.seed} (for exact replication)

TREATMENT ARMS:
{chr(10).join([f"  - {arm.name}: {arm.proportion*100:.1f}% ({arm.proportion} proportion)" for arm in config.arms])}
{rerandom_note}{balance_note}
IMPORTANT ASSUMPTIONS:
  1. Random seed ensures reproducibility (set seed {config.seed})
  2. runiform() generates uniform [0,1] random numbers
  3. Treatment proportions are approximate targets
  4. For stratified: randomization done separately within each stratum
  5. For cluster: all units in same cluster receive same treatment
  6. Missing values in key variables will cause errors - clean data first

RANDOMIZATION METHOD DETAILS:
  - Uses Stata's runiform() which generates pseudo-random uniform variates
  - Treatment assignment based on cumulative probability thresholds
  - For N arms with proportions p1, p2, ..., pN (summing to 1.0):
      * Arm 1 assigned if random_draw < p1
      * Arm 2 assigned if p1 <= random_draw < (p1+p2)
      * Arm k assigned if sum(p1...p(k-1)) <= random_draw < sum(p1...pk)
  - This ensures expected proportions match specified proportions

REFERENCES:
  - Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate
    balance in experiments. Annals of Statistics, 40(2), 1263-1282.
  - Bruhn, M., & McKenzie, D. (2009). In pursuit of balance: Randomization in
    practice in development field experiments. American Economic Journal: 
    Applied Economics, 1(4), 200-232.

================================================================================
*/

clear all
set more off
set seed {config.seed}

di _n(2) "================================================================================"
di "RANDOMIZATION SCRIPT - RCT Field Flow"
di "================================================================================"
di "Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}"
di "Method: {display_method}"
di "Seed: {config.seed}"
di "================================================================================" _n

* Load baseline data
di "Loading baseline data..."
import delimited "baseline_data.csv", clear  // Update with your actual file path
di "Observations loaded: " _N _n

* Validate data
di "Validating data..." _n
count if missing({config.id_column})
if r(N) > 0 {{
    di as error "WARNING: " r(N) " observations have missing ID ({config.id_column})"
}}

'''
    
    # Add strata validation if applicable
    if config.strata:
        for var in config.strata:
            code += f'''* Check {var} (stratum variable)
count if missing({var})
if r(N) > 0 {{
    di as error "WARNING: " r(N) " observations have missing {var}"
}}

'''
    
    # Add cluster validation if applicable  
    if config.cluster:
        code += f'''* Check cluster variable
count if missing({config.cluster})
if r(N) > 0 {{
    di as error "WARNING: " r(N) " observations have missing cluster ({config.cluster})"
}}

'''
    
    code += f'''{strata_comment}{cluster_comment}

di _n "================================================================================"
di "TREATMENT ARMS: {', '.join([f'{arm.name} ({arm.proportion*100:.1f}%)' for arm in config.arms])}"
di "================================================================================" _n

'''

    # Add rerandomization loop if iterations > 1
    if config.iterations > 1 and config.balance_covariates:
        balance_vars = " ".join(config.balance_covariates)
        
        code += f'''
* RERANDOMIZATION WITH BALANCE OPTIMIZATION
* Initialize tracking variables
gen mostbalanced_{config.treatment_column} = .
local minp = 1           // Track minimum p-value across covariates in each iteration
local bestp = 0          // Track best (highest) minimum p-value across all iterations
local best_iter = 0      // Track which iteration gave best balance

di _n "================================================================================"
di "RUNNING {config.iterations:,} RANDOMIZATION ITERATIONS"
di "================================================================================"
di "Finding assignment with best balance across covariates..."
di "Balance covariates: {', '.join(config.balance_covariates)}" _n

* Run {config.iterations:,} randomization iterations
forvalues i = 1/{config.iterations} {{
    * Display progress every 1000 iterations
    if mod(`i', 1000) == 0 {{
        di "  Iteration " %%6.0f `i' " / {config.iterations:,} (best p-value so far: " %%6.4f `bestp' ")"
    }}
    
    * Generate random numbers and sort within strata
    gen rand = runiform()
    {f'sort {" ".join(config.strata)} rand, stable' if config.strata else 'sort rand'}
    
    * Assign treatments{' within strata' if config.strata else ''}
    gen {config.treatment_column}_temp = ""
{chr(10).join([f'    {"bysort " + " ".join(config.strata) + ": " if config.strata else ""}replace {config.treatment_column}_temp = "{arm.name}" if _n <= {sum([a.proportion for a in config.arms[:idx+1]]):.6f} * _N' if idx < len(config.arms) - 1 else f'    replace {config.treatment_column}_temp = "{arm.name}" if {config.treatment_column}_temp == ""' for idx, arm in enumerate(config.arms)])}
    
    * Reset minimum p-value for this iteration
    local minp = 1
    
    * Check balance across all specified covariates
    foreach var of varlist {balance_vars} {{
        qui reg `var' i.{config.treatment_column}_temp, robust
        qui test {' '.join([f'{idx}.{config.treatment_column}_temp' for idx in range(1, len(config.arms))])}
        local pvalue = r(p)
        
        * Track the minimum p-value across all covariates in this iteration
        if `pvalue' < `minp' {{
            local minp = `pvalue'
        }}
    }}
    
    * Update if this randomization has better (higher) minimum p-value
    if `minp' > `bestp' {{
        local bestp = `minp'
        local best_iter = `i'
        replace mostbalanced_{config.treatment_column} = {config.treatment_column}_temp
    }}
    
    * Clean up temporary variables
    drop rand {config.treatment_column}_temp
}}

di _n "================================================================================"
di "RERANDOMIZATION COMPLETE"
di "================================================================================"
di "Best iteration: " `best_iter' " out of {config.iterations:,}"
di "Best min p-value: " %%6.4f `bestp'
di "  → Higher p-values indicate better balance"
di "  → This is the MINIMUM p-value across all tested covariates"
di "  → Selected assignment has best overall balance" _n

* Use the best balanced assignment
gen {config.treatment_column} = mostbalanced_{config.treatment_column}
drop mostbalanced_{config.treatment_column}

* Label treatment arms
label define {config.treatment_column}_lbl {' '.join([f'{idx} "{arm.name}"' for idx, arm in enumerate(config.arms)])}
label values {config.treatment_column} {config.treatment_column}_lbl
'''
    else:
        # Single randomization (no rerandomization)
        if cluster_code:
            code += cluster_code
        else:
            code += f"* Generate random treatment assignment\n{strata_prefix}{treatment_code}"
    
    code += f'''
* Treatment distribution
tab {config.treatment_column}
'''
    
    if config.strata:
        code += f'''
* Treatment distribution by strata
table {" ".join(config.strata)} {config.treatment_column}
'''
    
    code += balance_code
    
    # Add summary statistics
    code += f'''
di _n "================================================================================"
di "TREATMENT DISTRIBUTION SUMMARY"
di "================================================================================" _n

* Overall distribution
tab {config.treatment_column}, missing

* Calculate and display proportions
di _n "Treatment proportions:"
foreach arm in {' '.join([f'"{arm.name}"' for arm in config.arms])} {{
    count if {config.treatment_column} == `arm'
    local n_`arm' = r(N)
    local pct_`arm' = (r(N) / _N) * 100
    di "  `arm': " r(N) " observations (" %%4.1f `pct_`arm'' "%%)"
}}

'''
    
    # Add strata distribution if applicable
    if config.strata:
        strata_list = " ".join(config.strata)
        code += f'''
di _n "Treatment distribution by strata:"
table {strata_list} {config.treatment_column}, row col

'''
    
    # Add cluster summary if applicable
    if config.cluster:
        code += f'''
di _n "Cluster summary:"
distinct {config.cluster}
di "Total clusters: " r(ndistinct)
preserve
bysort {config.cluster}: gen cluster_tag = _n == 1
keep if cluster_tag == 1
tab {config.treatment_column}
di "Clusters per treatment arm shown above"
restore

'''
    
    code += f'''
di _n "================================================================================"
di "BALANCE CHECK SUMMARY"
di "================================================================================" _n

* Check balance on specified covariates
{f'''foreach var of varlist {' '.join(config.balance_covariates)} {{
    di _n "Balance check for `var':"
    di "  Null hypothesis: Treatment arms have equal means"
    oneway `var' {config.treatment_column}, tabulate
    
    * Highlight imbalance
    if r(p) < 0.05 {{
        di as error "  ⚠ WARNING: Significant imbalance detected (p = " %%6.4f r(p) ")"
    }}
    else {{
        di as text "  ✓ Acceptable balance (p = " %%6.4f r(p) ")"
    }}
}}''' if config.balance_covariates else '* No balance covariates specified'}

di _n "================================================================================"
di "SAVING RESULTS"
di "================================================================================" _n

* Save randomized assignments
export delimited using "randomized_assignments.csv", replace
di "Assignments saved to: randomized_assignments.csv"
di "Total observations: " _N
di "Treatment column: {config.treatment_column}"

di _n "================================================================================"
di "RANDOMIZATION COMPLETE"
di "================================================================================"
di "Review the output above to verify:"
di "  1. Treatment proportions match specified targets"
di "  2. No excessive missing data warnings"
di "  3. Balance tests show acceptable p-values (>0.05 preferred)"
{f'di "  4. All {" x ".join([f"N({var})" for var in config.strata])} strata combinations have observations"' if config.strata else ''}
{'di "  5. All clusters are assigned to a single treatment arm"' if config.cluster else ''}
di "================================================================================" _n
'''
    
    return code


# ----------------------------------------------------------------------------- #
# HOME                                                                          #
# ----------------------------------------------------------------------------- #


def render_home() -> None:
    st.title("📊 RCT Field Flow")
    st.markdown(
        """
        **Integrated toolkit** for designing RCTs, statistical power analysis, conducting randomization, managing enumerator interview assignments, 
        quality assurance, analysis, and live monitoring.
        """
    )

    # New workflow overview with RCT Design Wizard
    st.markdown("## 📋 Complete RCT Workflow")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Phase 1: Design & Planning
        1. **🎯 RCT Design** – Build your concept note with guided prompts and tips
           - Create comprehensive designs for education, health, agriculture projects and other sectors
           - View realistic sample concept notes from different sectors
           - Export concept note in multiple formats (Markdown, DOCX, PDF)
        
        ### Phase 2: Technical Setup
        2. **⚡ Power Calculations** – Determine sample size and power
           - Calculate minimum detectable effects (MDE)
           - Run power simulations with custom assumptions
           - Generate power analysis code (Stata/Python)
        
        3. **🎲 Randomization** – Configure random assigment arms, strata, rerandomization
           - Set up treatment arms and stratification variables
           - Support for clustered and cross-clustered designs
           - Real-time covariates balance checking
        
        ### Phase 3: Implementation
        4. **📋 SurveyCTO Case Assignment** – Build SurveyCTO-ready cases dataset
           - Assign interview cases to enumerators
           - Export case assignment files for SurveyCTO
           - Supports direct upload to SurveyCTO via API
        
        5. **🔍 Quality Checks** – Apply speed, outlier, duplicate checks
           - Monitor data quality during collection
           - Flag and resolve issues in real-time
           - Generate quality reports
        
        6. **📈 Monitoring Dashboard** – Track live productivity
           - Monitor survey completion rates
           - Track enumerator performance
           - Project completion timelines
        
        ### Phase 4: Analysis & Reporting
        7. **📊 Analysis & Results** – Estimate treatment effects
           - Calculate average treatment effects (ATE)
           - Analyze treatment effect heterogeneity
           - Generate publication-ready tables
        
        8. **🔍 Backcheck Selection** – Quality validation
           - Select representative sample for backchecks
           - Ensure data integrity
        
        9. **📄 Report Generation** – Create weekly reports
           - Auto-generate monitoring summaries
           - Share with stakeholders
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Start")
        st.info("""
        **Getting Started:**
        
        Start with the **🎯 RCT Design** page to:
        - Create your concept note
        - View samples from similar projects
        - Export for stakeholder review
        
        Then proceed to **⚡ Power** for sample size calculations.
        
        **Tips:**
        - All pages auto-save your work
        - Use sample data to test features
        - Refer to guides in each section
        """)
        
        st.markdown("---")
        
        # Quick navigation
        st.markdown("### 📍 Quick Navigation")
        if st.button("→ Go to RCT Design", use_container_width=True):
            st.session_state.current_page = "design"
            st.rerun()
        
        if st.button("→ Go to Power Calculations", use_container_width=True):
            st.session_state.current_page = "power"
            st.rerun()
        
        if st.button("→ Go to Randomization", use_container_width=True):
            st.session_state.current_page = "random"
            st.rerun()

    st.markdown("---")
    
    # Features highlight
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📚 Education Example**
        - Remedial literacy programs
        - Malawi primary schools
        - 3,200 students, 48 teachers
        
        **Budget:** $275,000
        """)
    
    with col2:
        st.markdown("""
        **🏥 Health Example**
        - Maternal health programs
        - Community worker visits
        - 8,000 pregnant women
        
        **Outcomes:** Facility births
        """)
    
    with col3:
        st.markdown("""
        **🌾 Agriculture Example**
        - Climate-smart farming
        - Drip irrigation + SMS
        - 2,500 farmers
        
        **Focus:** Productivity gains
        """)

    st.markdown("---")
    
    st.info(
        "💡 **Pro Tip:** If running locally, all features can be driven from the CLI. Run `rct-field-flow --help` "
        "to explore commands and options."
    )


# ----------------------------------------------------------------------------- #
# RCT DESIGN                                                                    #
# ----------------------------------------------------------------------------- #


def render_rct_design() -> None:
    """
    Render the RCT Design Wizard page.
    This now uses the new wizard.py which provides:
    - 15-section concept note builder
    - Sample concept notes (education, health, agriculture)
    - Multi-format export (Markdown, DOCX, PDF)
    - Real-time preview and validation
    """
    try:
        # Import the wizard dynamically to handle path resolution
        import importlib.util
        from pathlib import Path
        import os
        
        # Try multiple path resolution strategies for cross-platform compatibility
        wizard_path = None
        possible_paths = [
            Path(__file__).parent / "rct-design" / "wizard.py",  # Standard relative path
            Path(__file__).resolve().parent / "rct-design" / "wizard.py",  # Resolved path
            Path(os.path.dirname(os.path.abspath(__file__))) / "rct-design" / "wizard.py",  # Absolute path
        ]
        
        for path in possible_paths:
            if path.exists():
                wizard_path = path
                break
        
        if wizard_path is None:
            # Last resort: try direct import as a package
            try:
                # Try importing as a subpackage
                import sys
                rct_design_path = Path(__file__).parent / "rct-design"
                if str(rct_design_path) not in sys.path:
                    sys.path.insert(0, str(rct_design_path))
                import wizard as wizard_module
                wizard_module.main()
                return
            except ImportError:
                pass
            
            raise FileNotFoundError(f"Could not locate wizard.py. Tried paths: {[str(p) for p in possible_paths]}")
        
        spec = importlib.util.spec_from_file_location("wizard_module", wizard_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load wizard from {wizard_path}")
        
        wizard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wizard_module)
        
        # Run the wizard
        wizard_module.main()
        
    except (ImportError, FileNotFoundError) as e:
        import traceback
        st.error(f"Could not load RCT Design Wizard: {str(e)}")
        st.info("Please ensure the rct-design module is properly installed.")
        with st.expander("📋 Debug Info"):
            import os
            current_dir = Path(__file__).parent
            rct_design_dir = current_dir / "rct-design"
            wizard_file = rct_design_dir / "wizard.py"
            
            debug_info = f"""Error: {str(e)}
__file__: {__file__}
Current directory: {current_dir}
RCT design directory exists: {rct_design_dir.exists()}
Wizard file exists: {wizard_file.exists()}
Attempted paths: {[str(p) for p in possible_paths]}

Directory contents:
{os.listdir(current_dir) if current_dir.exists() else 'Directory not found'}

RCT-design contents:
{os.listdir(rct_design_dir) if rct_design_dir.exists() else 'Directory not found'}
"""
            st.code(debug_info, language="text")
            st.code(traceback.format_exc(), language="python")
    except Exception as e:
        import traceback
        st.error(f"Error running RCT Design Wizard: {str(e)}")
        with st.expander("📋 Technical Details"):
            st.code(traceback.format_exc(), language="python")


def render_design_welcome(
    card,
    formatted,
    team_name,
    app_description: str,
    participant_guidance: List[str],
    sprint_checklist: List[str],
):
    """Render the welcome/home page matching original rct-design architecture."""
    default_description = "This workshop guides you through designing an RCT."
    APP_DESCRIPTION = app_description or default_description
    PARTICIPANT_GUIDANCE = participant_guidance or []
    SPRINT_CHECKLIST = sprint_checklist or []
    
    # About This Activity expander
    with st.expander("📖 About This Activity", expanded=False):
        st.markdown(APP_DESCRIPTION if APP_DESCRIPTION else "This workshop guides you through designing an RCT.")
        st.markdown("---")
        st.markdown("**How to Use This Workbook:**")
        guidance_list = PARTICIPANT_GUIDANCE if PARTICIPANT_GUIDANCE else []
        for i, guidance in enumerate(guidance_list, 1):
            st.markdown(f"**{i}.** {guidance}")
    
    st.markdown("---")
    
    # Main content in two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Ready to Design Your RCT?
        
        This workshop will guide you through key steps to turn your program concept into 
        a rigorous randomized controlled trial (RCT) design. You'll work as a team to:
        
        1. **Frame the Challenge** – Clarify your core problem and success vision
        2. **Map the Theory of Change** – Connect your activities to outcomes 
        3. **Design Measurement** – Choose indicators and instruments
        4. **Plan Randomization** – Select your random assignment approach
        5. **Safeguard Implementation** – Set up monitoring and adaptation mechanisms
        6. **Decide and Commit** – Record your decision trigger and next steps
        
        **Each section takes 3 minutes.** Work through in order, capture decisions, 
        and mark any items [ ] you'll revisit during the gallery walk.
        """)
        
        st.markdown("### ✅ Sprint Checklist")
        checklist = SPRINT_CHECKLIST if SPRINT_CHECKLIST else []
        for item in checklist:
            st.markdown(f"- [ ] {item}")
    
    with col2:
        st.info("""
        ### 📍 Session Snapshot
        
        **Duration:** 30 min
        
        **Format:**
        - 4 min: Welcome
        - 18 min: Design Sprint
        - 5 min: Gallery
        - 3 min: Commit
        
        **Deliverables:**
        - Theory of Change
        - Measurement Plan
        - Randomization Design
        - Decision Trigger
        """)
    
    st.markdown("---")
    
    # Context Snapshot section
    st.subheader("📍 Context Snapshot")
    col1, col2, col3 = st.columns(3)
    
    context_sections = formatted.get('context_sections', [])
    if len(context_sections) >= 3:
        with col1:
            st.markdown("**Problem**")
            st.write(context_sections[0][1])
        with col2:
            st.markdown("**Resources**")
            st.write(context_sections[1][1])
        with col3:
            st.markdown("**Logistics**")
            st.write(context_sections[2][1])
    
    st.divider()
    
    # Program Concept section
    st.subheader("🎯 Program Concept")
    col1, col2, col3 = st.columns(3)
    
    concept_sections = formatted.get('concept_sections', [])
    if len(concept_sections) >= 3:
        with col1:
            st.markdown("**Activities**")
            st.write(concept_sections[0][1])
        with col2:
            st.markdown("**Approach**")
            st.write(concept_sections[1][1])
        with col3:
            st.markdown("**Engagement**")
            st.write(concept_sections[2][1])
    
    st.divider()
    
    # Decision Horizon & Metrics section
    st.subheader("� Decision Horizon & Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Decision Trigger**")
        st.info(formatted.get('decision_horizon', 'N/A'))
    with col2:
        st.metric("Reach", formatted.get('reach', 'N/A'))
    with col3:
        st.metric("Budget", formatted.get('budget', 'N/A'))
    
    st.divider()
    
    # Design Considerations
    st.subheader("⚠️ Design Considerations")
    st.warning(formatted.get('considerations', 'N/A'))
    
    st.divider()
    
    # Baseline Data section
    st.subheader("📊 Baseline Data for Randomization")
    
    data_file_map = {
        "education_bridge_to_basics": "data/sample_data/education_bridge_to_basics.csv",
        "health_community_care_loop": "data/sample_data/health_community_care_loop.csv",
        "agriculture_smart_water_boost": "data/sample_data/agriculture_smart_water_boost.csv"
    }
    
    card_id = st.session_state.design_program_card
    if card_id in data_file_map:
        data_path = Path(__file__).parent / "rct-design" / data_file_map[card_id]
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"""
                **Sample Data Available:**
                - {len(df):,} records
                - {df.shape[1]} variables
                - Ready for randomization practice
                
                Download this baseline data to use with the RCT Field Flow randomization tool.
                """)
            
            with col2:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Baseline Data",
                    data=csv_data,
                    file_name=data_path.name,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with st.expander("👁️ Preview Data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    # Call to action
    st.success("✓ You've reviewed your program card. Ready to start the design sprint?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 View Program Card", use_container_width=True):
            st.session_state.design_current_step = 0
            st.rerun()
    
    with col2:
        if st.button("▶️ Begin Design Sprint", type="primary", use_container_width=True):
            st.session_state.design_current_step = 2  # Start at first workbook step
            st.rerun()


def render_program_card_full(card, formatted, team_name):
    """Render full program card display page."""
    st.markdown(f"## 🎴 {formatted.get('title', 'N/A')}")
    st.markdown(f"**Sector:** {formatted.get('sector', 'N/A')} | **Theme:** {formatted.get('theme', 'N/A')}")
    
    st.divider()
    
    # Context section - render full sections same as welcome page
    st.subheader("📍 Context Snapshot")
    col1, col2, col3 = st.columns(3)
    
    context_sections = formatted.get('context_sections', [])
    if len(context_sections) >= 3:
        with col1:
            st.markdown("**Problem**")
            st.write(context_sections[0][1])
        with col2:
            st.markdown("**Resources**")
            st.write(context_sections[1][1])
        with col3:
            st.markdown("**Logistics**")
            st.write(context_sections[2][1])
    
    st.divider()
    
    # Program concept
    st.subheader("🎯 Program Concept")
    col1, col2, col3 = st.columns(3)
    
    concept_sections = formatted.get('concept_sections', [])
    if len(concept_sections) >= 3:
        with col1:
            st.markdown("**Activities**")
            st.write(concept_sections[0][1])
        with col2:
            st.markdown("**Approach**")
            st.write(concept_sections[1][1])
        with col3:
            st.markdown("**Engagement**")
            st.write(concept_sections[2][1])
    
    st.divider()
    
    # Decision horizon and metrics
    st.subheader("📊 Decision Horizon & Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Decision Trigger**")
        st.info(formatted.get('decision_horizon', 'N/A'))
    with col2:
        st.metric("Reach", formatted.get('reach', 'N/A'))
    with col3:
        st.metric("Budget", formatted.get('budget', 'N/A'))
    
    st.divider()
    
    # Considerations
    st.subheader("⚠️ Design Considerations")
    st.warning(formatted.get('considerations', 'N/A'))
    
    st.divider()
    
    # Baseline Data section
    st.subheader("📊 Baseline Data for Randomization")
    
    data_file_map = {
        "education_bridge_to_basics": "data/sample_data/education_bridge_to_basics.csv",
        "health_community_care_loop": "data/sample_data/health_community_care_loop.csv",
        "agriculture_smart_water_boost": "data/sample_data/agriculture_smart_water_boost.csv"
    }
    
    card_id = st.session_state.design_program_card
    if card_id in data_file_map:
        data_path = Path(__file__).parent / "rct-design" / data_file_map[card_id]
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"""
                **Sample Data Available:**
                - {len(df):,} records
                - {df.shape[1]} variables
                - Ready for randomization practice
                
                Download this baseline data to use with the RCT Field Flow randomization tool.
                """)
            
            with col2:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Baseline Data",
                    data=csv_data,
                    file_name=data_path.name,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with st.expander("👁️ Preview Data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    # Navigation buttons
    st.success("✓ You've reviewed your program card. Ready to start the design sprint?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Begin Design Sprint", type="primary", use_container_width=True):
            st.session_state.design_current_step = 2
            st.rerun()
    
    with col2:
        if st.button("← Back to Welcome", use_container_width=True):
            st.session_state.design_current_step = 1
            st.rerun()


def render_design_workbook(team_name, WORKBOOK_STEPS):
    """Render the design workbook steps interface."""
    # Workbook step is design_current_step - 2 (step 2 = workbook 0)
    current_step = st.session_state.design_current_step - 2
    
    if current_step < len(WORKBOOK_STEPS):
                step = WORKBOOK_STEPS[current_step]
                
                # Step header
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #164a7f 0%, #2fa6dc 100%); 
                            color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; opacity: 0.9;">
                        Step {step['number']} of {len(WORKBOOK_STEPS)}
                    </div>
                    <div style="font-size: 1.75rem; font-weight: 600; margin: 0.5rem 0;">
                        {step['title']}
                    </div>
                    <div style="font-size: 1.05rem; opacity: 0.95;">
                        {step['goal']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress dots
                progress_html = '<div style="display: flex; justify-content: center; gap: 0.5rem; margin: 1.5rem 0;">'
                for i in range(len(WORKBOOK_STEPS)):
                    if i < current_step:
                        progress_html += '<div style="width: 12px; height: 12px; border-radius: 50%; background: #4caf50;"></div>'
                    elif i == current_step:
                        progress_html += '<div style="width: 12px; height: 12px; border-radius: 50%; background: #164a7f; box-shadow: 0 0 8px rgba(22, 74, 127, 0.5);"></div>'
                    else:
                        progress_html += '<div style="width: 12px; height: 12px; border-radius: 50%; background: #ddd;"></div>'
                progress_html += '</div>'
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # Two column layout
                col1, col2 = st.columns([1, 1.2])
                
                with col1:
                    # Actions section
                    st.markdown('<div style="background: rgba(22, 74, 127, 0.06); border-left: 4px solid #2fa6dc; border-radius: 8px; padding: 1.25rem; margin: 1rem 0;">', unsafe_allow_html=True)
                    st.markdown("**Actions:**")
                    for action in step['actions']:
                        st.markdown(f"• {action}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tip
                    st.markdown(f'<div style="background: rgba(47, 166, 220, 0.12); border-left: 4px solid #2fa6dc; border-radius: 8px; padding: 1rem; margin: 1rem 0;"><strong>💡 Tip:</strong> {step["tip"]}</div>', unsafe_allow_html=True)
                
                with col2:
                    # Note fields
                    st.markdown("**Your Responses:**")
                    for field in step['fields']:
                        field_key = f"design_step{step['number']}_{field['key']}"
                        label = field.get('label', '')
                        placeholder = field.get('placeholder', '')
                        
                        if field['type'] == 'text':
                            value = st.text_input(
                                label,
                                value=st.session_state.design_workbook_responses.get(field_key, ''),
                                placeholder=placeholder,
                                key=field_key,
                                label_visibility="visible"
                            )
                        elif field['type'] == 'textarea':
                            rows = field.get('rows', 3)
                            value = st.text_area(
                                label,
                                value=st.session_state.design_workbook_responses.get(field_key, ''),
                                placeholder=placeholder,
                                key=field_key,
                                height=rows * 35,
                                label_visibility="visible"
                            )
                        
                        st.session_state.design_workbook_responses[field_key] = value
                
                # Navigation buttons
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if current_step > 0:
                        if st.button("← Previous Step", use_container_width=True):
                            st.session_state.design_current_step -= 1
                            st.rerun()
                
                with col2:
                    if current_step < len(WORKBOOK_STEPS) - 1:
                        if st.button("Next Step →", type="primary", use_container_width=True):
                            st.session_state.design_current_step += 1
                            st.rerun()
                    else:
                        if st.button("Complete & Generate Report →", type="primary", use_container_width=True):
                            # Save design data to session state
                            st.session_state.design_data = {
                                "team_name": team_name,
                                "program_card": st.session_state.design_program_card,
                                "timestamp": str(datetime.now()),
                                "responses": dict(st.session_state.design_workbook_responses)
                            }
                            # Log workbook completion
                            log_activity('rct_design_workbook_completed', {
                                'team_name': team_name,
                                'program_card': st.session_state.design_program_card
                            })
                            # Navigate to RCT Design report generation (step 8)
                            st.session_state.design_current_step = 8  # Report generation step
                            st.rerun()
    else:
        st.success("✅ All steps completed!")


def render_design_report_generation(team_name):
    """Render the RCT Design sprint report generation page."""
    st.title("📄 Generate Final Report")
    
    # Get design data
    program_card = st.session_state.design_program_card
    workbook_responses = st.session_state.design_workbook_responses
    
    st.markdown(f"""
    ### 🎉 {team_name} - Complete Your RCT Design Activity
    
    **Program Card:** {program_card}
    
    You've worked through the complete RCT design process:
    1. ✅ Selected a program
    2. ✅ Designed your RCT (Step 1-6)
    3. ✅ Documented your design decisions
    
    Now let's generate your final design report!
    """)
    
    st.markdown("---")
    
    # Report summary
    st.markdown("### 📋 Report Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Report Includes:**
        - Program selection and context
        - All 6 design sprint steps
        - Your team's responses
        - Key design decisions
        - Next steps for implementation
        """)
    
    with col2:
        st.markdown("""
        **Export Options:**
        - 📄 HTML (view in browser)
        - 📊 Full design workbook
        - 🎯 Ready for randomization
        """)
    
    st.markdown("---")
    
    # Preview responses
    st.markdown("### 👁️ Preview Your Responses")
    
    with st.expander("View All Workbook Responses", expanded=False):
        if workbook_responses:
            for key, value in workbook_responses.items():
                if value:
                    st.markdown(f"**{key}:** {value}")
        else:
            st.info("No responses captured yet")
    
    st.markdown("---")
    
    # Generate report button
    st.markdown("### 📝 Generate Your Report")
    
    if st.button("📄 Generate HTML Report", use_container_width=True, type="primary"):
        # Generate simple HTML report
        timestamp = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>RCT Design Report - {team_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 20px; line-height: 1.6; }}
                h1 {{ color: #164a7f; border-bottom: 3px solid #2fa6dc; padding-bottom: 10px; }}
                h2 {{ color: #2fa6dc; margin-top: 30px; }}
                .header {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #2fa6dc; }}
                .response {{ margin: 10px 0; }}
                .label {{ font-weight: bold; color: #164a7f; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 RCT Design Activity Report</h1>
                <p><strong>Team:</strong> {team_name}</p>
                <p><strong>Program:</strong> {program_card}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <h2>📋 Design Sprint Responses</h2>
        """
        
        # Add responses
        if workbook_responses:
            for key, value in workbook_responses.items():
                if value:
                    clean_key = key.replace("step", "Step ").replace("_", " ").title()
                    html_content += f"""
                    <div class="response">
                        <span class="label">{clean_key}:</span><br>
                        {value}
                    </div>
                    """
        
        html_content += """
            <hr style="margin: 40px 0;">
            <p style="text-align: center; color: #666;">
                Generated by RCT Field Flow | Developed by Aubrey Jolex<br>
                <a href="mailto:aubreyjolex@gmail.com">aubreyjolex@gmail.com</a>
            </p>
        </body>
        </html>
        """
        
        st.success("✅ Report generated successfully!")
        
        # Download button
        st.download_button(
            label="📥 Download HTML Report",
            data=html_content,
            file_name=f"RCT_Design_Report_{team_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )
        
        # Preview
        with st.expander("👁️ Preview Report", expanded=False):
            st.components.v1.html(html_content, height=600, scrolling=True)
    
    st.markdown("---")
    
    # Next steps
    st.markdown("### 🎲 Next Step: Randomization")
    st.info("With your design sprint complete and report generated, you're ready to randomize your baseline data.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Workbook", use_container_width=True):
            st.session_state.design_current_step = 7  # Last workbook step
            st.rerun()
    
    with col2:
        if st.button("🏠 Back to Welcome", use_container_width=True):
            st.session_state.design_current_step = 1
            st.rerun()
    
    with col3:
        if st.button("▶️ Proceed to Randomization", type="primary", use_container_width=True):
            st.session_state.current_page = "random"
            st.session_state.selected_page = "random"
            st.rerun()


# ----------------------------------------------------------------------------- #
# HOME                                                                          #
# ----------------------------------------------------------------------------- #


def render_randomization() -> None:
    st.title("🎲 Randomization")
    st.markdown(
        "Upload randomization data, configure treatment arms, and run rerandomization with balance checks."
    )
    
    # Display design data if coming from RCT Design
    if st.session_state.design_data:
        with st.info(f"🎯 **Design loaded:** Team {st.session_state.design_data.get('team_name')}", icon="ℹ️"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Program:** {st.session_state.design_data.get('program_card', 'N/A')}")
            with col2:
                st.markdown(f"**Started:** {st.session_state.design_data.get('timestamp', 'N/A')}")

    default_config = load_default_config().get("randomization", {})
    df = st.session_state.baseline_data

    upload = st.file_uploader("Upload randomization data (CSV)", type="csv", key="randomization_upload")
    if upload:
        df = pd.read_csv(upload)
        st.session_state.baseline_data = df
        st.success(f"Loaded {len(df):,} observations • {len(df.columns)} columns.")

    if df is None:
        st.info("Please upload a randomization data in CSV to configure randomization.")
        return

    st.markdown("#### Preview")
    st.dataframe(df.head(10), use_container_width=True)

    available_cols = list(df.columns)

    with st.form("randomization_form"):
        col1, col2 = st.columns(2)
        with col1:
            id_column = st.selectbox("ID Column", available_cols, key="rand_id_col")
            treatment_column = st.text_input(
                "Treatment Column Name",
                value=default_config.get("treatment_column", "treatment"),
                key="rand_treatment_col",
            )
            method_options = ["simple", "stratified", "cluster", "stratified + cluster"]
            method_labels = {
                "simple": "Simple Randomization",
                "stratified": "Stratified Randomization",
                "cluster": "Cluster Randomization",
                "stratified + cluster": "Stratified + Cluster Randomization"
            }
            default_method = default_config.get("method", "simple")
            method_index = method_options.index(default_method) if default_method in method_options else 0
            method = st.selectbox(
                "Method",
                method_options,
                format_func=lambda x: method_labels[x],
                index=method_index,
                key="rand_method",
                help="Choose randomization method. Stratified+Cluster randomizes clusters within strata."
            )
            
            with st.expander("ℹ️ Method explanations"):
                st.markdown("""
                - **Simple**: Each individual randomly assigned to treatment/control
                - **Stratified**: Randomize separately within each stratum (e.g., by gender, region)
                - **Cluster**: Randomize entire groups (e.g., villages, schools) - all members get same treatment
                - **Stratified + Cluster**: Randomize clusters within strata (e.g., randomize villages within districts)
                """)
            iterations = st.number_input(
                "Iterations",
                min_value=1,
                max_value=20000,
                value=int(default_config.get("iterations", 1)),
                step=1,
                key="rand_iterations",
            )
            seed_value = default_config.get("seed", 12345)
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                value=int(seed_value) if seed_value else 12345,
                step=1,
                key="rand_seed",
            )
        with col2:
            use_existing = st.checkbox(
                "Use existing assignment column if present",
                value=default_config.get("use_existing_assignment", True),
                key="rand_use_existing",
            )
            arms_count = st.number_input(
                "Number of treatment arms",
                min_value=2,
                max_value=6,
                value=len(default_config.get("arms", [])) or 2,
                step=1,
                key="rand_arms_count",
            )
            balance_covariates = st.multiselect(
                "Balance covariates",
                available_cols,
                key="rand_balance_covariates",
            )
            strata: List[str] = []
            cluster_col: str | None = None
            if method in ["stratified", "stratified + cluster"]:
                strata = st.multiselect(
                    "Strata columns (stratify randomization within groups)",
                    available_cols,
                    key="rand_strata",
                    help="Randomization will be done separately within each stratum to ensure balance."
                )
            if method in ["cluster", "stratified + cluster"]:
                cluster_col = st.selectbox(
                    "Cluster column (randomize entire groups)",
                    available_cols,
                    key="rand_cluster",
                    help="All units within the same cluster will receive the same treatment."
                )

        st.markdown("#### Treatment arms")
        arms_defaults = default_config.get("arms", [])
        current_count = int(arms_count)
        ensure_arm_state(arms_defaults, current_count)

        arms: List[TreatmentArm] = []
        for idx in range(current_count):
            name_key = f"arm_name_{idx}"
            prop_key = f"arm_prop_{idx}"

            arm_col1, arm_col2 = st.columns([2, 1])
            with arm_col1:
                name_value = st.text_input(f"Arm {idx + 1} name", key=name_key)
                clean_name = name_value.strip() or f"arm_{idx+1}"
            with arm_col2:
                proportion = st.number_input(
                    f"{clean_name} proportion",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.001,
                    key=prop_key,
                )
            arms.append(TreatmentArm(name=clean_name, proportion=float(proportion)))

        submitted = st.form_submit_button("Run randomization", type="primary")

    result: RandomizationResult | None = st.session_state.get("randomization_result")

    if submitted:
        is_valid = True
        total_prop = sum(a.proportion for a in arms)
        if abs(total_prop - 1.0) > 0.01:
            st.error(f"Arm proportions must sum to 1.0 (current total: {total_prop:.2f}).")
            is_valid = False

        if int(seed) <= 0:
            st.error("Random seed is required and must be a positive integer.")
            is_valid = False

        if is_valid:
            # Map "stratified + cluster" to "cluster" method with strata
            actual_method = "cluster" if method == "stratified + cluster" else method

            rand_config = RandomizationConfig(
                id_column=id_column,
                treatment_column=treatment_column or "treatment",
                method=actual_method,  # type: ignore[arg-type]
                arms=arms,
                strata=strata if strata else [],
                cluster=cluster_col,
                balance_covariates=balance_covariates if balance_covariates else [],
                iterations=int(iterations),
                seed=int(seed),
                use_existing_assignment=use_existing,
            )

            try:
                result = Randomizer(rand_config).run(df, verbose=False)
            except Exception as exc:
                st.error(f"Randomization failed: {exc}")
                is_valid = False
            else:
                st.session_state.randomization_result = result
                st.session_state.case_data = result.assignments.copy()
                st.session_state.randomization_config = rand_config  # Store config for validation

                # Store download data in session state to prevent recomputation on reruns
                st.session_state.csv_assignments = result.assignments.to_csv(index=False)
                st.session_state.python_rand_code = generate_python_randomization_code(rand_config, method)
                st.session_state.stata_rand_code = generate_stata_randomization_code(rand_config, method)

                st.success(
                    f"Randomization complete! Iterations: {result.iterations}. Best min p-value: {result.best_min_pvalue:.4f}"
                )

                # Keep track of method for downstream display after reruns
                st.session_state.randomization_method = actual_method

        # Refresh result reference after potential update
        result = st.session_state.get("randomization_result")

    if result is None:
        return

    actual_method = st.session_state.get("randomization_method", method)

    # Treatment distribution table
    st.markdown("#### Treatment Distribution")
    treatment_col = result.assignments.columns[result.assignments.columns.str.contains('treatment|assignment', case=False)][0] if any(result.assignments.columns.str.contains('treatment|assignment', case=False)) else treatment_column
    
    # Always show overall distribution first
    st.markdown("**Overall:**")
    counts = result.assignments[treatment_col].value_counts()
    total = len(result.assignments)
    pct = (counts / total * 100).round(2)
    
    dist_table = pd.DataFrame({
        'Treatment Arm': counts.index,
        'Count': counts.values,
        'Percentage': pct.values.astype(str) + '%'
    })
    st.dataframe(dist_table, use_container_width=True, hide_index=True)
    
    # Additionally show distribution by strata if applicable
    if strata and actual_method in ["stratified", "cluster"]:
        st.markdown("**By Strata:**")
        crosstab = pd.crosstab(
            [result.assignments[col] for col in strata],
            result.assignments[treatment_col],
            margins=True,
            margins_name="Total"
        )
        # Calculate percentages
        pct_tab = crosstab.div(crosstab["Total"], axis=0).multiply(100).round(2)
        pct_tab = pct_tab.drop(columns=["Total"])
        
        # Format as "count (pct%)"
        display_tab = crosstab.copy()
        for col in display_tab.columns:
            if col != "Total":
                display_tab[col] = display_tab[col].astype(str) + " (" + pct_tab[col].astype(str) + "%)"
        
        st.dataframe(display_tab, use_container_width=True)

    st.markdown("#### Assignments preview")
    st.dataframe(result.assignments.head(10), use_container_width=True)

    if not result.balance_table.empty:
        st.markdown("#### Balance table")
        balance_table = result.balance_table.copy()
        if "means" in balance_table.columns:
            means_wide = balance_table["means"].apply(lambda d: pd.Series(d)).rename(
                columns=lambda c: f"mean_{c}"
            )
            balance_table = pd.concat(
                [balance_table.drop(columns=["means"]), means_wide], axis=1
            )
        fmt: Dict[str, str] = {"p_value": "{:.4f}", "min_p_value": "{:.4f}"}
        fmt.update({col: "{:.2f}" for col in balance_table.columns if col.startswith("mean_")})
        style = balance_table.style.format(fmt)
        try:
            import matplotlib  # type: ignore  # noqa: F401

            style = style.background_gradient(subset=["p_value"], cmap="RdYlGn", vmin=0, vmax=1)
        except Exception:
            pass
        st.dataframe(style, use_container_width=True)

    # Downloads section - show whenever results exist in session state
    if ('csv_assignments' in st.session_state and 
        'python_rand_code' in st.session_state and 
        'stata_rand_code' in st.session_state):
        
        st.markdown("---")
        st.markdown("#### 📥 Download Results & Code")
        st.markdown("Download assignments, code, or analysis files.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📊 Download Assignments CSV",
                data=st.session_state.csv_assignments,
                file_name="randomized_assignments.csv",
                mime="text/csv",
                key="download_csv_assignments",
            )
        
        with col2:
            st.download_button(
                "🐍 Download Python Code",
                data=st.session_state.python_rand_code,
                file_name="randomization_code.py",
                mime="text/x-python",
                key="download_python_code",
                help="Python script with your exact randomization parameters"
            )
        
        with col3:
            st.download_button(
                "📈 Download Stata Code",
                data=st.session_state.stata_rand_code,
                file_name="randomization_code.do",
                mime="text/x-stata",
                key="download_stata_code",
                help="Stata do-file with your exact randomization parameters"
            )
    
    # Validation section
    st.markdown("---")
    st.markdown("#### 🔍 Validate Randomization Fairness")
    st.markdown("""
    Run your randomization multiple times with different seeds to verify that it's fair. 
    Each observation should have approximately equal probability of being assigned to each treatment arm.
    """)
    
    with st.expander("ℹ️ What is randomization validation?", expanded=False):
        st.markdown("""
        **Randomization validation** helps ensure your randomization code is working correctly by:
        
        - Running the randomization many times (e.g., 500 times) with different random seeds
        - Tracking how often each observation gets assigned to each treatment arm
        - Checking if probabilities match the expected proportions
        
        **When to use it:**
        - Before running your actual randomization in the field
        - After making changes to randomization code
        - When using complex stratification or clustering
        
        **What to look for:**
        - Assignment probabilities should match expected proportions (e.g., 50% for treatment in a 50/50 split)
        - Standard deviation should be reasonable (not too high)
        - No observations should be systematically favored or excluded
        
        See [RANDOMIZATION.md](https://github.com/ajolex/rct_field_flow/blob/master/docs/RANDOMIZATION.md) for more details.
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        n_simulations = st.number_input(
            "Number of simulations",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            key="validation_n_sims",
            help="More simulations = more accurate validation but takes longer"
        )
    with col2:
        show_plot = st.checkbox(
            "Show histogram",
            value=True,
            key="validation_show_plot",
            help="Display histogram of assignment probabilities"
        )
    
    if st.button("🚀 Run Validation", key="run_validation_btn", type="primary"):
        if df is None:
            st.error("No data loaded for validation")
            return

        # Get current randomization config from form or session state
        rand_config_to_validate = st.session_state.get("randomization_config")
        if rand_config_to_validate is None:
            st.warning("Please run randomization first to set up the configuration")
            return

        with st.spinner(f"Running {n_simulations} simulations..."):
            try:
                randomizer = Randomizer(rand_config_to_validate)
                validation_result = randomizer.validate_randomization(
                    df,
                    n_simulations=int(n_simulations),
                    base_seed=rand_config_to_validate.seed,
                    verbose=False,
                )

                summary_data = []
                for arm, stats in validation_result.summary_stats.items():
                    summary_data.append(
                        {
                            "Treatment Arm": arm,
                            "Expected": f"{stats['expected']:.1%}",
                            "Mean Probability": f"{stats['mean']:.4f}",
                            "Std Deviation": f"{stats['std']:.4f}",
                            "Min": f"{stats['min']:.4f}",
                            "Max": f"{stats['max']:.4f}",
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                histogram_bytes: bytes | None = None
                histogram_caption: str | None = None

                if show_plot:
                    if validation_result.assignment_probabilities.empty:
                        st.info("Assignment probabilities table is empty; nothing to plot.")
                    else:
                        try:
                            import matplotlib.pyplot as plt

                            first_arm = list(validation_result.summary_stats.keys())[0]
                            prob_col = f"prob_{first_arm}"

                            if prob_col in validation_result.assignment_probabilities.columns:
                                fig, ax = plt.subplots(figsize=(10, 5))

                                probs = validation_result.assignment_probabilities[prob_col]
                                expected_prob = validation_result.summary_stats[first_arm]["expected"]

                                ax.hist(probs, bins=30, edgecolor="black", alpha=0.7)
                                ax.axvline(
                                    expected_prob,
                                    color="red",
                                    linestyle="--",
                                    linewidth=2,
                                    label=f"Expected: {expected_prob:.3f}",
                                )
                                ax.set_xlabel(f"Probability of assignment to {first_arm}")
                                ax.set_ylabel("Frequency (number of observations)")
                                ax.set_title(
                                    f"Distribution of Assignment Probabilities - {first_arm}"
                                )
                                ax.legend()
                                ax.grid(True, alpha=0.3)

                                buffer = BytesIO()
                                fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                                buffer.seek(0)
                                histogram_bytes = buffer.getvalue()
                                buffer.close()
                                histogram_caption = (
                                    f"{first_arm}: expected probability {expected_prob:.3f}"
                                )
                                plt.close(fig)
                            else:
                                st.info(
                                    f"Could not locate probability column '{prob_col}' to build the histogram."
                                )
                        except Exception as e:
                            st.info(f"Could not generate plot: {e}")

                csv_validation = validation_result.assignment_probabilities.to_csv(index=False)
                st.session_state.validation_state = {
                    "summary_df": summary_df,
                    "is_valid": validation_result.is_valid,
                    "warnings": validation_result.warnings,
                    "csv_data": csv_validation,
                    "histogram_bytes": histogram_bytes,
                    "histogram_caption": histogram_caption,
                    "n_simulations": int(n_simulations),
                    "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "has_probabilities": not validation_result.assignment_probabilities.empty,
                    "histogram_generated": histogram_bytes is not None,
                    "show_plot_requested": bool(show_plot),
                    "python_code": generate_python_validation_code(
                        rand_config_to_validate,
                        int(n_simulations),
                    ),
                    "stata_code": generate_stata_validation_code(
                        rand_config_to_validate,
                        int(n_simulations),
                    ),
                }

            except Exception as e:
                st.error(f"Validation failed: {str(e)}")
                import traceback
                with st.expander("🐛 Error details"):
                    st.code(traceback.format_exc())

    validation_state = st.session_state.get("validation_state")
    if validation_state:
        st.markdown("---")
        st.markdown("### 📊 Validation Results")
        st.caption(
            f"Last run: {validation_state['ran_at']} • Simulations: {validation_state['n_simulations']:,}"
        )

        if validation_state.get("is_valid"):
            st.success("✓ Validation PASSED - Randomization appears fair!")
        else:
            st.error("✗ Validation FAILED - Potential issues detected")

        st.markdown("#### Assignment Probability Summary")
        st.dataframe(
            validation_state["summary_df"],
            use_container_width=True,
            hide_index=True,
        )

        warnings_list = validation_state.get("warnings") or []
        if warnings_list:
            st.warning("**Warnings:**")
            for warning in warnings_list:
                st.markdown(f"- ⚠️ {warning}")

        histogram_bytes = validation_state.get("histogram_bytes")
        histogram_caption = validation_state.get("histogram_caption")
        histogram_requested = validation_state.get("show_plot_requested", False)

        if show_plot:
            if histogram_bytes:
                st.markdown("#### Assignment Probability Histogram")
                st.image(
                    histogram_bytes,
                    caption=histogram_caption,
                    use_column_width=True,
                )
            else:
                message = (
                    "Histogram not available for this run. Re-run validation with 'Show histogram' enabled."
                    if not histogram_requested
                    else "Histogram could not be generated for this run. Ensure matplotlib is installed and rerun validation."
                )
                st.info(message)

        st.markdown("#### 📎 Validation Downloads")
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            st.download_button(
                "📥 Detailed Validation Results",
                data=validation_state.get("csv_data", ""),
                file_name="validation_results.csv",
                mime="text/csv",
                key="download_validation_csv",
            )
        with dl_col2:
            st.download_button(
                "🖼️ Download Histogram",
                data=histogram_bytes or b"",
                file_name="validation_histogram.png",
                mime="image/png",
                key="download_validation_histogram",
                disabled=histogram_bytes is None,
                help=(
                    "Run validation with 'Show histogram' enabled to generate this file."
                    if histogram_bytes is None
                    else "Save the histogram shown above as a PNG."
                ),
            )
        with dl_col3:
            code_payload = validation_state.get("python_code") or ""
            st.download_button(
                "🐍 Download Validation Code",
                data=code_payload,
                file_name="validation_code.py",
                mime="text/x-python",
                key="download_validation_code",
                disabled=not code_payload,
                help="Python script to reproduce this validation locally.",
            )
        with dl_col4:
            stata_payload = validation_state.get("stata_code") or ""
            st.download_button(
                "📊 Download Stata Validation Do-file",
                data=stata_payload,
                file_name="validation_code.do",
                mime="text/x-stata",
                key="download_validation_code_stata",
                disabled=not stata_payload,
                help="Stata do-file (uses Python integration) to reproduce the validation.",
            )


def render_power_analysis_results(ctx: Dict[str, Any]) -> None:
    """Render cached analytical power results (used to keep downloads after reruns)."""
    calculation_mode = ctx["calculation_mode"]
    design_type = ctx["design_type"]
    outcome_type = ctx["outcome_type"]
    baseline_mean = ctx["baseline_mean"]
    baseline_sd = ctx["baseline_sd"]
    alpha = ctx["alpha"]
    power = ctx["power"]
    treatment_share = ctx["treatment_share"]
    sample_size = ctx.get("sample_size")
    num_clusters = ctx.get("num_clusters")
    cluster_size = ctx.get("cluster_size")
    target_mde = ctx.get("target_mde")
    mde = ctx.get("mde")
    mde_result = ctx.get("mde_result")
    sample_result = ctx.get("sample_result")
    required_n = ctx.get("required_n")
    required_clusters = ctx.get("required_clusters")
    total_n = ctx.get("total_n")
    final_r_squared = ctx.get("final_r_squared", 0.0)
    final_compliance = ctx.get("final_compliance", 1.0)
    icc = ctx.get("icc", 0.0)
    clusters_for_curve = ctx.get("clusters_for_curve")
    table_num_clusters = ctx.get("table_num_clusters")
    code_results = ctx.get("code_results")
    assumption_params = ctx.get("assumption_params") or {}
    curve_params = ctx.get("curve_params") or {}

    if not assumption_params or not curve_params or code_results is None:
        return

    assumptions_obj = PowerAssumptions(**assumption_params)
    curve_assumptions = PowerAssumptions(**curve_params)

    st.markdown("---")
    st.markdown("### 📊 Results")

    if outcome_type == "binary":
        st.info(
            "📌 **Binary Outcome**: MDE represents change in proportion "
            "(e.g., from 50% to 55% = MDE of 0.05 or 5 percentage points)"
        )

    if calculation_mode == "mde" and mde_result:
        attrition_rate = ctx.get("attrition_rate", 0.0)
        
        # Display results without attrition
        st.markdown("#### Without Attrition")
        col1, col2, col3 = st.columns(3)
        with col1:
            if outcome_type == "binary":
                st.metric("MDE (Percentage Points)", f"{mde*100:.1f}pp")
                st.caption(f"Change from {baseline_mean*100:.1f}% to {(baseline_mean+mde)*100:.1f}%")
            else:
                st.metric("Minimum Detectable Effect", f"{mde:.3f}")
                st.caption(f"{(mde / baseline_mean * 100):.2f}% of baseline mean")

        with col2:
            if sample_size:
                st.metric("Sample Size", f"{int(sample_size):,}")
            if design_type == "cluster" and num_clusters and cluster_size:
                st.caption(f"{int(num_clusters)} clusters × {int(cluster_size)} individuals")

        with col3:
            st.metric("Power", f"{power*100:.0f}%")
            st.caption(f"at α = {alpha}")
        
        # Display results with attrition if applicable
        if attrition_rate > 0 and mde_result:
            st.markdown(f"#### With {attrition_rate*100:.0f}% Attrition")
            col1, col2, col3 = st.columns(3)
            
            mde_with_attrition = mde_result.get("mde_absolute_with_attrition", mde)
            sample_with_attrition = mde_result.get("sample_with_attrition") or mde_result.get("total_with_attrition")
            
            with col1:
                if outcome_type == "binary":
                    st.metric("MDE (Percentage Points)", f"{mde_with_attrition*100:.1f}pp", 
                             delta=f"{(mde_with_attrition - mde)*100:.1f}pp", delta_color="inverse")
                else:
                    st.metric("Minimum Detectable Effect", f"{mde_with_attrition:.3f}",
                             delta=f"{mde_with_attrition - mde:.3f}", delta_color="inverse")
                st.caption(f"{(mde_with_attrition / baseline_mean * 100):.2f}% of baseline mean")

            with col2:
                if sample_with_attrition:
                    st.metric("Required Sample Size", f"{int(sample_with_attrition):,}",
                             delta=f"+{int(sample_with_attrition - sample_size):,}")
                if design_type == "cluster" and mde_result.get("cluster_size_with_attrition") and num_clusters:
                    m_with_attrition = mde_result["cluster_size_with_attrition"]
                    st.caption(f"{int(num_clusters)} clusters × {int(m_with_attrition)} individuals (after attrition)")

            with col3:
                st.metric("Power", f"{power*100:.0f}%")
                st.caption("(maintained with larger N)")
        
        with st.container():
            sample_text = (
                f"in {int(num_clusters)} clusters" if design_type == "cluster" and num_clusters else ""
            )
            if attrition_rate > 0:
                st.success(
                    f"✅ **Without attrition**: With **{int(sample_size):,} individuals** ({sample_text}), "
                    f"you can detect an effect of **{mde:.3f}** with **{power*100:.0f}% power**.\n\n"
                    f"⚠️ **With {attrition_rate*100:.0f}% attrition**: Recruit **{int(sample_with_attrition):,} individuals** "
                    f"to maintain power and detect **{mde_with_attrition:.3f}** effect."
                )
            else:
                st.info(
                    f"With **{int(sample_size):,} individuals** ({sample_text}), "
                    f"you can detect an effect of **{mde:.3f}** ({(mde / baseline_mean * 100):.2f}% of baseline) "
                    f"with **{power*100:.0f}% power** at **α = {alpha}**."
                )
    elif calculation_mode == "sample_size" and sample_result:
        attrition_rate = ctx.get("attrition_rate", 0.0)
        
        if design_type == "individual":
            # Display results without attrition
            st.markdown("#### Without Attrition")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Required Sample Size", f"{int(required_n):,}")
            with col2:
                st.metric("Target MDE", f"{target_mde:.3f}")
                st.caption(f"{(target_mde / baseline_mean * 100):.2f}% of baseline mean")
            with col3:
                st.metric("Power", f"{power*100:.0f}%")
                st.caption(f"at α = {alpha}")
            
            # Display results with attrition if applicable
            if attrition_rate > 0 and sample_result:
                st.markdown(f"#### With {attrition_rate*100:.0f}% Attrition")
                n_with_attrition = sample_result.get("sample_size_with_attrition", required_n)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Required Sample Size", f"{int(n_with_attrition):,}",
                             delta=f"+{int(n_with_attrition - required_n):,}")
                with col2:
                    st.metric("Target MDE", f"{target_mde:.3f}")
                    st.caption("(maintained with larger N)")
                with col3:
                    st.metric("Power", f"{power*100:.0f}%")
                    st.caption("(preserved)")
            
            with st.container():
                if attrition_rate > 0:
                    n_with_attrition = sample_result.get("sample_size_with_attrition", required_n)
                    st.success(
                        f"✅ **Without attrition**: **{int(required_n):,} individuals** needed.\n\n"
                        f"⚠️ **With {attrition_rate*100:.0f}% attrition**: Recruit **{int(n_with_attrition):,} individuals** "
                        f"to maintain **{power*100:.0f}% power** for detecting **{target_mde:.3f}** effect."
                    )
                else:
                    st.info(
                        f"You need **{int(required_n):,} individuals** to detect an effect of "
                        f"**{target_mde:.3f}** ({(target_mde / baseline_mean * 100):.2f}% of baseline) "
                        f"with **{power*100:.0f}% power** at **α = {alpha}**."
                    )
        else:
            # Cluster design
            st.markdown("#### Without Attrition")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Required Clusters", f"{required_clusters:,}")
                st.caption(f"Total N = {total_n:,}")
            with col2:
                st.metric("Target MDE", f"{target_mde:.3f}")
                st.caption(f"{(target_mde / baseline_mean * 100):.2f}% of baseline mean")
            with col3:
                if cluster_size:
                    st.metric("Cluster Size", f"{int(cluster_size):,}")
                st.caption(f"ICC = {icc:.3f}")
            
            # Display results with attrition if applicable
            if attrition_rate > 0 and sample_result:
                st.markdown(f"#### With {attrition_rate*100:.0f}% Attrition")
                cluster_size_with_attrition = sample_result.get("cluster_size_with_attrition", cluster_size)
                total_with_attrition = sample_result.get("total_with_attrition", total_n)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Required Clusters", f"{required_clusters:,}")
                    st.caption(f"Total N = {total_with_attrition:,}")
                with col2:
                    st.metric("Cluster Size (Inflated)", f"{int(cluster_size_with_attrition):,}",
                             delta=f"+{int(cluster_size_with_attrition - cluster_size):,}")
                    st.caption("(accounts for attrition)")
                with col3:
                    st.metric("Power", f"{power*100:.0f}%")
                    st.caption("(preserved)")
            
            with st.container():
                if attrition_rate > 0:
                    cluster_size_with_attrition = sample_result.get("cluster_size_with_attrition", cluster_size)
                    total_with_attrition = sample_result.get("total_with_attrition", total_n)
                    st.success(
                        f"✅ **Without attrition**: **{required_clusters:,} clusters** × **{int(cluster_size)} individuals** = {total_n:,} total.\n\n"
                        f"⚠️ **With {attrition_rate*100:.0f}% attrition**: Recruit **{required_clusters:,} clusters** × **{int(cluster_size_with_attrition)} individuals** "
                        f"= {total_with_attrition:,} total to maintain **{power*100:.0f}% power**."
                    )
                else:
                    st.info(
                        f"You need **{required_clusters:,} clusters** ({total_n:,} individuals) "
                        f"to detect an effect of **{target_mde:.3f}** "
                        f"({(target_mde / baseline_mean * 100):.2f}% of baseline) "
                        f"with **{power*100:.0f}% power** at **α = {alpha}**."
                    )

    st.markdown("---")
    st.markdown("### 📈 Power Curves")

    tab1, tab2 = st.tabs(["Power vs Sample Size", "Cluster Analysis" if design_type == "cluster" else "Power Analysis"])

    with tab1:
        if design_type == "cluster" and not clusters_for_curve:
            st.warning("Cluster count unavailable. Recalculate to generate power curves.")
        else:
            power_df = generate_power_curve(curve_assumptions)
            import plotly.graph_objects as go

            fig = go.Figure()

            if design_type == "cluster":
                x_data = power_df['clusters']
                x_label = "Number of Clusters"
            else:
                x_data = power_df['sample_size']
                x_label = "Sample Size (N)"

            fig.add_trace(go.Scatter(
                x=x_data,
                y=power_df['power'],
                mode='lines',
                name='Power',
                line=dict(color='#1f77b4', width=3)
            ))

            fig.add_hline(y=0.8, line_dash="dash", line_color="gray",
                          annotation_text="80% Power", annotation_position="right")

            if calculation_mode == "mde":
                x_val = num_clusters if design_type == "cluster" else sample_size
                if x_val:
                    fig.add_vline(x=x_val, line_dash="dash", line_color="red",
                                  annotation_text=f"Current: {int(x_val):,}")
            else:
                x_val = required_clusters if design_type == "cluster" else required_n
                if x_val:
                    fig.add_vline(x=x_val, line_dash="dash", line_color="red",
                                  annotation_text=f"Required: {int(x_val):,}")

            fig.update_layout(
                title="Statistical Power vs Sample Size",
                xaxis_title=x_label,
                yaxis_title="Power (1-β)",
                yaxis=dict(range=[0, 1]),
                hovermode='x unified',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if design_type == "cluster" and cluster_size:
            attrition_rate = ctx.get("attrition_rate", 0.0)
            table_alpha = alpha
            table_power = power
            table_mean = baseline_mean
            table_sd = baseline_sd
            table_clusters = table_num_clusters or clusters_for_curve or 50

            table_assumptions = PowerAssumptions(
                alpha=table_alpha,
                power=table_power,
                baseline_mean=table_mean,
                baseline_sd=table_sd,
                outcome_type=outcome_type,
                treatment_share=treatment_share,
                design_type="cluster",
                num_clusters=table_clusters,
                cluster_size=int(cluster_size),
                icc=icc,
                r_squared=final_r_squared,
                compliance_rate=final_compliance,
                attrition_rate=attrition_rate
            )

            # Use updated cluster_sizes with smaller increments (rows) and num_clusters as columns
            mde_table_df = generate_cluster_size_table(
                table_assumptions,
                cluster_sizes=None,  # Will use default: [16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 35, 38, 40]
                num_clusters_options=None  # Will use default: [20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
            )

            mde_table = mde_table_df.pivot(
                index='cluster_size',
                columns='num_clusters',
                values='mde_absolute'
            )

            if attrition_rate > 0:
                st.info(f"📊 **MDE Table**: Values shown are **after {attrition_rate*100:.0f}% attrition adjustment** (final sample sizes)")
            else:
                st.info("📊 **MDE Table**: Values shown are for the specified sample sizes (no attrition)")
            
            st.dataframe(
                mde_table.style.format("{:.3f}").background_gradient(cmap='RdYlGn_r', axis=None),
                use_container_width=True
            )
            st.caption(
                "**Table Structure**: Rows = Cluster Size (m) with 1-2 unit increments, "
                "Columns = Number of Clusters (k). "
                "Each cell shows the MDE for that configuration. "
                "Smaller MDEs (greener) indicate higher power."
            )
        elif design_type == "cluster":
            st.warning("Cluster size missing. Recalculate to view cluster analysis.")
        else:
            st.info("Cluster size analysis is only available for cluster randomization designs.")

    st.markdown("---")
    st.markdown("### 💻 Downloadable Code")

    python_code = generate_python_code(
        curve_assumptions if calculation_mode == "mde" else assumptions_obj,
        code_results
    )
    stata_code = generate_stata_code(
        curve_assumptions if calculation_mode == "mde" else assumptions_obj,
        code_results
    )

    code_tab1, code_tab2 = st.tabs(["Python", "Stata"])

    with code_tab1:
        st.code(python_code, language="python")
        st.download_button(
            "Download Python Script",
            data=python_code,
            file_name="power_calculation.py",
            mime="text/x-python"
        )

    with code_tab2:
        st.code(stata_code, language="stata")
        st.download_button(
            "Download Stata Script",
            data=stata_code,
            file_name="power_calculation.do",
            mime="text/plain"
        )

    st.markdown("#### 📥 Quick Downloads")
    col_python, col_stata = st.columns(2)
    with col_python:
        st.download_button(
            "🐍 Download Python Code",
            data=python_code,
            file_name="power_calculation.py",
            mime="text/x-python",
            use_container_width=True
        )
    with col_stata:
        st.download_button(
            "📈 Download Stata Code",
            data=stata_code,
            file_name="power_calculation.do",
            mime="text/plain",
            use_container_width=True
        )


# ----------------------------------------------------------------------------- #
# POWER CALCULATIONS                                                            #
# ----------------------------------------------------------------------------- #


def render_power_calculations() -> None:
    st.title("⚡ Power Calculations")
    st.markdown(
        "Calculate statistical power, minimum detectable effects (MDE), and required sample sizes "
        "for your randomized controlled trial."
    )
    
    # Educational content
    with st.expander("📚 What is Statistical Power?", expanded=False):
        st.markdown("""
        **Statistical power** is the probability that your study will detect a treatment effect 
        if there truly is one. It's calculated as 1 - β, where β is the Type II error rate 
        (the probability of failing to detect a real effect).
        
        **Key concepts:**
        - **Power = 0.80** means you have an 80% chance of detecting a true effect
        - Higher power reduces the risk of false negatives (Type II errors)
        - Standard practice is to aim for 80% power (0.80)
        - Power increases with larger sample sizes and larger effect sizes
        
        **Reference:** [J-PAL Power Calculations Guide](https://www.povertyactionlab.org/resource/power-calculations)
        """)
    
    with st.expander("🎯 Key Determinants of Power", expanded=False):
        st.markdown("""
        Your study's statistical power depends on several factors:
        
        1. **Sample Size (N)**: Larger samples → Higher power
        2. **Effect Size (MDE)**: Larger effects → Easier to detect → Higher power
        3. **Significance Level (α)**: Lower α → Harder to reject null → Lower power
        4. **Outcome Variance (σ²)**: Lower variance → Higher power
        5. **Treatment Share**: Balanced allocation (50/50) → Maximum power
        6. **Covariates (R²)**: Better prediction → Lower residual variance → Higher power
        7. **Compliance**: Perfect compliance → Higher power
        8. **Clustering (ICC)**: Higher ICC → Lower power (requires more clusters/larger samples)
        """)
    
    with st.expander("📋 Required Assumptions", expanded=False):
        st.markdown("""
        To calculate power or sample size, you need to specify:
        
        **Basic Parameters:**
        - **Significance level (α)**: Usually 0.05 (5% false positive rate)
        - **Power (1-β)**: Usually 0.80 (80% chance of detecting real effects)
        - **Baseline outcome mean & SD**: From pilot data or similar studies
        - **Treatment share**: Proportion assigned to treatment (usually 0.5)
        
        **Effect Size:**
        - **MDE**: The minimum effect you want to detect (in outcome units)
        - Or specify **target sample size** and calculate what MDE is detectable
        
        **Design Features (if applicable):**
        - **Cluster randomization**: Number of clusters, cluster size, ICC
        - **Covariates**: R² from regressing outcome on baseline covariates
        - **Imperfect compliance**: Expected compliance rate for ITT vs LATE estimates
        """)
    
    # Design type selector
    st.markdown("---")
    st.markdown("### Design Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        outcome_type = st.radio(
            "Outcome Type",
            options=["continuous", "binary"],
            format_func=lambda x: "Continuous (e.g., test scores)" if x == "continuous" else "Binary (e.g., enrollment)",
            help="Binary outcomes use different formulas (power twoproportions in Stata). Note: Covariate adjustment not applicable for binary outcomes."
        )
    
    with col2:
        design_type = st.radio(
            "Randomization Design",
            options=["individual", "cluster"],
            format_func=lambda x: "Individual Randomization" if x == "individual" else "Cluster Randomization",
            help="Choose whether you're randomizing individuals or clusters (e.g., villages, schools)"
        )
    
    with col3:
        calculation_mode = st.radio(
            "What to Calculate?",
            options=["mde", "sample_size"],
            format_func=lambda x: "Minimum Detectable Effect (MDE)" if x == "mde" else "Required Sample Size",
            help="Calculate MDE for a given sample size, or calculate required sample size for a target MDE"
        )
    
    with col4:
        calculation_method = st.radio(
            "Calculation Method",
            options=["analytical", "simulation"],
            format_func=lambda x: "Analytical (Formula-based)" if x == "analytical" else "Simulation (Monte Carlo)",
            help="Analytical uses closed-form formulas (fast). Simulation uses Monte Carlo methods (flexible, validates assumptions)."
        )
    
    # Input form
    st.markdown("---")
    st.markdown("### Parameters")
    
    with st.form("power_calc_form"):
        # Basic parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha = st.number_input(
                "Significance Level (α)",
                min_value=0.001,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Probability of Type I error (false positive). Standard: 0.05"
            )
            
            if outcome_type == "binary":
                baseline_mean = st.number_input(
                    "Baseline Proportion (Control)",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.5,
                    step=0.01,
                    help="Proportion with outcome=1 in control group (e.g., 0.5 = 50% enrollment rate)"
                )
            else:
                baseline_mean = st.number_input(
                    "Baseline Outcome Mean",
                    value=100.0,
                    help="Average outcome in control group (in outcome units)"
                )
        
        with col2:
            power = st.number_input(
                "Statistical Power (1-β)",
                min_value=0.5,
                max_value=0.99,
                value=0.80,
                step=0.05,
                help="Probability of detecting a true effect. Standard: 0.80"
            )
            
            if outcome_type == "binary":
                # SD calculated automatically for binary
                baseline_sd = math.sqrt(baseline_mean * (1 - baseline_mean))
                st.info(f"SD (auto-calculated): {baseline_sd:.4f}")
            else:
                baseline_sd = st.number_input(
                    "Baseline Outcome SD",
                    min_value=0.01,
                    value=15.0,
                    help="Standard deviation of outcome in control group"
                )
        
        with col3:
            treatment_share = st.number_input(
                "Treatment Share",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.05,
                help="Proportion assigned to treatment. 0.5 maximizes power"
            )
        
        # Sample size or MDE input
        st.markdown("#### Sample Size / Effect Size")
        col1, col2 = st.columns(2)
        
        # Initialize variables
        sample_size = None
        num_clusters = None
        cluster_size = None
        target_mde = None

        if calculation_mode == "mde":
            with col1:
                if design_type == "individual":
                    sample_size = st.number_input(
                        "Total Sample Size (N)",
                        min_value=10,
                        value=400,
                        step=10,
                        help="Total number of individuals in the study"
                    )
                else:
                    num_clusters = st.number_input(
                        "Number of Clusters",
                        min_value=4,
                        value=40,
                        step=2,
                        help="Total number of clusters (e.g., villages, schools)"
                    )
            
            with col2:
                if design_type == "cluster":
                    cluster_size = st.number_input(
                        "Cluster Size (m)",
                        min_value=2,
                        value=25,
                        step=1,
                        help="Average number of individuals per cluster"
                    )
                    sample_size = int(num_clusters * cluster_size)
        
        else:  # sample_size mode
            with col1:
                target_mde = st.number_input(
                    "Target MDE (Effect Size)",
                    min_value=0.01,
                    value=5.0,
                    step=0.5,
                    help="Minimum effect you want to detect (in outcome units)"
                )
            
            if design_type == "cluster":
                with col2:
                    cluster_size = st.number_input(
                        "Cluster Size (m)",
                        min_value=2,
                        value=25,
                        step=1,
                        help="Average number of individuals per cluster"
                    )
        
        # Cluster-specific parameters
        if design_type == "cluster":
            st.markdown("#### Cluster Design Parameters")
            
            # Ensure cluster_size is defined for DEFF calculation
            if cluster_size is None:
                cluster_size = 25  # Default value
            
            icc = st.number_input(
                "Intracluster Correlation (ICC)",
                min_value=0.0,
                max_value=0.99,
                value=0.05,
                step=0.01,
                help="Correlation of outcomes within clusters. Typical values: 0.01-0.20"
            )
            
            deff_value = 1 + (cluster_size - 1) * icc
            st.info(
                f"**Design Effect (DEFF):** {deff_value:.3f} — "
                f"This inflates required sample size by {(deff_value - 1) * 100:.1f}%"
            )
        else:
            icc = 0.0
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            if outcome_type == "binary":
                st.info("📌 **Note**: For binary outcomes, covariate adjustment (R²) is not applicable per J-PAL guidelines.")
                r_squared = 0.0
                compliance_rate = st.number_input(
                    "Compliance Rate",
                    min_value=0.01,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    help="Expected compliance rate (1.0 = perfect compliance)",
                    key="power_compliance_binary"
                )
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    r_squared = st.number_input(
                        "R² from Covariates",
                        min_value=0.0,
                        max_value=0.99,
                        value=0.0,
                        step=0.05,
                        help="Variance explained by baseline covariates (0 if no covariates)",
                        key="power_r_squared"
                    )
                
                with col2:
                    compliance_rate = st.number_input(
                        "Compliance Rate",
                        min_value=0.01,
                        max_value=1.0,
                        value=1.0,
                        step=0.05,
                        help="Expected compliance rate (1.0 = perfect compliance)",
                        key="power_compliance_continuous"
                    )
        
        st.markdown("---")
        st.markdown("#### Attrition Rate")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            attrition_rate = st.slider(
                "Expected Attrition Rate",
                min_value=0.00,
                max_value=0.15,
                value=0.10,
                step=0.01,
                format="%.2f",
                help="Expected proportion of participants lost to follow-up (0.05-0.15 typical)",
                key="power_attrition"
            )
        
        with col2:
            if attrition_rate > 0:
                attrition_factor = 1 / (1 - attrition_rate)
                # Display simplified inflation factor
                inflation_display = 1 + attrition_rate
                st.metric("Inflation Factor", f"{inflation_display:.2f}x")
                st.caption(f"Sample multiplied by {attrition_factor:.2f}")
        
        with col3:
            if attrition_rate > 0:
                st.info(f"💡 With {attrition_rate*100:.0f}% attrition, recruit {attrition_factor:.0%} extra participants to maintain power")
            else:
                st.info("💡 Set attrition > 0 to adjust sample size")
        
        if compliance_rate and compliance_rate < 1.0:
            st.warning(
                f"With {compliance_rate*100:.0f}% compliance, power for ITT estimates is maintained, "
                "but power for LATE (complier treatment effect) is reduced. "
                f"Effective sample size for LATE: ~{compliance_rate*100:.0f}% of nominal."
            )
        
        # Simulation-specific parameters
        if calculation_method == "simulation":
            st.markdown("---")
            st.markdown("##### Simulation Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                num_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Monte Carlo iterations. More iterations = more accurate but slower (1000 recommended)"
            )
            
            with col2:
                sim_seed = st.number_input(
                    "Random Seed",
                    min_value=1,
                    max_value=999999,
                    value=123456,
                    help="For reproducibility. Same seed = same results"
                )
            
            if design_type == "cluster":
                within_cluster_var = st.number_input(
                    "Within-Cluster Variance",
                    min_value=0.01,
                    value=1.0,
                    step=0.1,
                    help="Variance of individual-level errors within clusters"
                )
            else:
                within_cluster_var = 1.0  # Not used for individual design
        else:
            # Default values for analytical method
            num_simulations = 1000
            sim_seed = 123456
            within_cluster_var = 1.0
        
        calculate_button = st.form_submit_button("⚡ Calculate", use_container_width=True)

    current_clusters: int | None = None

    # Perform calculations
    if calculate_button:
        try:
            # Validate inputs for cluster design
            if design_type == "cluster":
                if cluster_size is None or cluster_size <= 0:
                    st.error("Please specify a valid cluster size.")
                    return
                if calculation_mode == "mde" and (num_clusters is None or num_clusters <= 0):
                    st.error("Please specify a valid number of clusters.")
                    return
            
            # Validate inputs for individual design
            if design_type == "individual" and calculation_mode == "mde":
                if sample_size is None or sample_size <= 0:
                    st.error("Please specify a valid sample size.")
                    return
            
            # Validate MDE for sample_size calculation
            if calculation_mode == "sample_size" and (target_mde is None or target_mde <= 0):
                st.error("Please specify a valid target MDE.")
                return
            
            # Ensure r_squared and compliance_rate have proper defaults
            final_r_squared = r_squared if r_squared is not None else 0.0
            final_compliance = compliance_rate if compliance_rate is not None else 1.0
            
            # ==========================================
            # SIMULATION-BASED CALCULATIONS
            # ==========================================
            if calculation_method == "simulation":
                st.session_state.pop("power_calc_state", None)
                st.markdown("---")
                st.markdown("### 🎲 Simulation Results")
                st.info(f"Running {num_simulations:,} Monte Carlo simulations... (Seed: {sim_seed})")
                
                # Note: Simulation only supports power estimation (calculation_mode must be "mde")
                if calculation_mode == "sample_size":
                    st.error("❌ Simulation method currently only supports MDE calculation (not sample size calculation). Please select 'Minimum Detectable Effect (MDE)' or switch to 'Analytical' method.")
                    return
                
                # Determine effect size for simulation
                # For simulation, we need an effect size to test power
                # If user specified MDE mode, we use the calculated MDE from analytical formula as effect
                if design_type == "individual":
                    test_effect = baseline_sd * 0.2  # Default to 0.2 SD effect for demonstration
                else:
                    test_effect = baseline_sd * 0.2
                
                #Ask user for effect size to simulate
                st.markdown("#### Effect Size to Test")
                col1, col2 = st.columns(2)
                with col1:
                    test_effect_input = st.number_input(
                        "Effect Size (absolute)",
                        min_value=0.01,
                        value=float(test_effect),
                        step=0.01,
                        help="The treatment effect you want to test statistical power for"
                    )
                with col2:
                    effect_as_pct = (test_effect_input / baseline_mean) * 100 if baseline_mean != 0 else 0
                    st.metric("As % of Baseline", f"{effect_as_pct:.1f}%")
                
                test_effect = test_effect_input
                
                # Create simulation assumptions
                sim_assumptions = SimulationAssumptions(
                    alpha=alpha,
                    test_type="two",  # Two-sided test
                    effect_size=test_effect,
                    outcome_type=outcome_type,
                    control_mean=baseline_mean,
                    control_sd=baseline_sd if outcome_type == "continuous" else None,
                    design_type=design_type,
                    treatment_share=treatment_share,
                    sample_size=int(sample_size) if design_type == "individual" else None,
                    num_clusters=int(num_clusters) if design_type == "cluster" else None,
                    cluster_size=int(cluster_size) if design_type == "cluster" else None,
                    icc=icc if design_type == "cluster" else 0.0,
                    within_cluster_var=within_cluster_var if design_type == "cluster" else 1.0,
                    num_simulations=int(num_simulations),
                    seed=int(sim_seed)
                )
                
                # Run simulation
                with st.spinner("Running simulations..."):
                    sim_result = run_power_simulation(sim_assumptions)
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Estimated Power", f"{sim_result['power']*100:.1f}%")
                    st.caption(f"{sim_result['rejections']}/{sim_result['num_simulations']} rejections")
                
                with col2:
                    st.metric("Sample Size", f"{sim_result.get('sample_size', sim_result.get('total_sample', 0)):,}")
                    if design_type == "cluster":
                        st.caption(f"{sim_result['num_clusters']} clusters × {sim_result['cluster_size']} individuals")
                
                with col3:
                    st.metric("Test Effect Size", f"{test_effect:.3f}")
                    st.caption(f"{(test_effect / baseline_mean * 100):.1f}% of baseline")
                
                with col4:
                    st.metric("Mean Estimate", f"{sim_result['mean_effect_estimate']:.3f}")
                    st.caption(f"SD: {sim_result['sd_effect_estimate']:.3f}")
                
                # Design effect and ICC for cluster designs
                if design_type == "cluster":
                    st.markdown("#### Cluster Design Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Design Effect", f"{sim_result['design_effect']:.3f}")
                    with col2:
                        st.metric("Specified ICC", f"{sim_result['specified_icc']:.3f}")
                    with col3:
                        mean_icc = sim_result.get('mean_estimated_icc', np.nan)
                        if not np.isnan(mean_icc):
                            st.metric("Mean Estimated ICC", f"{mean_icc:.3f}")
                        else:
                            st.metric("Mean Estimated ICC", "N/A")
                
                # Summary
                st.success(
                    f"✅ **Simulation Complete**: With the specified design, you have approximately "
                    f"**{sim_result['power']*100:.1f}% power** to detect an effect of **{test_effect:.3f}** "
                    f"at α = {alpha} (two-sided test)."
                )
                
                # Power curve via simulation
                st.markdown("---")
                st.markdown("### 📈 Power Curve (Simulation-Based)")
                
                with st.spinner("Generating power curve via simulation..."):
                    sim_power_df = generate_simulation_power_curve(
                        sim_assumptions,
                        num_points=6  # Fewer points since simulation is slower
                    )
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                if design_type == "cluster":
                    x_data = sim_power_df['clusters']
                    x_label = "Number of Clusters"
                    current_x = num_clusters
                else:
                    x_data = sim_power_df['sample_size']
                    x_label = "Sample Size (N)"
                    current_x = sample_size
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=sim_power_df['power'],
                    mode='lines+markers',
                    name='Simulated Power',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # Add reference lines
                fig.add_hline(y=0.8, line_dash="dash", line_color="gray", 
                             annotation_text="80% Power", annotation_position="right")
                
                if current_x:
                    fig.add_vline(x=current_x, line_dash="dash", line_color="red",
                                 annotation_text=f"Current: {int(current_x):,}")
                
                fig.update_layout(
                    title=f"Statistical Power vs Sample Size (Simulation: {num_simulations} iterations each)",
                    xaxis_title=x_label,
                    yaxis_title="Power (1-β)",
                    yaxis=dict(range=[0, 1]),
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Code generation for simulation
                st.markdown("---")
                st.markdown("### 💻 Downloadable Simulation Code")
                
                code_tab1, code_tab2 = st.tabs(["Python", "Stata"])
                
                with code_tab1:
                    python_sim_code = generate_simulation_python_code(sim_assumptions, sim_result)
                    st.code(python_sim_code, language="python")
                    st.download_button(
                        "📥 Download Python Code",
                        data=python_sim_code,
                        file_name="power_simulation.py",
                        mime="text/x-python"
                    )
                
                with code_tab2:
                    stata_sim_code = generate_simulation_stata_code(sim_assumptions, sim_result)
                    st.code(stata_sim_code, language="stata")
                    st.download_button(
                        "📥 Download Stata Code",
                        data=stata_sim_code,
                        file_name="power_simulation.do",
                        mime="text/x-stata"
                    )
                
                st.markdown("---")
                st.info("💡 **Note**: Simulation-based power calculations provide empirical validation of analytical formulas and allow for more flexible assumptions about data distributions.")
                
                return  # Exit early for simulation method
            
            # ==========================================
            # ANALYTICAL CALCULATIONS (Original Logic)
            # ==========================================
            
            # Create PowerAssumptions object
            
            if calculation_mode == "mde":
                # For MDE calculation, don't set mde in assumptions
                assumptions = PowerAssumptions(
                    alpha=alpha,
                    power=power,
                    baseline_mean=baseline_mean,
                    baseline_sd=baseline_sd,
                    outcome_type=outcome_type,
                    treatment_share=treatment_share,
                    design_type=design_type,
                    num_clusters=int(num_clusters) if design_type == "cluster" else None,
                    cluster_size=int(cluster_size) if design_type == "cluster" else None,
                    icc=icc if design_type == "cluster" else 0.0,
                    r_squared=final_r_squared,
                    compliance_rate=final_compliance,
                    attrition_rate=attrition_rate
                )
            else:
                # For sample size calculation, set mde in assumptions
                # For cluster designs, num_clusters will be computed from MDE/cluster_size
                assumptions = PowerAssumptions(
                    alpha=alpha,
                    power=power,
                    baseline_mean=baseline_mean,
                    baseline_sd=baseline_sd,
                    outcome_type=outcome_type,
                    treatment_share=treatment_share,
                    design_type=design_type,
                    mde=target_mde,
                    num_clusters=None,  # Will be calculated in sample size mode
                    cluster_size=int(cluster_size) if design_type == "cluster" else None,
                    icc=icc if design_type == "cluster" else 0.0,
                    r_squared=final_r_squared,
                    compliance_rate=final_compliance,
                    attrition_rate=attrition_rate
                )
            
            # Calculate results (store in session for reuse)
            mde_result = None
            sample_result = None
            mde = None
            required_n = None
            required_clusters = None
            total_n = None

            if calculation_mode == "mde":
                mde_result = calculate_mde(assumptions, sample_size=sample_size)
                mde = mde_result['mde_absolute']
                if design_type == "cluster" and num_clusters is not None:
                    current_clusters = int(num_clusters)
            else:
                sample_result = calculate_sample_size(assumptions)
                if design_type == "individual":
                    required_n = int(sample_result['sample_size'])
                else:
                    required_clusters = int(sample_result['num_clusters'])
                    total_n = int(sample_result['total_individuals'])
                    current_clusters = int(required_clusters)

            clusters_for_curve = current_clusters if design_type == "cluster" else None
            code_results = mde_result if calculation_mode == "mde" else sample_result
            effect_for_curve = mde if calculation_mode == "mde" else target_mde

            curve_params = {
                "alpha": alpha,
                "power": power,
                "baseline_mean": baseline_mean,
                "baseline_sd": baseline_sd,
                "outcome_type": outcome_type,
                "treatment_share": treatment_share,
                "design_type": design_type,
                "effect_size": effect_for_curve,
                "num_clusters": clusters_for_curve if design_type == "cluster" else None,
                "cluster_size": int(cluster_size) if design_type == "cluster" and cluster_size else None,
                "icc": icc if design_type == "cluster" else 0.0,
                "r_squared": final_r_squared,
                "compliance_rate": final_compliance,
                "attrition_rate": attrition_rate,
            }

            st.session_state.power_calc_state = {
                "calculation_method": "analytical",
                "calculation_mode": calculation_mode,
                "design_type": design_type,
                "outcome_type": outcome_type,
                "alpha": alpha,
                "power": power,
                "baseline_mean": baseline_mean,
                "baseline_sd": baseline_sd,
                "treatment_share": treatment_share,
                "sample_size": int(sample_size) if sample_size else None,
                "num_clusters": int(num_clusters) if num_clusters else None,
                "cluster_size": int(cluster_size) if cluster_size else None,
                "target_mde": target_mde,
                "mde": mde,
                "mde_result": mde_result,
                "sample_result": sample_result,
                "required_n": required_n,
                "required_clusters": required_clusters,
                "total_n": total_n,
                "final_r_squared": final_r_squared,
                "final_compliance": final_compliance,
                "attrition_rate": attrition_rate,
                "icc": icc,
                "clusters_for_curve": clusters_for_curve,
                "table_num_clusters": (clusters_for_curve or 50) if design_type == "cluster" else None,
                "code_results": code_results,
                "assumption_params": asdict(assumptions),
                "curve_params": curve_params,
            }

        except ValueError as e:
            st.error(f"❌ Calculation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

    power_state = st.session_state.get("power_calc_state")
    if power_state and power_state.get("calculation_method") == "analytical":
        render_power_analysis_results(power_state)

# ----------------------------------------------------------------------------- #
# CASE ASSIGNMENT                                                               #
# ----------------------------------------------------------------------------- #


def render_case_assignment() -> None:
    st.title("📋 Case Assignment")
    st.markdown("Assign interview cases to SurveyCTO teams and produce upload-ready cases dataset.")

    # Data source selection
    df = st.session_state.case_data
    upload = st.file_uploader("Upload randomized data (CSV)", type="csv", key="case_upload")
    if upload:
        df = pd.read_csv(upload)
        st.session_state.case_data = df
        st.success(f"Loaded {len(df):,} rows for case assignment.")

    if df is None:
        st.info("💡 Provide randomized data via the Randomization tab or upload a CSV here.")
        return

    st.markdown("#### 📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    available_cols = df.columns.tolist()
    
    # Configuration mode selector
    st.markdown("---")
    config_mode = st.radio(
        "Configuration mode",
        ["Interactive (Recommended)", "YAML (Advanced)"],
        key="case_config_mode",
        help="Interactive mode provides forms for easy configuration. YAML mode is for advanced users."
    )
    
    if config_mode == "YAML (Advanced)":
        st.markdown("#### ⚙️ Configuration (YAML)")
        default_case_cfg = load_default_config().get("case_assignment", {})
        config_text = st.text_area(
            "Case assignment configuration",
            value=yaml_dump(default_case_cfg),
            height=400,
            key="case_config_text",
        )

        if st.button("Generate SurveyCTO cases dataset", type="primary"):
            try:
                config = yaml_load(config_text)
                roster = assign_cases(df, config)
            except Exception as exc:
                st.error(f"Assignment failed: {exc}")
                return

            st.success(f"✅ Generated cases with {len(roster):,} cases.")
            st.dataframe(roster.head(20), use_container_width=True)

            csv_buffer = io.StringIO()
            roster.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download roster CSV",
                data=csv_buffer.getvalue(),
                file_name="surveycto_case_roster.csv",
                mime="text/csv",
            )
    else:
        # Interactive configuration form
        st.markdown("#### ⚙️ Case Assignment Configuration")
        
        with st.form("case_assignment_form"):
            st.markdown("##### Basic Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                case_id_column = st.selectbox(
                    "Case ID column",
                    available_cols,
                    index=available_cols.index("participant_id") if "participant_id" in available_cols else 0,
                    key="case_id_column",
                    help="Column containing unique case identifiers"
                )
                
                treatment_column = st.selectbox(
                    "Treatment column",
                    available_cols,
                    index=available_cols.index("treatment") if "treatment" in available_cols else 0,
                    key="case_treatment_column",
                    help="Column containing treatment assignment"
                )
            
            with col2:
                team_column = st.selectbox(
                    "Team column (optional)",
                    ["None"] + available_cols,
                    key="case_team_column",
                    help="Existing column with team assignments (if any)"
                )
                team_column = None if team_column == "None" else team_column
                
                default_team = st.text_input(
                    "Default team name",
                    value="unassigned",
                    key="case_default_team",
                    help="Team name for cases not matching any rule"
                )
            
            # Label template
            st.markdown("##### Case Label Template")
            label_template = st.text_input(
                "Label template",
                value="{" + case_id_column + "}",
                key="case_label_template",
                help=f"Use {{column_name}} to reference columns. Example: {{{case_id_column}}} - {{community}}"
            )
            
            # Team rules
            st.markdown("##### Team Assignment Rules")
            st.markdown("Define rules to automatically assign cases to teams based on data values.")
            
            num_rules = st.number_input(
                "Number of team rules",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="case_num_rules",
                help="Rules are evaluated in order. First matching rule wins."
            )
            
            team_rules = []
            for i in range(int(num_rules)):
                st.markdown(f"**Rule {i+1}**")
                rule_col1, rule_col2, rule_col3 = st.columns([2, 2, 1])
                
                with rule_col1:
                    rule_name = st.text_input(
                        f"Team name for rule {i+1}",
                        key=f"rule_name_{i}",
                        placeholder=f"team_{chr(97+i)}"
                    )
                
                with rule_col2:
                    match_column = st.selectbox(
                        "Match on column",
                        available_cols,
                        key=f"rule_match_col_{i}"
                    )
                
                with rule_col3:
                    # Show unique values for the selected column
                    unique_vals = df[match_column].dropna().unique().tolist()
                    
                match_values = st.multiselect(
                    "Match these values",
                    unique_vals,
                    key=f"rule_match_vals_{i}",
                    help=f"Cases with {match_column} in this list will be assigned to {rule_name or 'this team'}"
                )
                
                if rule_name and match_values:
                    team_rules.append({
                        "name": rule_name,
                        "match": {match_column: match_values}
                    })
            
            # Form IDs configuration - improved interface
            st.markdown("##### SurveyCTO Form IDs")
            st.markdown("Specify form IDs for each treatment arm or case criteria.")
            
            treatment_arms = df[treatment_column].dropna().unique().tolist()
            form_ids = {}
            
            # Default form ID
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                default_forms = st.text_input(
                    "Default form ID(s)",
                    value="follow_up",
                    key="case_default_forms",
                    help="Comma-separated form IDs for cases not matching specific criteria"
                )
                form_ids["default"] = [f.strip() for f in default_forms.split(",") if f.strip()]
            
            with form_col2:
                form_separator = st.text_input(
                    "Form ID separator",
                    value=",",
                    key="case_form_separator",
                    help="Character to separate multiple form IDs in cases"
                )
            
            # Form ID assignment with search/filter
            st.markdown("**Treatment-Specific Form IDs**")
            st.markdown("Click to expand and assign form IDs to treatment arms or use search")
            
            # Create a more efficient interface with expanders or search
            col1, col2 = st.columns([2, 1])
            
            with col1:
                form_search = st.text_input(
                    "🔍 Search or filter treatment arms",
                    value="",
                    key="case_form_search",
                    placeholder="Type to search treatment arms...",
                    help="Start typing to filter the list of treatment arms"
                )
            
            with col2:
                add_form_id = st.checkbox(
                    "Add custom form IDs",
                    value=False,
                    key="case_add_custom_forms",
                    help="Click to add treatment-specific form IDs"
                )
            
            if add_form_id:
                # Filter treatment arms based on search
                filtered_arms = [arm for arm in treatment_arms 
                                if form_search.lower() in str(arm).lower()] if form_search else treatment_arms
                
                if not filtered_arms and form_search:
                    st.info(f"No treatment arms match '{form_search}'. Showing all arms below.")
                    filtered_arms = treatment_arms
                
                # Show expandable sections for each arm
                for i, arm in enumerate(filtered_arms):
                    with st.expander(f"📋 {arm} - Form IDs", expanded=False):
                        arm_forms = st.text_input(
                            f"Form ID(s) for '{arm}'",
                            key=f"case_forms_{arm}",
                            placeholder="e.g., form_1, form_2",
                            help=f"Comma-separated form IDs for {arm} cases. Leave blank to use default."
                        )
                        if arm_forms:
                            form_ids[str(arm)] = [f.strip() for f in arm_forms.split(",") if f.strip()]
                    
                    # Show count of cases for this arm
                    arm_count = len(df[df[treatment_column] == arm])
                    st.caption(f"  └─ {arm_count:,} cases in this arm")
            else:
                # Show summary without expansion
                st.info("💡 Check 'Add custom form IDs' above to assign specific forms to each treatment arm")
                st.markdown(f"**Treatment Arms in Data ({len(treatment_arms)} total):**")
                
                # Show arms in a compact table format
                arms_data = []
                for arm in treatment_arms:
                    count = len(df[df[treatment_column] == arm])
                    arms_data.append({
                        "Treatment Arm": arm,
                        "Cases": count,
                        "Form IDs": form_ids.get(str(arm), ["(using default)"])[0] if str(arm) in form_ids else "(using default)"
                    })
                
                arms_df = pd.DataFrame(arms_data)
                st.dataframe(arms_df, use_container_width=True, hide_index=True)
            
            # Additional columns
            st.markdown("##### Additional Roster Columns")
            additional_columns = st.multiselect(
                "Include these columns in the roster",
                [col for col in available_cols if col not in [case_id_column, treatment_column]],
                key="case_additional_columns",
                help="Extra columns to include in the SurveyCTO case roster"
            )
            
            # Submit button
            submitted = st.form_submit_button("Generate SurveyCTO Cases", type="primary")
        
        if submitted:
            # Build configuration
            config = {
                "case_id_column": case_id_column,
                "label_template": label_template,
                "team_column": team_column,
                "default_team": default_team,
                "team_rules": team_rules,
                "form_ids": form_ids,
                "additional_columns": additional_columns,
                "treatment_column": treatment_column,
                "form_separator": form_separator
            }
            
            try:
                roster = assign_cases(df, config)
                
                st.success(f"✅ Generated cases with {len(roster):,} cases!")
                
                # Show team distribution
                st.markdown("#### Team Distribution")
                team_counts = roster['users'].value_counts()
                team_dist = pd.DataFrame({
                    'Team': team_counts.index,
                    'Cases': team_counts.values,
                    'Percentage': (team_counts.values / len(roster) * 100).round(1).astype(str) + '%'
                })
                st.dataframe(team_dist, use_container_width=True, hide_index=True)
                
                # Show form distribution
                st.markdown("#### Form Assignment Distribution")
                form_counts = roster['formids'].value_counts()
                form_dist = pd.DataFrame({
                    'Form ID(s)': form_counts.index,
                    'Cases': form_counts.values
                })
                st.dataframe(form_dist, use_container_width=True, hide_index=True)
                
                st.markdown("#### Cases Preview")
                st.dataframe(roster.head(20), use_container_width=True)
                
                # Store roster in session state for upload
                st.session_state['generated_roster'] = roster
                
                # Download and Upload options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_buffer = io.StringIO()
                    roster.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📥 Download cases CSV",
                        data=csv_buffer.getvalue(),
                        file_name="surveycto_case_roster.csv",
                        mime="text/csv",
                    )
                
                with col2:
                    if st.button("🚀 Upload to SurveyCTO", help="Upload cases via SurveyCTO API"):
                        st.session_state['show_upload_form'] = True
                
            except Exception as exc:
                st.error(f"❌ Assignment failed: {exc}")
                import traceback
                st.code(traceback.format_exc())
    
    # Upload to SurveyCTO section
    if st.session_state.get('show_upload_form') and st.session_state.get('generated_roster') is not None:
        st.markdown("---")
        st.markdown("### 🚀 Upload Cases to SurveyCTO")
        
        roster = st.session_state['generated_roster']
        
        # Validate roster has required columns
        required_cols = ['id', 'label', 'users', 'formids']
        missing_cols = [col for col in required_cols if col not in roster.columns]
        
        if missing_cols:
            st.error(f"❌ Cases missing required columns: {missing_cols}")
            st.info("Required columns: id (or caseid), label, users, formids")
        else:
            # Rename 'id' to 'caseid' if needed (SurveyCTO expects 'caseid')
            upload_roster = roster.copy()
            if 'id' in upload_roster.columns and 'caseid' not in upload_roster.columns:
                upload_roster = upload_roster.rename(columns={'id': 'caseid'})
            
            with st.form("upload_cases_form"):
                st.markdown("#### SurveyCTO Connection")
                
                col1, col2 = st.columns(2)
                with col1:
                    scto_server = st.text_input(
                        "SurveyCTO Server",
                        placeholder="yourserver.surveycto.com",
                        help="Your SurveyCTO server URL"
                    )
                    scto_username = st.text_input(
                        "Username",
                        help="SurveyCTO account username"
                    )
                
                with col2:
                    scto_form_id = st.text_input(
                        "Form ID",
                        help="The form ID to upload cases for"
                    )
                    scto_password = st.text_input(
                        "Password",
                        type="password",
                        help="SurveyCTO account password"
                    )
                
                st.markdown("#### Upload Options")
                upload_mode = st.radio(
                    "Upload mode",
                    ["merge", "append", "replace"],
                    help="""
                    • merge: Update existing cases and add new ones (recommended)
                    • append: Only add new cases, skip existing ones
                    • replace: Delete ALL existing cases and upload only these
                    """
                )
                
                mode_descriptions = {
                    "merge": "✅ **Merge** will update existing cases with matching IDs and add new cases.",
                    "append": "➕ **Append** will only add new cases and skip any with existing IDs.",
                    "replace": "⚠️ **Replace** will DELETE all existing cases and upload only these cases!"
                }
                st.info(mode_descriptions[upload_mode])
                
                if upload_mode == "replace":
                    st.warning("⚠️ WARNING: Replace mode will permanently delete all existing cases. This cannot be undone!")
                    confirm_replace = st.checkbox("I understand this will delete all existing cases")
                else:
                    confirm_replace = True
                
                upload_submitted = st.form_submit_button(
                    f"Upload {len(upload_roster)} cases to SurveyCTO",
                    type="primary" if upload_mode != "replace" else "secondary",
                    disabled=not confirm_replace
                )
            
            if upload_submitted:
                if not all([scto_server, scto_username, scto_password, scto_form_id]):
                    st.error("❌ Please fill in all SurveyCTO connection fields")
                else:
                    try:
                        with st.spinner(f"Uploading {len(upload_roster)} cases to SurveyCTO..."):
                            client = SurveyCTO(
                                server=scto_server,
                                username=scto_username,
                                password=scto_password
                            )
                            
                            result = client.upload_cases_from_dataframe(
                                df=upload_roster,
                                form_id=scto_form_id,
                                mode=upload_mode
                            )
                            
                            st.success("✅ Successfully uploaded cases to SurveyCTO!")
                            st.json(result)
                            
                            # Clear the upload form
                            if st.button("Close upload form"):
                                st.session_state['show_upload_form'] = False
                                st.rerun()
                            
                    except requests.exceptions.HTTPError as e:
                        st.error(f"❌ Upload failed: {e}")
                        st.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response details'}")
                    except Exception as exc:
                        st.error(f"❌ Upload failed: {exc}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Add visualizations and validation
            st.markdown("---")
            st.markdown("### 📊 Roster Validation & Visualizations")
            
            if upload_roster is not None and not upload_roster.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Cases", len(upload_roster))
                
                with col2:
                    if 'team' in upload_roster.columns:
                        unique_teams = upload_roster['team'].nunique()
                        st.metric("Teams", unique_teams)
                
                with col3:
                    if treatment_column in upload_roster.columns:
                        unique_arms = upload_roster[treatment_column].nunique()
                        st.metric("Treatment Arms", unique_arms)
                
                with col4:
                    st.metric("Status", "✅ Ready to upload")
                
                # Team distribution
                if 'team' in upload_roster.columns:
                    st.markdown("#### 👥 Distribution by Team")
                    team_dist = upload_roster['team'].value_counts().reset_index()
                    team_dist.columns = ['Team', 'Cases']
                    st.dataframe(team_dist, use_container_width=True)
                    
                    # Visualize
                    st.bar_chart(team_dist.set_index('Team')['Cases'])
                
                # Treatment arm distribution
                if treatment_column in upload_roster.columns:
                    st.markdown("#### 🎯 Distribution by Treatment Arm")
                    arm_dist = upload_roster[treatment_column].value_counts().reset_index()
                    arm_dist.columns = ['Treatment Arm', 'Cases']
                    arm_dist['Percentage'] = (arm_dist['Cases'] / arm_dist['Cases'].sum() * 100).round(1)
                    st.dataframe(arm_dist, use_container_width=True)
                
                # Validation checks
                st.markdown("#### ✓ Validation Checks")
                
                validation_passed = True
                
                # Check for duplicates
                if case_id_column in upload_roster.columns:
                    duplicates = upload_roster[case_id_column].duplicated().sum()
                    if duplicates > 0:
                        st.warning(f"⚠️ {duplicates} duplicate case IDs found")
                        validation_passed = False
                    else:
                        st.success("✓ No duplicate case IDs")
                
                # Check for missing values
                missing = upload_roster.isnull().sum()
                if missing.sum() > 0:
                    st.warning(f"⚠️ {missing.sum()} missing values found")
                    validation_passed = False
                else:
                    st.success("✓ No missing values")
                
                # Check treatment distribution
                if treatment_column in upload_roster.columns:
                    arm_counts = upload_roster[treatment_column].value_counts()
                    if len(arm_counts) > 1:
                        imbalance = (arm_counts.max() - arm_counts.min()) / len(upload_roster) * 100
                        if imbalance > 20:
                            st.info(f"ℹ️ Moderate distribution imbalance ({imbalance:.1f}%)")
                        else:
                            st.success("✓ Treatment arm distribution balanced")
                
                if validation_passed:
                    st.success("✅ All validation checks passed!")


# ----------------------------------------------------------------------------- #
# QUALITY CHECKS                                                                #
# ----------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------- #
# DATA VISUALIZATION                                                            #
# ----------------------------------------------------------------------------- #


def detect_unique_id(df: pd.DataFrame) -> List[str]:
    """
    Detect potential unique ID columns in the dataset.
    Prioritizes SurveyCTO KEY column if present.
    
    Returns list of column names that could serve as unique identifiers,
    ordered by likelihood (columns with all unique values first).
    """
    candidates = []
    
    # SurveyCTO KEY column gets highest priority
    if 'KEY' in df.columns:
        if df['KEY'].nunique() == len(df):
            candidates.append(('KEY', 2.0))  # Highest priority
    
    for col in df.columns:
        if col == 'KEY':  # Already handled
            continue
            
        # Check if column has all unique values
        if df[col].nunique() == len(df) and df[col].notna().all():
            candidates.append((col, 1.0))  # Perfect candidate
        elif df[col].nunique() == len(df):
            candidates.append((col, 0.9))  # Has nulls but otherwise unique
        elif df[col].nunique() / len(df) > 0.95:  # More than 95% unique
            candidates.append((col, 0.8))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [col for col, score in candidates]


def detect_duplicates(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Detect and return duplicate rows based on ID column."""
    duplicates = df[df.duplicated(subset=[id_col], keep=False)].copy()
    if not duplicates.empty:
        duplicates = duplicates.sort_values(by=[id_col])
    return duplicates


def detect_wide_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect repeated patterns in column names that indicate wide format data.
    
    Returns dict with pattern as key and list of matching columns as values.
    Patterns detected:
    - var_1, var_2, ... (suffix with underscore) - SurveyCTO repeat groups in wide format
    - var1, var2, ... (suffix without underscore)
    - var_1_1, var_1_2, ... (nested repeats)
    - var1_1, var1_2, ... (nested without underscore)
    
    Special SurveyCTO patterns:
    - geopoint fields: fieldname-Latitude, fieldname-Longitude, fieldname-Altitude, fieldname-Accuracy
    - select_multiple: fieldname_1, fieldname_2, ... (dummy variables)
    """
    import re
    
    patterns = {}
    geopoint_groups = {}
    
    # Detect geopoint fields (4 columns: Latitude, Longitude, Altitude, Accuracy)
    for col in df.columns:
        if col.endswith('-Latitude'):
            base = col[:-9]  # Remove '-Latitude'
            expected_cols = [f"{base}-Latitude", f"{base}-Longitude", 
                           f"{base}-Altitude", f"{base}-Accuracy"]
            if all(c in df.columns for c in expected_cols):
                geopoint_groups[f"{base} (geopoint)"] = expected_cols
    
    # Add geopoint patterns (these are special, not for reshaping but for grouping)
    for key, cols in geopoint_groups.items():
        patterns[key] = cols
    
    # Pattern 1: var_N or var_N_M (with underscores)
    pattern1 = re.compile(r'^(.+?)_(\d+)(?:_(\d+))?$')
    # Pattern 2: varN or varNM (without underscores)  
    pattern2 = re.compile(r'^(.+?)(\d+)(?:_(\d+))?$')
    
    # Skip geopoint component columns
    geopoint_cols = set()
    for cols in geopoint_groups.values():
        geopoint_cols.update(cols)
    
    for col in df.columns:
        # Skip SurveyCTO metadata columns
        if col in ['KEY', 'instanceID', 'SubmissionDate', 'formdef_version', 
                   'review_status', 'review_comments', 'review_corrections', 'review_quality']:
            continue
        
        # Skip geopoint component columns
        if col in geopoint_cols:
            continue
            
        # Try pattern with underscores first
        match = pattern1.match(col)
        if match:
            base = match.group(1)
            if match.group(3):  # Nested pattern
                key = f"{base}_*_*"
            else:
                key = f"{base}_*"
            
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(col)
            continue
        
        # Try pattern without underscores
        match = pattern2.match(col)
        if match:
            base = match.group(1)
            if match.group(3):  # Nested pattern
                key = f"{base}*_*"
            else:
                key = f"{base}*"
            
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(col)
    
    # Filter out patterns with less than 2 columns (not really repeated)
    # Exception: keep geopoint groups even if marked as pattern
    patterns = {k: v for k, v in patterns.items() 
               if len(v) >= 2 or '(geopoint)' in k}
    
    return patterns


def reshape_single_pattern(
    df: pd.DataFrame,
    id_col: str,
    pattern: str,
    cols: List[str]
) -> pd.DataFrame:
    """
    Reshape a single pattern from wide to long format.
    Memory-efficient by processing only one pattern at a time.
    
    Args:
        df: DataFrame in wide format
        id_col: Column to use as identifier
        pattern: Pattern string (e.g., 'var_*')
        cols: List of columns matching this pattern
        
    Returns:
        DataFrame in long format for this specific pattern
    """
    import re
    
    # Skip geopoint patterns
    if '(geopoint)' in pattern:
        return df[[id_col] + cols].copy()
    
    # Determine if nested pattern
    is_nested = pattern.count('*') > 1
    
    # Extract base name - preserve full variable name, just remove the repeat suffix pattern
    # For pattern 'shs_count_g1_*', base_name should be 'shs_count_g1', not 'shs'
    base_name = pattern.replace('_*', '').replace('*', '')
    
    if is_nested:
        # Nested pattern - create temp data list
        temp_data = []
        for col in cols:
            if '_' in pattern.replace('*', ''):
                match = re.search(r'_(\d+)_(\d+)$', col)
            else:
                match = re.search(r'(\d+)_(\d+)$', col)
            
            if match:
                idx1, idx2 = match.groups()
                # Process in chunks to save memory
                for idx in range(0, len(df), 10000):
                    chunk = df[col].iloc[idx:idx+10000]
                    for i, val in enumerate(chunk, start=idx):
                        parent_key = df[id_col].iloc[i]
                        temp_data.append({
                            'KEY': f"{parent_key}_{idx1}_{idx2}",
                            'PARENT_KEY': parent_key,
                            'repeat_1': int(idx1),
                            'repeat_2': int(idx2),
                            base_name: val
                        })
        
        if temp_data:
            nested_df = pd.DataFrame(temp_data)
            # Remove empty rows
            data_cols = [c for c in nested_df.columns if c not in ['KEY', 'PARENT_KEY', 'repeat_1', 'repeat_2']]
            if data_cols:
                mask = nested_df[data_cols].notna().any(axis=1)
                for col in data_cols:
                    if nested_df[col].dtype == 'object':
                        mask = mask & (nested_df[col] != 'None') & (nested_df[col] != 'nan') & (nested_df[col] != '')
                nested_df = nested_df[mask]
            
            # Drop rows where index variables are blank (if they exist)
            index_cols = [c for c in nested_df.columns if c.endswith('_index') or c.startswith('index')]
            if index_cols:
                for idx_col in index_cols:
                    if idx_col in nested_df.columns:
                        if nested_df[idx_col].dtype == 'object':
                            nested_df = nested_df[
                                nested_df[idx_col].notna() & 
                                (nested_df[idx_col] != '') & 
                                (nested_df[idx_col] != 'None') & 
                                (nested_df[idx_col] != 'nan')
                            ]
                        else:
                            nested_df = nested_df[nested_df[idx_col].notna()]
            
            return nested_df
        return pd.DataFrame()
    else:
        # Single level pattern - use melt
        temp_df = df[[id_col] + cols].copy()
        
        melted = pd.melt(
            temp_df,
            id_vars=[id_col],
            value_vars=cols,
            var_name='_temp_var',
            value_name=base_name
        )
        
        # Extract repeat index
        if '_' in pattern:
            repeat_series = melted['_temp_var'].str.extract(r'_(\d+)$')[0]
        else:
            repeat_series = melted['_temp_var'].str.extract(r'(\d+)$')[0]
        
        melted['repeat'] = pd.to_numeric(repeat_series, errors='coerce').astype('Int64')
        melted = melted.drop('_temp_var', axis=1)
        
        # Add PARENT_KEY (original KEY from wide format) and generate unique KEY for each row
        melted = melted.rename(columns={id_col: 'PARENT_KEY'})
        melted['KEY'] = [f"{parent_key}_{repeat}" for parent_key, repeat in zip(melted['PARENT_KEY'], melted['repeat'])]
        
        # Reorder columns: KEY, PARENT_KEY, repeat, then data columns
        cols_order = ['KEY', 'PARENT_KEY', 'repeat'] + [c for c in melted.columns if c not in ['KEY', 'PARENT_KEY', 'repeat']]
        melted = melted[cols_order]
        
        # Remove empty rows immediately after reshape (drop rows where all data columns are null/None)
        data_cols = [c for c in melted.columns if c not in ['KEY', 'PARENT_KEY', 'repeat']]
        if data_cols:
            # Drop rows where all data columns are null or string 'None'
            mask = melted[data_cols].notna().any(axis=1)
            for col in data_cols:
                if melted[col].dtype == 'object':
                    mask = mask & (melted[col] != 'None') & (melted[col] != 'nan') & (melted[col] != '')
            melted = melted[mask]
        
        # Drop rows where index variables are blank (if they exist)
        # Check for columns ending with '_index' or starting with 'index'
        index_cols = [c for c in melted.columns if c.endswith('_index') or c.startswith('index')]
        if index_cols:
            before_index_drop = len(melted)
            for idx_col in index_cols:
                if idx_col in melted.columns:
                    # Drop rows where index column is null or blank
                    if melted[idx_col].dtype == 'object':
                        melted = melted[
                            melted[idx_col].notna() & 
                            (melted[idx_col] != '') & 
                            (melted[idx_col] != 'None') & 
                            (melted[idx_col] != 'nan')
                        ]
                    else:
                        melted = melted[melted[idx_col].notna()]
            
            dropped_by_index = before_index_drop - len(melted)
            if dropped_by_index > 0:
                # Note: This will be visible if called with progress tracking
                pass
        
        return melted


def reshape_wide_to_long(
    df: pd.DataFrame,
    id_col: str,
    pattern_cols: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Reshape wide format data to long format based on detected patterns.
    Intelligently merges patterns from the same repeat group (same suffixes).
    
    Args:
        df: DataFrame in wide format
        id_col: Column to use as identifier (like KEY in SurveyCTO)
        pattern_cols: Dict from detect_wide_patterns()
        
    Returns:
        Dictionary mapping repeat group names to their long-format DataFrames (each with KEY/PARENT_KEY)
    """
    import re
    
    if not pattern_cols:
        return {}

    def _structure_signature(frame: pd.DataFrame) -> tuple[int, bool, int]:
        """
        Signature for grouping patterns.
        Primarily uses row count (SurveyCTO rule: same repeat group ⇒ same number of rows),
        but distinguishes top-level vs nested repeats via PARENT_KEY presence and repeat depth.
        """
        row_count = len(frame)
        has_parent = "PARENT_KEY" in frame.columns
        repeat_depth = len([c for c in frame.columns if c.startswith("repeat")])
        return (row_count, has_parent, repeat_depth or 0)

    def _has_multiple_matches(frame: pd.DataFrame, keys: List[str]) -> bool:
        """Check if dataset has more than one row per merge key combination."""
        if not keys:
            return False
        return frame.duplicated(subset=keys).any()

    def _choose_dataset_name(name_candidates: List[str], base_candidates: List[str]) -> str:
        """Select the most descriptive dataset name available for a repeat group."""
        filtered = [name for name in name_candidates if name]
        repeat_focused = [name for name in filtered if "repeat" in name.lower()]
        if repeat_focused:
            return repeat_focused[0]
        if filtered:
            return filtered[0]

        unique_bases: List[str] = []
        for base in base_candidates:
            if base and base not in unique_bases:
                unique_bases.append(base)
        if unique_bases:
            merged = "_".join(unique_bases[:3])
            return f"{merged}_repeat"
        return "repeat_group"
    
    st.write("### 🔄 Reshaping Wide to Long Format")
    st.info("ℹ️ Patterns from the same repeat group will be merged together")
    
    # Group patterns by their repeat structure (same suffixes)
    repeat_groups = {}  # Maps repeat_key -> list of (pattern, cols)
    patterns_skipped = []
    
    # Create progress expander for detailed progress
    progress_expander = st.expander("📋 Reshaping Progress", expanded=True)
    
    with progress_expander:
        st.write("**Step 1: Grouping patterns by repeat structure...**")
    
    for pattern, cols in pattern_cols.items():
        # Skip geopoint patterns (just for grouping display)
        if '(geopoint)' in pattern:
            continue
        
        # Skip if too many columns
        if len(cols) > 100:
            patterns_skipped.append(f"{pattern} ({len(cols)} columns - too many)")
            continue
        
        # Extract repeat indices from first column to determine repeat structure
        if cols:
            first_col = cols[0]
            # Extract all numbers from column name
            numbers = re.findall(r'_(\d+)', first_col)
            
            if numbers:
                # Create repeat key based on number of repeat levels
                if len(numbers) == 1:
                    # Single level repeat: firstn_1, firstn_2, etc.
                    repeat_key = "repeat_single"
                    num_repeats = len(cols)
                elif len(numbers) == 2:
                    # Nested repeat: var_1_1, var_1_2, etc.
                    repeat_key = "repeat_nested"
                    num_repeats = len(cols)
                else:
                    repeat_key = f"repeat_{len(numbers)}level"
                    num_repeats = len(cols)
                
                # Check memory limits based on repeat group
                if repeat_key not in repeat_groups:
                    repeat_groups[repeat_key] = []
                
                repeat_groups[repeat_key].append((pattern, cols))
    
    with progress_expander:
        st.write(f"  ✓ Found {len(repeat_groups)} repeat group(s)")
        for repeat_key, patterns_list in repeat_groups.items():
            st.write(f"    - {repeat_key}: {len(patterns_list)} variable pattern(s)")
    
    # Now reshape each pattern and group by repeat structure
    with progress_expander:
        st.write("\n**Step 2: Reshaping and intelligently grouping patterns...**")
    
    # First, reshape all patterns
    reshaped_patterns = {}  # Maps pattern_name -> (reshaped_df, repeat_group_name, base_name)
    
    for repeat_key, patterns_list in repeat_groups.items():
        for pattern, cols in patterns_list:
            try:
                reshaped = reshape_single_pattern(df, id_col, pattern, cols)
                if not reshaped.empty:
                    base_name = pattern.replace('_*', '').replace('*', '')
                    
                    # Extract repeat group identifier from index columns
                    index_cols = [c for c in reshaped.columns if c.endswith('_index') or c.startswith('index')]
                    
                    if index_cols:
                        first_index = index_cols[0]
                        if first_index.endswith('_index'):
                            repeat_group_name = first_index[:-6]
                        elif first_index.startswith('index'):
                            repeat_group_name = first_index
                        else:
                            repeat_group_name = base_name
                    else:
                        repeat_group_name = base_name
                    
                    reshaped_patterns[pattern] = (reshaped, repeat_group_name, base_name)
                    
                    with progress_expander:
                        st.write(f"  ✓ {pattern}: {len(reshaped):,} rows")
            except Exception as e:
                with progress_expander:
                    st.warning(f"  ✗ {pattern}: {str(e)}")
                patterns_skipped.append(f"{pattern} (error: {str(e)})")
    
    # Step 3: Group patterns by (repeat_group_name, row_count) - patterns with same rows belong together
    with progress_expander:
        st.write(f"\n**Step 3: Grouping {len(reshaped_patterns)} pattern(s) by repeat structure...**")
    
    repeat_structure_groups: Dict[tuple[int, bool, int], Dict[str, Any]] = {}
    
    for pattern, (reshaped_df, repeat_group_name, base_name) in reshaped_patterns.items():
        row_count = len(reshaped_df)
        structure_key = _structure_signature(reshaped_df)
        
        if structure_key not in repeat_structure_groups:
            repeat_structure_groups[structure_key] = {
                "row_count": row_count,
                "repeat_depth": structure_key[2],
                "has_parent_key": structure_key[1],
                "patterns": [],
                "repeat_names": [],
                "base_names": [],
                "display_name": None,
            }
        
        group_entry = repeat_structure_groups[structure_key]
        group_entry["patterns"].append((pattern, reshaped_df, base_name))
        group_entry["repeat_names"].append(repeat_group_name)
        group_entry["base_names"].append(base_name)
    
    with progress_expander:
        st.write(f"  ✓ Found {len(repeat_structure_groups)} unique repeat structure(s)")
        for group_data in repeat_structure_groups.values():
            group_data["display_name"] = _choose_dataset_name(
                group_data["repeat_names"],
                group_data["base_names"],
            )
            st.write(
                f"    - {group_data['display_name']} • {group_data['row_count']:,} rows "
                f"(repeat depth: {group_data['repeat_depth']}) "
                f"→ {len(group_data['patterns'])} variable pattern(s)"
            )
    
    # Step 4: Merge patterns within each repeat structure group
    with progress_expander:
        st.write(f"\n**Step 4: Merging patterns within each repeat structure...**")
    
    final_datasets = {}
    used_dataset_names: set[str] = set()
    
    for group_key, group_data in repeat_structure_groups.items():
        repeat_group_name = group_data.get("display_name") or "repeat_group"
        row_count = group_data.get("row_count", 0)
        patterns_in_group = group_data.get("patterns", [])
        with progress_expander:
            st.write(f"\n  Processing: {repeat_group_name} ({row_count} rows, {len(patterns_in_group)} patterns)")
        
        # Merge all patterns in this group on KEY
        merged_df = None
        merged_var_names = []
        skipped_in_group = []
        
        for idx, (pattern, reshaped_df, base_name) in enumerate(patterns_in_group):
            with progress_expander:
                st.write(f"    [{idx+1}/{len(patterns_in_group)}] Processing {base_name}...")
            
            if merged_df is None:
                merged_df = reshaped_df.copy()
                merged_var_names.append(base_name)
                with progress_expander:
                    st.write(f"      ✓ Starting with {base_name} ({len(reshaped_df)} rows, {len(reshaped_df.columns)} cols)")
            else:
                # Get structural columns that should be excluded from overlap check
                structural_cols = ['KEY', 'PARENT_KEY', 'repeat', 'repeat_1', 'repeat_2']
                structural_cols.extend([c for c in reshaped_df.columns if c.endswith('_index') or c.startswith('index')])
                
                # Get actual data columns from the new dataset
                data_cols = [c for c in reshaped_df.columns if c not in structural_cols]
                
                # Get existing data columns from merged dataset  
                existing_data_cols = [c for c in merged_df.columns if c not in structural_cols]
                
                # Check for TRUE overlaps (same column name)
                overlapping = set(data_cols) & set(existing_data_cols)
                
                with progress_expander:
                    st.write(f"      Data cols to add: {data_cols}")
                    st.write(f"      Existing data cols: {existing_data_cols}")
                    st.write(f"      Overlapping: {overlapping}")
                
                if overlapping:
                    with progress_expander:
                        st.write(f"      ⚠️ Skipping {base_name} - overlapping columns: {overlapping}")
                    skipped_in_group.append(base_name)
                    continue
                
                # Merge on KEY (should be perfect 1:1 within same repeat structure)
                merge_keys = [c for c in ['KEY', 'PARENT_KEY', 'repeat', 'repeat_1', 'repeat_2'] 
                             if c in merged_df.columns and c in reshaped_df.columns]
                
                if not merge_keys:
                    with progress_expander:
                        st.write(f"      ✗ No common merge keys with {base_name}")
                    skipped_in_group.append(base_name)
                    continue
                
                left_multi = _has_multiple_matches(merged_df, merge_keys)
                right_multi = _has_multiple_matches(reshaped_df, merge_keys)
                
                if left_multi and right_multi:
                    with progress_expander:
                        st.write("      ⚠️ Skipping merge: would create many-to-many matches on "
                                 f"{merge_keys}. Prefer SurveyCTO-style 1:m or m:1 joins.")
                    skipped_in_group.append(f"{base_name} (m:m risk)")
                    continue
                
                try:
                    before_rows = len(merged_df)
                    merged_df = merged_df.merge(
                        reshaped_df[merge_keys + data_cols],
                        on=merge_keys,
                        how='inner',  # Inner join - only keep matching rows
                        suffixes=('', '_dup')
                    )
                    after_rows = len(merged_df)
                    merged_var_names.append(base_name)
                    with progress_expander:
                        st.write(f"      ✓ Merged {base_name} ({before_rows} → {after_rows} rows, +{len(data_cols)} cols)")
                except Exception as e:
                    with progress_expander:
                        st.write(f"      ✗ Failed to merge {base_name}: {str(e)}")
                    skipped_in_group.append(base_name)
                    continue
        
        if merged_df is not None and not merged_df.empty:
            # Create intelligent dataset name
            dataset_name = f"{repeat_group_name}"
            if dataset_name in used_dataset_names:
                suffix = 2
                while f"{dataset_name}_{suffix}" in used_dataset_names:
                    suffix += 1
                dataset_name = f"{dataset_name}_{suffix}"
            used_dataset_names.add(dataset_name)
            
            final_datasets[dataset_name] = merged_df
            
            with progress_expander:
                st.write(f"\n  ✅ **{dataset_name}**: {len(merged_df):,} rows × {len(merged_df.columns)} cols")
                st.write(f"      Merged variables: {', '.join(merged_var_names)}")
                if skipped_in_group:
                    st.write(f"      Skipped: {', '.join(skipped_in_group)}")
    
    # Show skipped patterns
    if patterns_skipped:
        with st.expander(f"⚠️ Skipped {len(patterns_skipped)} pattern(s)", expanded=False):
            st.warning("These patterns were not reshaped:")
            for skipped in patterns_skipped:
                st.text(f"• {skipped}")
    
    if not final_datasets:
        st.error("❌ No patterns could be reshaped. All patterns failed.")
        st.info("Please check your data or select different patterns.")
        return {}
    
    # Show final summary
    total_rows = sum(len(df) for df in final_datasets.values())
    with progress_expander:
        st.write("\n" + "="*60)
        st.write("**📊 Reshaping Summary:**")
        st.write(f"   - Variable patterns detected: {len(reshaped_patterns)}")
        st.write(f"   - Repeat structures identified: {len(repeat_structure_groups)}")
        st.write(f"   - Final merged datasets: {len(final_datasets)}")
        st.write(f"   - Total rows: {total_rows:,}")
        st.write("\n💡 Variables from the same repeat group (same row count) are intelligently merged together")
    
    st.success(f"✅ Reshaping complete! {len(final_datasets)} dataset(s) created with {total_rows:,} total rows")
    st.caption("💡 Each dataset represents ONE repeat group with all its variables merged (e.g., firstn, lastn, age together)")
    
    return final_datasets


def generate_stata_reshape_dofile(
    pattern_cols: Dict[str, List[str]],
    reshaped_datasets: Dict[str, pd.DataFrame],
    id_col: str = "KEY"
) -> str:
    """
    Generate a Stata do-file that replicates the Python reshaping operations.
    
    Args:
        pattern_cols: Dictionary of detected patterns from detect_wide_patterns()
        reshaped_datasets: Dictionary of final reshaped datasets
        id_col: ID column used for reshaping (default: "KEY")
        
    Returns:
        String containing complete Stata do-file code
    """
    import re
    from datetime import datetime
    
    dofile = []
    
    # Header
    dofile.append("*" + "="*78)
    dofile.append("* Wide-to-Long Reshape Do-File")
    dofile.append(f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dofile.append("* This file replicates the Python reshaping operations in Stata")
    dofile.append("*" + "="*78)
    dofile.append("")
    dofile.append("clear all")
    dofile.append("set more off")
    dofile.append("")
    
    # Instructions
    dofile.append("* INSTRUCTIONS:")
    dofile.append("* 1. Load your wide-format dataset before running this code")
    dofile.append("* 2. Adjust file paths as needed")
    dofile.append("* 3. Each section reshapes one repeat group")
    dofile.append("")
    dofile.append("*" + "-"*78)
    dofile.append("* Save original wide-format data")
    dofile.append("*" + "-"*78)
    dofile.append("")
    dofile.append('tempfile wide_original')
    dofile.append('save `wide_original\'')
    dofile.append("")
    
    # Group patterns by their structure (similar to Python logic)
    pattern_groups = {}
    for pattern, cols in pattern_cols.items():
        # Extract base name
        base_name = pattern.replace('_*', '').replace('*', '')
        
        # Determine pattern structure
        if cols:
            first_col = cols[0]
            numbers = re.findall(r'_(\d+)', first_col)
            
            if numbers:
                if len(numbers) == 1:
                    pattern_type = "single_level"
                elif len(numbers) == 2:
                    pattern_type = "nested"
                else:
                    pattern_type = f"{len(numbers)}_level"
                
                # Try to identify repeat group
                repeat_group = base_name
                for dataset_name in reshaped_datasets.keys():
                    if base_name in dataset_name:
                        repeat_group = dataset_name
                        break
                
                if repeat_group not in pattern_groups:
                    pattern_groups[repeat_group] = []
                
                pattern_groups[repeat_group].append({
                    'pattern': pattern,
                    'base_name': base_name,
                    'cols': cols,
                    'pattern_type': pattern_type,
                    'numbers': numbers
                })
    
    # Generate reshape code for each group
    for group_idx, (repeat_group, patterns) in enumerate(pattern_groups.items(), 1):
        dofile.append("")
        dofile.append("*" + "="*78)
        dofile.append(f"* Repeat Group {group_idx}: {repeat_group}")
        dofile.append("*" + "="*78)
        dofile.append("")
        
        # Restore original data for each reshape
        dofile.append("* Restore original wide data")
        dofile.append('use `wide_original\', clear')
        dofile.append("")
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            base_name = pattern_info['base_name']
            cols = pattern_info['cols']
            pattern_type = pattern_info['pattern_type']
            
            dofile.append(f"* Reshape pattern: {pattern}")
            dofile.append(f"* Variable: {base_name}")
            dofile.append(f"* Columns: {len(cols)}")
            dofile.append("")
            
            if pattern_type == "single_level":
                # Single-level reshape: var_1, var_2, var_3, ...
                # Stata command: reshape long var_, i(KEY) j(repeat)
                dofile.append(f"* Single-level reshape")
                dofile.append(f"reshape long {base_name}_, i({id_col}) j(repeat)")
                dofile.append(f"rename {base_name}_ {base_name}")
                dofile.append("")
                
                # Drop missing observations
                dofile.append("* Drop rows where all data values are missing")
                dofile.append(f"drop if missing({base_name})")
                dofile.append("")
                
            elif pattern_type == "nested":
                # Nested reshape: var_1_1, var_1_2, var_2_1, var_2_2, ...
                # Requires two reshape commands
                dofile.append(f"* Nested reshape (2 levels)")
                dofile.append(f"* First reshape: outer level")
                dofile.append(f"reshape long {base_name}_@_, i({id_col}) j(repeat_1)")
                dofile.append("")
                dofile.append(f"* Second reshape: inner level")
                dofile.append(f"reshape long {base_name}_, i({id_col} repeat_1) j(repeat_2)")
                dofile.append(f"rename {base_name}_ {base_name}")
                dofile.append("")
                
                # Drop missing observations
                dofile.append("* Drop rows where all data values are missing")
                dofile.append(f"drop if missing({base_name})")
                dofile.append("")
            
            # Save individual pattern dataset
            dofile.append(f"* Save reshaped data for {base_name}")
            dofile.append(f'tempfile temp_{base_name}')
            dofile.append(f'save `temp_{base_name}\'')
            dofile.append("")
            
            # Restore for next pattern in group
            dofile.append('use `wide_original\', clear')
            dofile.append("")
        
        # Merge all patterns in the group
        if len(patterns) > 1:
            dofile.append("*" + "-"*78)
            dofile.append(f"* Merge all variables in {repeat_group}")
            dofile.append("*" + "-"*78)
            dofile.append("")
            
            # Start with first pattern
            first_pattern = patterns[0]
            dofile.append(f"* Start with {first_pattern['base_name']}")
            dofile.append(f"use `temp_{first_pattern['base_name']}\', clear")
            dofile.append("")
            
            # Merge remaining patterns
            for pattern_info in patterns[1:]:
                base_name = pattern_info['base_name']
                pattern_type = pattern_info['pattern_type']
                
                if pattern_type == "single_level":
                    merge_vars = f"{id_col} repeat"
                elif pattern_type == "nested":
                    merge_vars = f"{id_col} repeat_1 repeat_2"
                else:
                    merge_vars = f"{id_col}"
                
                dofile.append(f"* Merge {base_name}")
                dofile.append(f"merge 1:1 {merge_vars} using `temp_{base_name}\', nogenerate")
                dofile.append("")
        
        # Save final merged dataset
        clean_name = repeat_group.replace(':', '_').replace(' ', '_').replace(',', '')
        dofile.append(f"* Save final merged dataset: {repeat_group}")
        dofile.append(f'save "{clean_name}_long.dta", replace')
        dofile.append("")
        dofile.append(f"* Export to CSV")
        dofile.append(f'export delimited using "{clean_name}_long.csv", replace')
        dofile.append("")
    
    # Footer
    dofile.append("")
    dofile.append("*" + "="*78)
    dofile.append("* End of reshape do-file")
    dofile.append("*" + "="*78)
    dofile.append("")
    dofile.append(f"* Generated {len(pattern_groups)} repeat group dataset(s)")
    dofile.append("* Review output files to verify reshape operations")
    
    return "\n".join(dofile)


def merge_datasets_on_key(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Intelligently merge datasets based on 1:1 KEY matching.
    Similar to Stata's merge with _merge==3 (perfect matches only).
    
    Strategy:
    1. Loop through all datasets and try 1:1 merge on KEY
    2. Keep only perfectly matched rows (_merge==3 equivalent)
    3. Save merged datasets separately
    4. For unmatched rows, try merging with other datasets iteratively
    5. Retain unmatched datasets as separate dataframes
    
    Args:
        datasets: Dict mapping pattern names to DataFrames (all must have 'KEY' column)
        
    Returns:
        Dictionary with merged and unmerged datasets
    """
    if not datasets or len(datasets) < 2:
        st.info("📊 Only one dataset available - no merging needed")
        return datasets
    
    st.write("### 🔗 Intelligent KEY-based Merging")
    st.info("ℹ️ Attempting to merge datasets with matching KEY values (1:1 merge, perfect matches only)")
    
    merge_expander = st.expander("📋 Merge Progress", expanded=True)
    
    # Convert dict to list of (name, df) tuples for easier manipulation
    remaining_datasets = [(name, df.copy()) for name, df in datasets.items()]
    merged_datasets = {}
    merge_counter = 1
    
    # Track merging history
    with merge_expander:
        st.write("**Starting merge process...**")
    
    # Iteratively try to merge datasets
    while len(remaining_datasets) > 1:
        merged_in_iteration = False
        
        # Try to merge first dataset with each other dataset
        base_name, base_df = remaining_datasets[0]
        
        for i in range(1, len(remaining_datasets)):
            other_name, other_df = remaining_datasets[i]
            
            # Check if both have KEY column
            if 'KEY' not in base_df.columns or 'KEY' not in other_df.columns:
                continue

            merge_keys = ['KEY']
            base_multi = base_df.duplicated(subset=merge_keys).any()
            other_multi = other_df.duplicated(subset=merge_keys).any()
            if base_multi and other_multi:
                with merge_expander:
                    st.write(f"⚠️ Skipping merge {base_name} + {other_name}: "
                             "both datasets have multiple rows per KEY (would be m:m).")
                continue
            
            # Perform 1:1 merge on KEY
            merged = base_df.merge(
                other_df,
                on='KEY',
                how='inner',  # Inner join = only perfect matches
                suffixes=('', '_OTHER'),
                indicator=True
            )
            
            # Check if we got any matches
            if len(merged) > 0:
                # Handle PARENT_KEY columns (should be the same, so drop duplicate)
                if 'PARENT_KEY_OTHER' in merged.columns:
                    merged = merged.drop(columns=['PARENT_KEY_OTHER'])
                
                # Handle repeat columns
                repeat_cols = [c for c in merged.columns if c.startswith('repeat') and c.endswith('_OTHER')]
                for col in repeat_cols:
                    base_col = col.replace('_OTHER', '')
                    if base_col in merged.columns:
                        merged = merged.drop(columns=[col])
                
                # Drop the merge indicator
                merged = merged.drop(columns=['_merge'])
                
                # Calculate merge statistics
                base_unmatched = len(base_df) - len(merged)
                other_unmatched = len(other_df) - len(merged)
                match_rate = (len(merged) / min(len(base_df), len(other_df))) * 100
                
                with merge_expander:
                    st.write(f"\n✅ **Merge #{merge_counter}**: `{base_name}` + `{other_name}`")
                    st.write(f"   - Matched: {len(merged):,} rows ({match_rate:.1f}% match rate)")
                    st.write(f"   - Unmatched from {base_name}: {base_unmatched:,} rows")
                    st.write(f"   - Unmatched from {other_name}: {other_unmatched:,} rows")
                
                # Create merged dataset name
                merged_name = f"🔗 Merged_{merge_counter}: {base_name} + {other_name}"
                merged_datasets[merged_name] = merged
                merge_counter += 1
                
                # Extract unmatched rows for future merging
                base_unmatched_df = base_df[~base_df['KEY'].isin(merged['KEY'])].copy()
                other_unmatched_df = other_df[~other_df['KEY'].isin(merged['KEY'])].copy()
                
                # Remove the two merged datasets from remaining
                remaining_datasets.pop(i)  # Remove other dataset
                remaining_datasets.pop(0)  # Remove base dataset
                
                # Add unmatched portions back if they have rows
                if len(base_unmatched_df) > 0:
                    remaining_datasets.append((f"{base_name} (unmatched)", base_unmatched_df))
                if len(other_unmatched_df) > 0:
                    remaining_datasets.append((f"{other_name} (unmatched)", other_unmatched_df))
                
                merged_in_iteration = True
                break  # Start over with new remaining list
        
        # If no merges happened in this iteration, we're done
        if not merged_in_iteration:
            # Move the first dataset to final results as unmerged
            name, df = remaining_datasets.pop(0)
            merged_datasets[f"📄 {name}"] = df
            with merge_expander:
                st.write(f"\n📄 **{name}**: No merge partners found - keeping separate ({len(df):,} rows)")
    
    # Add any remaining single dataset
    if remaining_datasets:
        name, df = remaining_datasets[0]
        merged_datasets[f"📄 {name}"] = df
        with merge_expander:
            st.write(f"\n📄 **{name}**: Last dataset - keeping separate ({len(df):,} rows)")
    
    # Show final summary
    with merge_expander:
        st.write("\n" + "="*60)
        st.write("**📊 Merge Summary:**")
        st.write(f"   - Original datasets: {len(datasets)}")
        st.write(f"   - Final datasets: {len(merged_datasets)}")
        st.write(f"   - Merged datasets: {merge_counter - 1}")
        st.write(f"   - Unmerged datasets: {len([k for k in merged_datasets.keys() if k.startswith('📄')])}")
    
    total_rows = sum(len(df) for df in merged_datasets.values())
    st.success(f"✅ Merge complete! {len(merged_datasets)} dataset(s) ready for analysis ({total_rows:,} total rows)")
    
    return merged_datasets


def build_reshaped_download_payloads(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Prepare cached download payloads (Excel workbook + zipped CSV files) for reshaped datasets.
    """
    if not datasets:
        return {}

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    def _clean_name(name: str, fallback: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in (name or fallback))
        safe = safe.strip().replace(" ", "_") or fallback
        return safe[:50]

    # Build Excel workbook (one sheet per dataset)
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        for idx, (dataset_name, dataset_df) in enumerate(datasets.items(), start=1):
            sheet_name = dataset_name.replace("*", "").replace(":", "")[:31] or f"dataset_{idx}"
            dataset_df.to_excel(writer, index=False, sheet_name=sheet_name)
    excel_bytes = excel_buffer.getvalue()

    # Build zipped CSV archive
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        for idx, (dataset_name, dataset_df) in enumerate(datasets.items(), start=1):
            base_name = _clean_name(dataset_name, f"dataset_{idx}")
            csv_name = f"{base_name}.csv"
            suffix = 2
            while csv_name in used_names:
                csv_name = f"{base_name}_{suffix}.csv"
                suffix += 1
            used_names.add(csv_name)
            archive.writestr(csv_name, dataset_df.to_csv(index=False).encode("utf-8"))
    zip_bytes = zip_buffer.getvalue()

    return {
        "excel_bytes": excel_bytes,
        "excel_filename": f"long_format_datasets_{timestamp}.xlsx",
        "zip_bytes": zip_bytes,
        "zip_filename": f"long_format_datasets_{timestamp}.zip",
        "generated_at": timestamp,
    }


def clean_and_convert_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert data types:
    - Convert numeric strings to numbers
    - Handle SurveyCTO value labels (keep labels as categorical)
    - Handle SurveyCTO date/time fields
    - Handle select_multiple space-separated values
    - Strip whitespace from strings
    """
    df_clean = df.copy()
    
    # Convert SurveyCTO SubmissionDate to datetime if present
    if 'SubmissionDate' in df_clean.columns:
        try:
            df_clean['SubmissionDate'] = pd.to_datetime(df_clean['SubmissionDate'], errors='coerce')
        except:
            pass
    
    for col in df_clean.columns:
        # Skip already processed datetime columns
        if df_clean[col].dtype == 'datetime64[ns]':
            continue
            
        # Handle geopoint numeric columns (Latitude, Longitude, Altitude, Accuracy)
        if col.endswith(('-Latitude', '-Longitude', '-Altitude', '-Accuracy')):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            continue
        
        if df_clean[col].dtype == 'object':
            # Strip whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # Handle select_multiple dummy variables (should be 0/1)
            # These typically have column names like fieldname_1, fieldname_2, etc.
            # and contain only 0, 1, or empty values
            non_null_vals = df_clean[col].replace(['', 'nan', 'None', 'NaN', 'na'], pd.NA).dropna()
            if len(non_null_vals) > 0:
                unique_vals = set(non_null_vals.unique())
                if unique_vals.issubset({'0', '1', '0.0', '1.0'}):
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
                    continue
            
            # Try to convert to numeric if all non-null values are numeric
            # This handles cases like "1", "2", "3.5" stored as strings
            # But preserves value labels (text responses)
            try:
                # Check if values look numeric (ignoring NaN/empty)
                non_null = df_clean[col].replace(['', 'nan', 'None', 'NaN', 'na'], pd.NA).dropna()
                if len(non_null) > 0:
                    # Try conversion
                    converted = pd.to_numeric(non_null, errors='coerce')
                    # If more than 80% successfully converted, treat as numeric
                    if converted.notna().sum() / len(non_null) > 0.8:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except:
                pass
    
    return df_clean


def classify_variables(df: pd.DataFrame, max_unique_for_categorical: int = 20, special_values: List[float] = None) -> Dict[str, List[str]]:
    """
    Classify variables as numeric or categorical.
    
    Args:
        df: DataFrame to classify
        max_unique_for_categorical: Max unique values to consider a numeric var as categorical
        special_values: List of values to treat as missing (-999, -888, -666, etc.)
        
    Returns:
        Dict with 'numeric', 'categorical', and 'special_values' keys
    """
    if special_values is None:
        special_values = [-999, -888, -666]
    
    numeric_vars = []
    categorical_vars = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's really categorical (like status codes 1,2,3)
            if df[col].nunique() <= max_unique_for_categorical:
                categorical_vars.append(col)
            else:
                numeric_vars.append(col)
        else:
            categorical_vars.append(col)
    
    return {
        'numeric': numeric_vars,
        'categorical': categorical_vars,
        'special_values': special_values
    }


def generate_numeric_summary(df: pd.DataFrame, columns: List[str], special_values: List[float] = None) -> pd.DataFrame:
    """
    Generate summary statistics for numeric variables.
    Excludes special values (-999, -888, -666) from calculations.
    """
    if not columns:
        return pd.DataFrame()
    
    if special_values is None:
        special_values = [-999, -888, -666]
    
    stats = []
    for col in columns:
        if col not in df.columns:
            continue
        
        series = df[col]
        # Count special values
        special_count = series.isin(special_values).sum() if special_values else 0
        # Exclude special values from calculations
        series_clean = series[~series.isin(special_values)] if special_values else series
        
        stats.append({
            'Variable': col,
            'Count': series_clean.notna().sum(),
            'Missing': series.isna().sum(),
            'Special': special_count,
            'Mean': series_clean.mean(),
            'Std': series_clean.std(),
            'Min': series_clean.min(),
            'Q25': series_clean.quantile(0.25),
            'Median': series_clean.median(),
            'Q75': series_clean.quantile(0.75),
            'Max': series_clean.max(),
        })
    
    return pd.DataFrame(stats)


def generate_categorical_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Generate frequency table for a categorical variable."""
    if column not in df.columns:
        return pd.DataFrame()
    
    freq = df[column].value_counts().reset_index()
    freq.columns = ['Value', 'Count']
    freq['Percentage'] = (freq['Count'] / len(df) * 100).round(2)
    freq = freq.sort_values('Count', ascending=False)
    
    # Add missing count if any
    missing_count = df[column].isna().sum()
    if missing_count > 0:
        missing_row = pd.DataFrame({
            'Value': ['(Missing)'],
            'Count': [missing_count],
            'Percentage': [(missing_count / len(df) * 100).round(2)]
        })
        freq = pd.concat([freq, missing_row], ignore_index=True)
    
    return freq


def render_data_visualization() -> None:
    """
    Render the Data Visualization page.
    
    Features:
    - Multiple data sources (CSV upload, SurveyCTO API, System Config)
    - Automatic unique ID detection
    - Duplicate detection and removal
    - Wide-to-long reshaping with pattern detection
    - Variable classification (numeric/categorical)
    - Interactive filtering and selection
    - Summary statistics and visualizations
    - Cross-tabulation
    - Downloadable reports
    """
    st.title("📊 Data Visualization & Exploration")
    st.markdown("""
    Explore your data with automatic wide-to-long reshaping, summary statistics, 
    and interactive visualizations for both numeric and categorical variables.
    """)
    
    # Initialize session state for visualization
    if "viz_data" not in st.session_state:
        st.session_state.viz_data = None
    if "viz_data_reshaped" not in st.session_state:
        st.session_state.viz_data_reshaped = None  # Dict of pattern_name -> long-format DataFrame
    if "viz_download_cache" not in st.session_state:
        st.session_state.viz_download_cache = None  # Cached download payloads for reshaped data
    if "viz_id_column" not in st.session_state:
        st.session_state.viz_id_column = None
    
    # ------------------------------------------------------------------------ #
    # Data Source Selection
    # ------------------------------------------------------------------------ #
    
    st.markdown("---")
    st.markdown("### 📁 Data Source")
    
    cfg = load_default_config()
    
    source = st.radio(
        "Load data from",
        ["Upload CSV", "SurveyCTO API", "Use project config"],
        key="viz_data_source",
        horizontal=True
    )
    
    data: pd.DataFrame | None = None
    
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            key="viz_csv_upload"
        )
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file, low_memory=False)
                # Clean and convert data types
                data = clean_and_convert_data(data)
                st.success(f"✓ Loaded {len(data):,} rows and {len(data.columns)} columns")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    
    elif source == "SurveyCTO API":
        with st.form("surveycto_viz_form", clear_on_submit=False):
            server = st.text_input(
                "Server URL", 
                value=cfg.get("surveycto", {}).get("server", ""),
                placeholder="yourserver.surveycto.com",
                help="Your SurveyCTO server domain (e.g., myorg.surveycto.com)"
            )
            username = st.text_input(
                "Username", 
                value=cfg.get("surveycto", {}).get("username", ""),
                help="Your SurveyCTO account username"
            )
            password = st.text_input(
                "Password", 
                type="password",
                help="Your SurveyCTO account password"
            )
            form_id = st.text_input(
                "Form ID", 
                value=cfg.get("surveycto", {}).get("form_id", ""),
                placeholder="my_form_name",
                help="Exact form ID (case-sensitive, use underscores not spaces). Find in Form Design > Form IDs."
            )
            
            st.caption("💡 **Tip**: Form IDs are case-sensitive. Use exact form ID from SurveyCTO (e.g., 'ADB_Questionnaire_test', not 'ADB Questionnaire Test')")
            
            submitted = st.form_submit_button("📥 Fetch Data")
            
            if submitted:
                if not all([server, username, password, form_id]):
                    st.error("All fields are required")
                else:
                    try:
                        with st.spinner("Fetching data from SurveyCTO..."):
                            scto = SurveyCTO(server, username, password)
                            data = scto.get_submissions(form_id)
                            # Clean and convert data types
                            data = clean_and_convert_data(data)
                            st.session_state.viz_data = data.copy()
                            st.success(f"✓ Fetched {len(data):,} submissions")
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
    
    else:  # Use project config
        monitor_cfg = cfg.get("monitoring", {})
        if monitor_cfg.get("server") and monitor_cfg.get("form_id"):
            try:
                with st.spinner("Loading data from project config..."):
                    scto = SurveyCTO(
                        monitor_cfg["server"],
                        monitor_cfg["username"],
                        monitor_cfg.get("password", "")
                    )
                    data = scto.get_submissions(monitor_cfg["form_id"])
                    # Clean and convert data types
                    data = clean_and_convert_data(data)
                    st.session_state.viz_data = data.copy()
                    st.success(f"✓ Loaded {len(data):,} submissions from config")
            except Exception as e:
                st.error(f"Error loading from config: {e}")
                st.info("Configure monitoring settings in config/default.yaml or use another data source")
        else:
            st.info("No monitoring configuration found. Please upload CSV or use SurveyCTO API.")
    
    # Use data from session state if available
    if st.session_state.viz_data is not None:
        data = st.session_state.viz_data
    
    if data is None or data.empty:
        st.info("👆 Please load data to begin exploration")
        return
    
    # ------------------------------------------------------------------------ #
    # Data Preview & Validation
    # ------------------------------------------------------------------------ #
    
    st.markdown("---")
    st.markdown("### 🔍 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(data):,}")
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        st.metric("Memory", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (data.isna().sum().sum() / (len(data) * len(data.columns)) * 100)
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(data.head(20), use_container_width=True)
    
    # SurveyCTO-specific info
    surveycto_cols = [c for c in ['KEY', 'SubmissionDate', 'formdef_version'] if c in data.columns]
    if surveycto_cols:
        with st.expander("ℹ️ SurveyCTO Data Detected", expanded=False):
            st.markdown("""
            This appears to be SurveyCTO data. Special features detected:
            """)
            features = []
            if 'KEY' in data.columns:
                features.append(f"- **KEY column**: {data['KEY'].nunique():,} unique submissions")
            if 'SubmissionDate' in data.columns:
                features.append("- **SubmissionDate**: Automatically converted to datetime")
            
            # Check for geopoints
            geopoint_fields = [c[:-9] for c in data.columns if c.endswith('-Latitude')]
            if geopoint_fields:
                features.append(f"- **Geopoint fields** ({len(geopoint_fields)}): {', '.join(geopoint_fields)}")
            
            # Check for select_multiple patterns
            select_mult = [c for c in data.columns if '_' in c and c.split('_')[-1].isdigit()]
            if select_mult:
                features.append(f"- **Possible select_multiple dummy variables** ({len(select_mult)} columns)")
            
            # Check for repeat groups (wide format)
            repeat_patterns = [c for c in data.columns if c.endswith(tuple([f'_{i}' for i in range(1, 10)]))]
            if repeat_patterns:
                features.append(f"- **Repeat groups in wide format**: Use reshaping to convert to long format")
            
            for f in features:
                st.markdown(f)
    
    # ------------------------------------------------------------------------ #
    # ID Detection & Duplicate Handling
    # ------------------------------------------------------------------------ #
    
    st.markdown("---")
    st.markdown("### 🔑 ID Column & Duplicate Check")
    
    # Detect potential ID columns
    id_candidates = detect_unique_id(data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if id_candidates:
            id_col = st.selectbox(
                "Select ID column",
                options=id_candidates,
                help="Detected columns with unique or mostly unique values"
            )
        else:
            id_col = st.selectbox(
                "Select ID column",
                options=list(data.columns),
                help="No columns with all unique values detected. Please select manually."
            )
        
        st.session_state.viz_id_column = id_col
    
    with col2:
        check_duplicates = st.button("🔍 Check for Duplicates", key="viz_check_dupes", use_container_width=True)
    
    if check_duplicates and id_col:
        duplicates = detect_duplicates(data, id_col)
        
        if not duplicates.empty:
            st.warning(f"⚠️ Found {len(duplicates)} duplicate rows based on '{id_col}'")
            
            with st.expander("View Duplicates", expanded=True):
                st.dataframe(duplicates, use_container_width=True)
            
            if st.button("🗑️ Remove Duplicates (keep first occurrence)", key="viz_remove_dupes"):
                data = data.drop_duplicates(subset=[id_col], keep='first')
                st.session_state.viz_data = data.copy()
                st.success(f"✓ Removed duplicates. {len(data):,} rows remaining.")
                st.rerun()
        else:
            st.success(f"✓ No duplicates found in '{id_col}'")
    
    # ------------------------------------------------------------------------ #
    # Wide-to-Long Reshaping
    # ------------------------------------------------------------------------ #
    
    st.markdown("---")
    st.markdown("### 🔄 Data Reshaping (Wide to Long)")
    
    st.markdown("""
    Automatically detect and reshape repeated patterns in your data:
    - `var_1`, `var_2`, ... → Long format with `var` and `repeat` columns
    - `var1`, `var2`, ... → Same transformation
    - `var_1_1`, `var_1_2`, ... → Long with `var`, `repeat_1`, `repeat_2`
    """)
    
    enable_reshape = st.checkbox("Enable automatic reshaping", value=False, key="enable_reshape")
    
    if enable_reshape and id_col:
        with st.spinner("Detecting repeated patterns..."):
            patterns = detect_wide_patterns(data)
        
        if patterns:
            st.success(f"✓ Detected {len(patterns)} repeated pattern(s)")
            
            # Special value handling configuration
            st.write("---")
            st.write("#### ⚙️ Special Value Handling")
            st.caption("Specify values that represent special responses (will be treated as missing in numeric analysis)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                dont_know_val = st.text_input(
                    "Don't Know",
                    value="-999",
                    key="viz_dont_know_value",
                    help="Value representing 'Don't know' responses"
                )
            with col2:
                refuse_val = st.text_input(
                    "Refuse to Answer",
                    value="-888",
                    key="viz_refuse_value",
                    help="Value representing 'Refuse to answer' responses"
                )
            with col3:
                other_val = st.text_input(
                    "Other (Specify)",
                    value="-666",
                    key="viz_other_value",
                    help="Value representing 'Other specify' responses"
                )
            
            # Parse special values
            special_values = []
            for val_str in [dont_know_val, refuse_val, other_val]:
                if val_str.strip():
                    try:
                        special_values.append(float(val_str.strip()))
                    except ValueError:
                        pass
            
            if special_values:
                st.info(f"ℹ️ Special values {special_values} will be treated as missing in numeric analysis")
            
            st.session_state.viz_special_values = special_values
            
            st.write("---")
            
            with st.expander("🔍 Detected Patterns", expanded=False):
                for pattern, cols in patterns.items():
                    st.markdown(f"**Pattern:** `{pattern}`")
                    st.write(f"Columns ({len(cols)}): {', '.join(cols[:10])}" + 
                            (f" ... and {len(cols)-10} more" if len(cols) > 10 else ""))
                    st.markdown("---")
            
            selected_patterns = st.multiselect(
                "Select patterns to reshape",
                options=list(patterns.keys()),
                default=list(patterns.keys()),
                help="Choose which patterns should be converted to long format",
                key="viz_select_patterns"
            )
            
            # Additional merge option for cross-repeat merging
            st.write("---")
            st.write("#### 🔗 Cross-Repeat Group Merging")
            st.caption("Variables from the same repeat structure are automatically merged. This option merges across different repeat structures.")
            
            cross_merge = st.checkbox(
                "Attempt to merge different repeat groups (if they share KEY values)",
                value=False,
                key="viz_cross_merge",
                help="Try to merge datasets from different repeat structures if they have matching KEY values. Use with caution."
            )
            
            if st.button("▶️ Apply Reshaping", type="primary", key="viz_apply_reshape"):
                selected_pattern_cols = {k: v for k, v in patterns.items() if k in selected_patterns}
                
                with st.spinner("Reshaping data to long format..."):
                    reshaped_data = reshape_wide_to_long(data, id_col, selected_pattern_cols)
                    
                if reshaped_data:
                    if cross_merge:
                        with st.spinner("Attempting cross-repeat group merging..."):
                            reshaped_data = merge_datasets_on_key(reshaped_data)
                    
                    st.session_state.viz_data_reshaped = reshaped_data
                    st.session_state.viz_download_cache = build_reshaped_download_payloads(reshaped_data)
                    st.success("✅ Reshaping complete. Downloads are cached below.")
                else:
                    st.warning("No patterns could be reshaped. Check warnings above.")

            current_long = st.session_state.get("viz_data_reshaped")
            if isinstance(current_long, dict) and current_long:
                st.write("\n### 📊 Reshaped Datasets Summary")
                
                summary_data = []
                for pattern_name, pattern_df in current_long.items():
                    summary_data.append({
                        'Dataset': pattern_name,
                        'Rows': len(pattern_df),
                        'Columns': len(pattern_df.columns),
                        'Data Columns': len([c for c in pattern_df.columns if c not in ['KEY', 'PARENT_KEY', 'repeat', 'repeat_1', 'repeat_2']])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.write("\n#### 📥 Download Long Format Datasets")
                st.caption("Excel mirrors SurveyCTO exports (one sheet per repeat). CSV option provides a ZIP with one file per dataset.")
                
                download_cache = st.session_state.get("viz_download_cache")
                if (not download_cache) and current_long:
                    download_cache = build_reshaped_download_payloads(current_long)
                    st.session_state.viz_download_cache = download_cache
                
                if download_cache:
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    with dl_col1:
                        st.download_button(
                            label="📊 Download All Datasets (Excel)",
                            data=download_cache["excel_bytes"],
                            file_name=download_cache["excel_filename"],
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Cached workbook – downloading will not clear analysis results.",
                            use_container_width=True,
                            key="viz_download_excel"
                        )
                    with dl_col2:
                        st.download_button(
                            label="📁 Download CSV Bundle (ZIP)",
                            data=download_cache["zip_bytes"],
                            file_name=download_cache["zip_filename"],
                            mime="application/zip",
                            help="ZIP archive with UTF-8 CSV files for each repeat group.",
                            use_container_width=True,
                            key="viz_download_zip"
                        )
                    with dl_col3:
                        # Generate Stata do-file
                        selected_pattern_cols = {k: v for k, v in patterns.items() if k in selected_patterns}
                        stata_dofile = generate_stata_reshape_dofile(
                            selected_pattern_cols,
                            current_long,
                            id_col
                        )
                        st.download_button(
                            label="📜 Download Stata Do-File",
                            data=stata_dofile,
                            file_name=f"reshape_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.do",
                            mime="text/plain",
                            help="Stata code that replicates the reshaping operations",
                            use_container_width=True,
                            key="viz_download_stata"
                        )
                    st.caption("📎 Downloads remain available after clicking. No need to rerun reshaping.")
        else:
            st.info("No repeated patterns detected in column names")
    
    # Multi-dataset analysis - analyze all datasets simultaneously
    st.markdown("---")
    st.markdown("### 📊 Dataset Analysis")
    
    # Check if long format datasets are available
    has_long_format = (st.session_state.viz_data_reshaped is not None and 
                      isinstance(st.session_state.viz_data_reshaped, dict) and 
                      len(st.session_state.viz_data_reshaped) > 0)
    
    # Prepare datasets for analysis
    if has_long_format:
        # Analyze ALL long format datasets by default
        datasets_to_analyze = st.session_state.viz_data_reshaped
        st.info(f"📊 Analyzing **{len(datasets_to_analyze)} long-format dataset(s)** simultaneously (SurveyCTO style - each pattern separate)")
        st.caption("💡 Each dataset shown below with its own summary statistics. Toggle expanders to view details.")
        analyze_mode = "multi"
    else:
        # Only wide format available
        datasets_to_analyze = {"Wide Format (original)": data}
        st.info(f"📋 Analyzing **wide format**: {len(data):,} rows × {len(data.columns)} columns")
        st.caption("💡 Enable reshaping above to create long-format datasets for repeat group analysis")
        analyze_mode = "single"
    
    # ------------------------------------------------------------------------ #
    # Multi-Dataset Analysis
    # ------------------------------------------------------------------------ #
    
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics for All Datasets")
    
    special_vals = st.session_state.get('viz_special_values', [-999, -888, -666])
    
    # Global filter settings (apply to all datasets)
    st.write("#### 🎛️ Global Filters (applies to all datasets)")
    col1, col2 = st.columns(2)
    
    with col1:
        filter_type = st.radio(
            "Filter by type",
            ["All", "Numeric Only", "Categorical Only"],
            horizontal=True,
            key="var_filter_type"
        )
    
    with col2:
        search_term = st.text_input("🔍 Search variables by name", key="var_search", placeholder="Enter variable name...")
    
    # ------------------------------------------------------------------------ #
    # Multi-Dataset Analysis Sections
    # ------------------------------------------------------------------------ #
    
    # Analyze each dataset in expandable sections
    for idx, (dataset_name, working_data) in enumerate(datasets_to_analyze.items()):
        with st.expander(f"📊 **{dataset_name}** ({len(working_data):,} rows × {len(working_data.columns)} cols)", expanded=(idx == 0)):
            
            # Classify variables for this dataset
            var_types = classify_variables(working_data, max_unique_for_categorical=20, special_values=special_vals)
            
            # Apply filters
            if filter_type == "Numeric Only":
                available_vars = var_types['numeric']
            elif filter_type == "Categorical Only":
                available_vars = var_types['categorical']
            else:
                available_vars = list(working_data.columns)
            
            # Apply search filter
            if search_term:
                available_vars = [v for v in available_vars if search_term.lower() in v.lower()]
            
            # Dataset metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Variables", len(available_vars))
            with col2:
                st.metric("Numeric", len([v for v in available_vars if v in var_types['numeric']]))
            with col3:
                st.metric("Categorical", len([v for v in available_vars if v in var_types['categorical']]))
            with col4:
                missing_pct = (working_data.isna().sum().sum() / (len(working_data) * len(working_data.columns)) * 100)
                st.metric("Missing %", f"{missing_pct:.1f}")
            
            # Tabs for different analysis types
            tab1, tab2, tab3 = st.tabs([
                "📈 Numeric Summary",
                "📊 Data Preview",
                "📋 Variable List"
            ])
            
            # TAB 1: Numeric Summary
            with tab1:
                numeric_vars_to_analyze = [v for v in available_vars if v in var_types['numeric']]
                
                if not numeric_vars_to_analyze:
                    st.info("No numeric variables in this dataset")
                else:
                    # Show summary for all numeric variables
                    summary_df = generate_numeric_summary(working_data, numeric_vars_to_analyze, special_vals)
                    
                    if not summary_df.empty:
                        # Format for display
                        format_dict = {
                            'Count': '{:,.0f}',
                            'Missing': '{:,.0f}',
                            'Special': '{:,.0f}',
                            'Mean': '{:,.2f}',
                            'Std': '{:,.2f}',
                            'Min': '{:,.2f}',
                            'Q25': '{:,.2f}',
                            'Median': '{:,.2f}',
                            'Q75': '{:,.2f}',
                            'Max': '{:,.2f}',
                        }
                        
                        st.dataframe(
                            summary_df.style.format(format_dict),
                            use_container_width=True,
                            hide_index=True
                        )
                        if special_vals:
                            st.caption(f"ℹ️ Statistics exclude special values {special_vals}")
                        
                        # Download button
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Summary (CSV)",
                            csv,
                            f"{dataset_name}_summary.csv",
                            "text/csv",
                            key=f"download_summary_{idx}"
                        )
            
            # TAB 2: Data Preview
            with tab2:
                st.dataframe(working_data.head(20), use_container_width=True)
            
            # TAB 3: Variable List
            with tab3:
                var_list = []
                for col in available_vars:
                    var_type = "Numeric" if col in var_types['numeric'] else "Categorical"
                    unique_count = working_data[col].nunique()
                    missing_count = working_data[col].isna().sum()
                    var_list.append({
                        'Variable': col,
                        'Type': var_type,
                        'Unique': unique_count,
                        'Missing': missing_count
                    })
                
                if var_list:
                    var_df = pd.DataFrame(var_list)
                    st.dataframe(var_df, use_container_width=True, hide_index=True)
    
    # Global download section for all datasets
    st.markdown("---")
    st.markdown("### 📥 Download All Summaries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download all numeric summaries combined
        if st.button("📊 Download All Numeric Summaries (Excel)", use_container_width=True):
            from io import BytesIO
            excel_buffer = BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                for dataset_name, working_data in datasets_to_analyze.items():
                    var_types = classify_variables(working_data, max_unique_for_categorical=20, special_values=special_vals)
                    numeric_vars = var_types['numeric']
                    
                    if numeric_vars:
                        summary_df = generate_numeric_summary(working_data, numeric_vars, special_vals)
                        sheet_name = dataset_name.replace('*', '').replace(':', '')[:31]
                        summary_df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            excel_data = excel_buffer.getvalue()
            st.download_button(
                "⬇️ Download",
                excel_data,
                f"all_numeric_summaries_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_all_summaries"
            )
    
    with col2:
        # Info about download
        st.info("💡 Excel file contains one sheet per dataset with all numeric variable summaries")


def render_quality_checks() -> None:
    st.title("✅ Quality Checks")
    st.markdown("Apply duration, duplicate, outlier, and intervention checks to submission data.")

    cfg = load_default_config()
    
    source = st.radio(
        "Data source",
        ["Use project config", "Upload CSV", "SurveyCTO API"],
        key="quality_data_source",
    )

    data: pd.DataFrame | None = None

    if source == "Use project config":
        try:
            from rct_field_flow.monitor import load_submissions as mon_load_submissions
            submissions = mon_load_submissions(cfg)
            data = submissions
            if not data.empty:
                st.success(f"Loaded {len(data):,} submissions from project config.")
        except Exception as exc:
            st.error(f"Couldn't load submissions using project config: {exc}")
            return
    elif source == "Upload CSV":
        upload = st.file_uploader("Upload submissions CSV", type="csv", key="quality_csv_upload")
        if upload:
            data = pd.read_csv(upload)
            st.session_state.quality_data = data
        else:
            data = st.session_state.get("quality_data")
        if data is None:
            st.info("Upload a CSV file to continue.")
            return
    else:  # SurveyCTO API
        col1, col2 = st.columns(2)
        with col1:
            server_default = cfg.get("surveycto", {}).get("server", "")
            server = st.text_input("SurveyCTO server (without https://)", value=server_default, key="quality_api_server")
            username_default = cfg.get("surveycto", {}).get("username", "")
            username = st.text_input("Username", value=username_default, key="quality_api_user")
        with col2:
            password = st.text_input("Password", type="password", key="quality_api_pass")
            form_default = cfg.get("surveycto", {}).get("form_id", "")
            form_id = st.text_input("Form ID", value=form_default, key="quality_api_form")

        if st.button("Fetch SurveyCTO submissions", key="quality_fetch_api"):
            if not all([server, username, password, form_id]):
                st.error("Server, username, password, and form ID are required.")
            else:
                try:
                    client = SurveyCTO(server=server, username=username, password=password)
                    api_df = client.get_submissions(form_id)
                    st.session_state.quality_api_df = api_df
                    st.success(f"Fetched {len(api_df):,} submissions from SurveyCTO.")
                except Exception as exc:
                    st.error(f"Failed to fetch SurveyCTO submissions: {exc}")
        data = st.session_state.get("quality_api_df")
        if data is None:
            st.info("Enter credentials and click the fetch button to load live data.")
            return

    if data is None or data.empty:
        st.warning("No submissions available. Check your data source.")
        return

    df = data
    
    # Show data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Get column lists for interactive configuration (after data is loaded)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Show column information
    with st.expander("ℹ️ Column Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Numeric columns ({len(numeric_cols)}):**")
            if numeric_cols:
                st.write(", ".join(numeric_cols[:10]))
                if len(numeric_cols) > 10:
                    st.write(f"... and {len(numeric_cols) - 10} more")
            else:
                st.warning("No numeric columns found in dataset")
        with col2:
            st.write(f"**All columns ({len(all_cols)}):**")
            if all_cols:
                st.write(", ".join(all_cols[:10]))
                if len(all_cols) > 10:
                    st.write(f"... and {len(all_cols) - 10} more")

    # Configuration mode selector
    config_mode = st.radio(
        "Configuration method",
        ["Interactive (recommended)", "YAML (advanced)"],
        key="config_mode",
        horizontal=True
    )
    
    if config_mode == "YAML (advanced)":
        # Original YAML-based configuration
        default_quality_cfg = load_default_config().get("quality_checks", {})
        config_text = st.text_area(
            "Quality check configuration (YAML)",
            value=yaml_dump(default_quality_cfg),
            height=240,
            key="quality_config_text",
        )

        if st.button("Run quality checks", type="primary"):
            try:
                config = yaml_load(config_text)
                results: QualityResults = flag_all(df, config)
            except Exception as exc:
                st.error(f"Quality checks failed: {exc}")
                return

            st.success("Quality checks completed.")
            st.markdown("#### Flag counts")
            st.write(results.flag_counts)

            st.markdown("#### Enumerator summary")
            st.dataframe(results.enumerator_summary, use_container_width=True)

            st.markdown("#### Flagged submissions (first 200)")
            display_flags = pd.concat([df, results.flags], axis=1)
            flagged_rows = display_flags[results.flags.any(axis=1)].head(200)
            st.dataframe(flagged_rows, use_container_width=True)

            csv_buffer = io.StringIO()
            display_flags.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download flagged dataset CSV",
                data=csv_buffer.getvalue(),
                file_name="quality_checks_output.csv",
                mime="text/csv",
            )
    
    else:  # Interactive mode
        st.markdown("### Configure Quality Checks")
        
        # Configuration tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            " Outlier Detection",
            "⏱️ Duration Checks",
            "👥 Duplicate Detection",
            "✅ Intervention Fidelity"
        ])
        
        # Tab 1: Outlier Detection
        with tab1:
            st.markdown("**Detect outliers in numeric variables using IQR or standard deviation methods**")
            
            # Reshape option
            with st.expander("🔄 Reshape repeated measures (Wide to Long)", expanded=False):
                st.info("📋 **Example:** `icm_hr_worked_7d_1`, `icm_hr_worked_7d_2`, `icm_hr_worked_7d_3` → `icm_hr_worked_7d`")
                
                enable_reshape = st.checkbox(
                    "Enable data reshaping before outlier detection",
                    value=False,
                    help="Reshape wide data with repeated measures into long format",
                    key="enable_reshape"
                )
                
                if enable_reshape:
                    st.markdown("**Identify variable patterns to reshape**")
                    
                    # Auto-detect potential reshape patterns
                    import re
                    potential_patterns = {}
                    for col in all_cols:
                        # Look for patterns like var_1, var_2, var_3 or var_a, var_b, var_c
                        match = re.match(r'^(.+?)(_\d+|_[a-z])$', col)
                        if match:
                            base = match.group(1)
                            if base not in potential_patterns:
                                potential_patterns[base] = []
                            potential_patterns[base].append(col)
                    
                    # Filter to patterns with multiple columns
                    potential_patterns = {k: v for k, v in potential_patterns.items() if len(v) > 1}
                    
                    if potential_patterns:
                        st.success(f"🔍 Found {len(potential_patterns)} potential reshape patterns")
                        
                        # Show detected patterns in an expander
                        with st.expander("View detected patterns", expanded=False):
                            for base, cols in list(potential_patterns.items())[:10]:
                                st.write(f"**{base}**: {', '.join(cols[:5])}" + (f" ... (+{len(cols)-5} more)" if len(cols) > 5 else ""))
                        
                        # Let user select which patterns to reshape
                        selected_patterns = st.multiselect(
                            "Select variable patterns to reshape",
                            options=list(potential_patterns.keys()),
                            help="Choose base variable names to reshape from wide to long",
                            key="reshape_patterns"
                        )
                    else:
                        st.warning("No reshape patterns automatically detected. Enter pattern manually below.")
                        selected_patterns = []
                    
                    # Manual pattern entry
                    st.markdown("**Or enter pattern manually:**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        manual_pattern = st.text_input(
                            "Variable pattern (use * as wildcard)",
                            placeholder="e.g., icm_hr_worked_7d_*",
                            help="Enter the pattern for columns to reshape. Use * for the varying part.",
                            key="manual_pattern"
                        )
                    with col2:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                        add_manual = st.button("Add pattern", key="add_manual_pattern")
                    
                    if manual_pattern and add_manual:
                        # Convert wildcard to base name
                        base_name = manual_pattern.replace("_*", "").replace("*", "")
                        if base_name not in selected_patterns:
                            selected_patterns.append(base_name)
                            st.success(f"✅ Added pattern: {base_name}")
                    
                    # Key columns for reshaping
                    if selected_patterns:
                        st.markdown("**Specify ID columns (preserved during reshape)**")
                        id_cols = st.multiselect(
                            "ID columns",
                            all_cols,
                            default=[col for col in ['caseid', 'id', 'respondent_id', 'KEY'] if col in all_cols],
                            help="Columns that uniquely identify each observation (e.g., caseid, enumerator)",
                            key="reshape_id_cols"
                        )
                        
                        if id_cols:
                            if st.button("🔄 Apply Reshape", type="primary", key="apply_reshape_btn"):
                                with st.spinner("Reshaping data..."):
                                    try:
                                        reshaped_dfs = []
                                        
                                        for pattern in selected_patterns:
                                            # Find all columns matching this pattern
                                            matching_cols = [col for col in all_cols if col.startswith(pattern + "_")]
                                            
                                            if len(matching_cols) > 1:
                                                # Get columns to keep (id columns + matching columns)
                                                cols_to_keep = id_cols + matching_cols
                                                subset = df[cols_to_keep].copy()
                                                
                                                # Reshape using melt to convert wide to long
                                                melted = subset.melt(
                                                    id_vars=id_cols,
                                                    value_vars=matching_cols,
                                                    var_name=f'{pattern}_index',
                                                    value_name=pattern
                                                )
                                                
                                                # Drop rows with missing values
                                                melted = melted.dropna(subset=[pattern])
                                                
                                                reshaped_dfs.append(melted)
                                                
                                                st.success(f"✅ Reshaped {len(matching_cols)} columns for '{pattern}' → {len(melted)} observations")
                                        
                                        if reshaped_dfs:
                                            # Merge all reshaped dataframes
                                            if len(reshaped_dfs) == 1:
                                                reshaped_data = reshaped_dfs[0]
                                            else:
                                                # Merge multiple patterns
                                                reshaped_data = reshaped_dfs[0]
                                                for i in range(1, len(reshaped_dfs)):
                                                    reshaped_data = reshaped_data.merge(
                                                        reshaped_dfs[i],
                                                        on=id_cols,
                                                        how='outer'
                                                    )
                                            
                                            # Store in session state
                                            st.session_state['reshaped_data'] = reshaped_data
                                            st.session_state['reshaped_patterns'] = selected_patterns
                                            
                                            st.success(f"📊 Reshaped dataset: {len(reshaped_data)} rows × {len(reshaped_data.columns)} columns")
                                            
                                            with st.expander("Preview reshaped data", expanded=True):
                                                st.dataframe(reshaped_data.head(20), use_container_width=True)
                                    
                                    except Exception as e:
                                        st.error(f"❌ Error during reshaping: {e}")
                                        if 'reshaped_data' in st.session_state:
                                            del st.session_state['reshaped_data']
                        else:
                            st.warning("⚠️ Please select at least one ID column for reshaping")
            
            st.divider()
            
            # Check if we have reshaped data
            if 'reshaped_data' in st.session_state:
                reshaped_data = st.session_state['reshaped_data']
                reshaped_patterns = st.session_state.get('reshaped_patterns', [])
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"✨ **Using reshaped data** ({len(reshaped_data)} observations). Reshaped variables: {', '.join(reshaped_patterns)}")
                with col2:
                    if st.button("🔄 Reset", help="Clear reshape and use original data", key="clear_reshape_btn"):
                        del st.session_state['reshaped_data']
                        if 'reshaped_patterns' in st.session_state:
                            del st.session_state['reshaped_patterns']
                        st.rerun()
                
                # Get columns from reshaped data
                available_numeric_cols = reshaped_data.select_dtypes(include=['number']).columns.tolist()
                available_all_cols = reshaped_data.columns.tolist()
                
                # Highlight reshaped variables
                reshaped_var_options = []
                other_var_options = []
                for col in (available_all_cols if not available_numeric_cols else available_numeric_cols):
                    if col in reshaped_patterns:
                        reshaped_var_options.append(f"📊 {col} (reshaped)")
                    else:
                        other_var_options.append(col)
                
                # Combine with reshaped vars first
                outlier_var_options = reshaped_var_options + other_var_options
            else:
                # Use original data
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns automatically detected. Select columns manually below.")
                    st.info("💡 **Tip:** Columns may need to be converted to numeric. The tool will attempt automatic conversion.")
                
                outlier_var_options = all_cols if not numeric_cols else numeric_cols
            
            outlier_vars = st.multiselect(
                "Select variables to check for outliers",
                options=outlier_var_options,
                help="Choose variables to analyze. Must contain numeric values.",
                key="outlier_vars"
            )
            
            # Clean up outlier_vars to remove formatting
            outlier_vars = [var.replace("📊 ", "").replace(" (reshaped)", "") for var in outlier_vars]
            
            col1, col2 = st.columns(2)
            
            with col1:
                outlier_method = st.selectbox(
                    "Detection method",
                    ["IQR", "Standard Deviation"],
                    help="IQR: Q1/Q3 ± threshold×IQR | SD: mean ± threshold×SD",
                    key="outlier_method"
                )
            
            with col2:
                if outlier_method == "IQR":
                    outlier_threshold = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1,
                                                 help="Standard: 1.5 (mild outliers)",
                                                 key="outlier_threshold")
                else:
                    outlier_threshold = st.slider("Standard deviations", 1.0, 5.0, 3.0, 0.5,
                                                 help="Standard: 3.0 (99.7% coverage)",
                                                 key="outlier_threshold")
            
            group_by_outlier = st.selectbox(
                "Group analysis by (optional)",
                ["None"] + all_cols,
                help="Detect outliers within groups (e.g., by enumerator)",
                key="group_by_outlier"
            )
        
        # Tab 2: Duration Checks
        with tab2:
            st.markdown("**Flag surveys that are too fast or too slow**")
            
            if not numeric_cols:
                st.warning("⚠️ No numeric columns automatically detected. Select column manually below.")
                st.info("💡 **Tip:** Duration column should contain numeric values (seconds or minutes).")
            
            duration_col = st.selectbox(
                "Duration column",
                ["None"] + (all_cols if not numeric_cols else numeric_cols),
                help="Select survey duration column (must be numeric)",
                key="duration_col"
            )
            
            duration_unit = st.radio("Time unit", ["Seconds", "Minutes"], horizontal=True, key="duration_unit")
            
            check_method = st.radio(
                "Detection method",
                ["Quantile-based", "Absolute thresholds"],
                horizontal=True,
                key="check_method"
            )
            
            if check_method == "Quantile-based":
                col1, col2 = st.columns(2)
                with col1:
                    speed_quantile = st.slider("Flag fastest (%)", 0.0, 50.0, 5.0, 1.0, key="speed_quantile")
                with col2:
                    slow_quantile = st.slider("Flag slowest (%)", 50.0, 100.0, 95.0, 1.0, key="slow_quantile")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    min_duration = st.number_input(f"Minimum ({duration_unit.lower()})",
                                                  value=10 if duration_unit == "Minutes" else 600, 
                                                  min_value=0,
                                                  key="min_duration")
                with col2:
                    max_duration = st.number_input(f"Maximum ({duration_unit.lower()})",
                                                  value=120 if duration_unit == "Minutes" else 7200, 
                                                  min_value=0,
                                                  key="max_duration")
        
        # Tab 3: Duplicate Detection
        with tab3:
            st.markdown("**Identify duplicate submissions**")
            
            duplicate_keys = st.multiselect(
                "Key columns for duplicate detection",
                all_cols,
                help="Submissions with same values are duplicates",
                key="duplicate_keys"
            )
            
            check_gps_dups = st.checkbox("Also check GPS duplicates", value=False, key="check_gps_dups")
            
            if check_gps_dups:
                if not numeric_cols:
                    st.warning("⚠️ GPS columns should be numeric (latitude/longitude coordinates)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    lat_col = st.selectbox("Latitude", ["None"] + (all_cols if not numeric_cols else numeric_cols), key="lat_col")
                with col2:
                    lon_col = st.selectbox("Longitude", ["None"] + (all_cols if not numeric_cols else numeric_cols), key="lon_col")
                with col3:
                    if lat_col != "None" and lon_col != "None":
                        gps_threshold = st.slider("Proximity (m)", 1, 100, 10, key="gps_threshold")
        
        # Tab 4: Intervention Fidelity
        with tab4:
            st.markdown("**Verify treatment assignment and intervention delivery**")
            
            treatment_col = st.selectbox(
                "Treatment/group column",
                ["None"] + all_cols,
                key="treatment_col"
            )
            
            if treatment_col != "None":
                treatment_vals = df[treatment_col].dropna().unique().tolist()
                expected_vals = st.multiselect(
                    "Valid treatment values",
                    treatment_vals,
                    default=treatment_vals,
                    key="expected_vals"
                )
        
        # Run checks button
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            run_checks = st.button("▶️ Run Quality Checks", type="primary", use_container_width=True)
        with col2:
            group_results_by = st.selectbox("Group by", ["None"] + all_cols)
        
        # Run interactive quality checks
        if run_checks:
            # Use reshaped data if available, otherwise use original
            if 'reshaped_data' in st.session_state:
                working_df = st.session_state['reshaped_data'].copy()
                st.info(f"✨ Running checks on reshaped data ({len(working_df)} observations)")
            else:
                working_df = df.copy()
            
            flagged_records = []
            
            # 1. Outlier detection
            if outlier_vars:
                with st.spinner("Detecting outliers..."):
                    for var in outlier_vars:
                        # Check if variable exists in working dataframe
                        if var not in working_df.columns:
                            st.warning(f"⚠️ Column '{var}' not found in dataset. Skipping...")
                            continue
                        
                        # Try to convert to numeric
                        try:
                            working_df[var] = pd.to_numeric(working_df[var], errors='coerce')
                        except Exception:
                            st.warning(f"⚠️ Could not convert '{var}' to numeric. Skipping...")
                            continue
                        
                        # Check if we have numeric values after conversion
                        if not pd.api.types.is_numeric_dtype(working_df[var]):
                            st.warning(f"⚠️ Column '{var}' does not contain numeric values. Skipping...")
                            continue
                        
                        if group_by_outlier != "None":
                            for group_val, group_data in working_df.groupby(group_by_outlier):
                                values = group_data[var].dropna()
                                if len(values) > 3:
                                    if outlier_method == "IQR":
                                        q1, q3 = values.quantile([0.25, 0.75])
                                        iqr = q3 - q1
                                        lower = q1 - outlier_threshold * iqr
                                        upper = q3 + outlier_threshold * iqr
                                    else:
                                        mean, std = values.mean(), values.std()
                                        lower = mean - outlier_threshold * std
                                        upper = mean + outlier_threshold * std
                                    
                                    outliers = group_data[(group_data[var] < lower) | (group_data[var] > upper)]
                                    for idx, row in outliers.iterrows():
                                        flagged_records.append({
                                            'check_type': 'outlier',
                                            'variable': var,
                                            'value': row[var],
                                            'group': f"{group_by_outlier}={group_val}",
                                            'lower_bound': round(lower, 2),
                                            'upper_bound': round(upper, 2),
                                            'record_index': idx
                                        })
                        else:
                            values = working_df[var].dropna()
                            if len(values) > 3:
                                if outlier_method == "IQR":
                                    q1, q3 = values.quantile([0.25, 0.75])
                                    iqr = q3 - q1
                                    lower = q1 - outlier_threshold * iqr
                                    upper = q3 + outlier_threshold * iqr
                                else:
                                    mean, std = values.mean(), values.std()
                                    lower = mean - outlier_threshold * std
                                    upper = mean + outlier_threshold * std
                                
                                outliers = working_df[(working_df[var] < lower) | (working_df[var] > upper)]
                                for idx, row in outliers.iterrows():
                                    flagged_records.append({
                                        'check_type': 'outlier',
                                        'variable': var,
                                        'value': row[var],
                                        'group': 'Overall',
                                        'lower_bound': round(lower, 2),
                                        'upper_bound': round(upper, 2),
                                        'record_index': idx
                                    })
            
            # 2. Duration checks
            if duration_col != "None":
                with st.spinner("Checking survey duration..."):
                    if duration_col not in working_df.columns:
                        st.error(f"❌ Column '{duration_col}' not found in dataset.")
                        st.stop()
                    
                    # Try to convert to numeric
                    try:
                        working_df[duration_col] = pd.to_numeric(working_df[duration_col], errors='coerce')
                    except Exception:
                        st.error(f"❌ Could not convert '{duration_col}' to numeric. Please select a numeric column.")
                        st.stop()
                    
                    if not pd.api.types.is_numeric_dtype(working_df[duration_col]):
                        st.error(f"❌ Column '{duration_col}' does not contain numeric values. Please select a different column.")
                        st.stop()
                    
                    durations = working_df[duration_col].copy()
                    
                    if duration_unit == "Minutes":
                        durations = durations * 60
                    
                    if check_method == "Quantile-based":
                        lower_thresh = durations.quantile(speed_quantile / 100)
                        upper_thresh = durations.quantile(slow_quantile / 100)
                    else:
                        lower_thresh = min_duration if duration_unit == "Seconds" else min_duration * 60
                        upper_thresh = max_duration if duration_unit == "Seconds" else max_duration * 60
                    
                    fast_surveys = working_df[working_df[duration_col] < lower_thresh]
                    slow_surveys = working_df[working_df[duration_col] > upper_thresh]
                    
                    for idx, row in fast_surveys.iterrows():
                        flagged_records.append({
                            'check_type': 'duration_fast',
                            'variable': duration_col,
                            'value': row[duration_col],
                            'threshold': round(lower_thresh, 2),
                            'record_index': idx
                        })
                    
                    for idx, row in slow_surveys.iterrows():
                        flagged_records.append({
                            'check_type': 'duration_slow',
                            'variable': duration_col,
                            'value': row[duration_col],
                            'threshold': round(upper_thresh, 2),
                            'record_index': idx
                        })
            
            # 3. Duplicate detection
            if duplicate_keys:
                with st.spinner("Detecting duplicates..."):
                    duplicates = working_df[working_df.duplicated(subset=duplicate_keys, keep=False)]
                    
                    for idx, row in duplicates.iterrows():
                        key_vals = {k: row[k] for k in duplicate_keys}
                        flagged_records.append({
                            'check_type': 'duplicate',
                            'keys': str(key_vals),
                            'record_index': idx
                        })
            
            # 4. Treatment fidelity
            if treatment_col != "None":
                with st.spinner("Checking intervention fidelity..."):
                    invalid_treatments = working_df[~working_df[treatment_col].isin(expected_vals)]
                    
                    for idx, row in invalid_treatments.iterrows():
                        flagged_records.append({
                            'check_type': 'invalid_treatment',
                            'variable': treatment_col,
                            'value': row[treatment_col],
                            'expected': ', '.join(map(str, expected_vals)),
                            'record_index': idx
                        })
            
            # Display results
            st.success(f"✅ Quality checks complete! Found {len(flagged_records)} issues.")
            
            if flagged_records:
                flagged_df = pd.DataFrame(flagged_records)
                
                # Group results if requested
                if group_results_by != "None" and group_results_by in working_df.columns:
                    flagged_df = flagged_df.merge(
                        working_df[[group_results_by]],
                        left_on='record_index',
                        right_index=True,
                        how='left'
                    )
                    
                    # Show summary by group
                    summary = flagged_df.groupby([group_results_by, 'check_type']).size().reset_index(name='count')
                    st.markdown("#### Issues by Group")
                    st.dataframe(summary, use_container_width=True)
                    
                    # Show details in expanders
                    st.markdown("#### Detailed Results")
                    for group_val, group_data in flagged_df.groupby(group_results_by):
                        with st.expander(f"{group_results_by}: {group_val} ({len(group_data)} issues)"):
                            st.dataframe(group_data, use_container_width=True)
                else:
                    # Show all flagged cases
                    st.dataframe(flagged_df, use_container_width=True)
                
                # Download button
                csv = flagged_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Flagged Cases",
                    csv,
                    "flagged_cases.csv",
                    "text/csv",
                    key='download-flagged'
                )
            else:
                st.info("✨ No issues found! All data passed quality checks.")


# ----------------------------------------------------------------------------- #
# ANALYSIS & RESULTS                                                            #
# ----------------------------------------------------------------------------- #


def render_analysis() -> None:
    st.title("📊 Analysis & Results")
    st.markdown("Run statistical analysis on your endline data with interactive configuration.")
    
    # Initialize session state for analysis
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data: pd.DataFrame | None = None
    if "baseline_for_attrition" not in st.session_state:
        st.session_state.baseline_for_attrition: pd.DataFrame | None = None
    
    # Data source tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Endline Data", "📊 Attrition Analysis", "ℹ️ Help"])
    
    with tab1:
        st.markdown("#### Load Endline/Follow-up Data")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            data_source = st.radio(
                "Data source",
                ["Upload File", "SurveyCTO API"],
                key="analysis_data_source",
                horizontal=True
            )
        
        if data_source == "Upload File":
            upload = st.file_uploader(
                "Upload endline data (CSV or Stata .dta)",
                type=["csv", "dta"],
                key="analysis_upload",
                help="Upload your follow-up/endline survey data in CSV or Stata format"
            )
            if upload:
                try:
                    if upload.name.endswith('.dta'):
                        df = load_data(upload)
                        st.session_state.analysis_data = df
                        st.success(f"✅ Loaded Stata file: {len(df):,} observations with {len(df.columns)} variables.")
                    else:
                        df = pd.read_csv(upload)
                        st.session_state.analysis_data = df
                        st.success(f"✅ Loaded CSV file: {len(df):,} observations with {len(df.columns)} variables.")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            # SurveyCTO API
            col1, col2 = st.columns(2)
            with col1:
                server = st.text_input("SurveyCTO server", key="analysis_api_server", placeholder="myproject")
                username = st.text_input("Username", key="analysis_api_user")
            with col2:
                password = st.text_input("Password", type="password", key="analysis_api_pass")
                form_id = st.text_input("Form ID", key="analysis_api_form")
            
            if st.button("📥 Fetch from SurveyCTO", key="analysis_fetch"):
                if not all([server, username, password, form_id]):
                    st.error("All fields are required.")
                else:
                    try:
                        client = SurveyCTO(server=server, username=username, password=password)
                        df = client.get_submissions(form_id)
                        st.session_state.analysis_data = df
                        st.success(f"✅ Fetched {len(df):,} observations from SurveyCTO.")
                    except Exception as exc:
                        st.error(f"Failed to fetch data: {exc}")
        
        df = st.session_state.analysis_data
        
        if df is not None and not df.empty:
            st.markdown("---")
            
            # Data Cleaning Section
            st.markdown("#### 🧹 Data Cleaning (Optional)")
            
            with st.expander("📖 Data Cleaning Guide - Click to configure", expanded=False):
                st.markdown("""
                **Outlier Treatment**: Reduces impact of extreme values that may distort analysis.
                - **Winsorization**: Caps extremes (values < p1 → set to p1, values > p99 → set to p99)
                - **Trimming**: Removes extreme observations entirely
                - **Size-adjusted**: Regresses on strata dummies, then caps residuals (accounts for group differences)
                
                **Missing Value Treatment**: Strategies for handling incomplete data.
                - **Listwise deletion**: Remove any observation with missing values (reduces N)
                - **Indicator method**: Create missing dummy + impute (preserves N, controls for missingness)
                - **Imputation**: Fill missing values with mean, median, or zero
                
                **Pattern Detection**: Identify suspicious data patterns.
                - **All-zero patterns**: Flag observations where multiple variables are all zero (potential non-response)
                - **Impossible values**: Detect implausible values (e.g., negative counts, extreme ages)
                """)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Outlier Treatment**")
                    
                    outlier_method = st.radio(
                        "Method",
                        ["None", "Winsorization", "Trimming", "Size-adjusted Winsorization"],
                        index=0,
                        key="outlier_method",
                        help="Choose how to handle extreme values"
                    )
                    
                    if outlier_method != "None":
                        winsor_level = st.number_input(
                            "Percentile cutoff (each tail)",
                            min_value=0.1,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            key="winsor_level",
                            help="E.g., 1.0 = affect 1st & 99th percentiles (2% of data total)"
                        )
                        
                        if outlier_method == "Size-adjusted Winsorization":
                            strata_var = st.selectbox(
                                "Stratification variable",
                                ["None"] + [col for col in df.columns if df[col].dtype in ['object', 'category'] or df[col].nunique() < 20],
                                key="strata_var_winsor",
                                help="Regress outcomes on this variable before capping residuals"
                            )
                            strata_var = None if strata_var == "None" else strata_var
                        else:
                            strata_var = None
                
                with col2:
                    st.markdown("**🔢 Missing Value Treatment**")
                    
                    missing_method = st.radio(
                        "Method",
                        ["None", "Listwise deletion", "Indicator + Imputation", "Mean/Median imputation"],
                        index=0,
                        key="missing_method",
                        help="Choose how to handle missing data"
                    )
                    
                    if missing_method in ["Indicator + Imputation", "Mean/Median imputation"]:
                        imputation_value = st.radio(
                            "Imputation strategy",
                            ["Zero", "Mean", "Median"],
                            index=1,
                            key="imputation_value",
                            help="Value to use when filling missing data"
                        )
                    else:
                        imputation_value = None
                
                st.markdown("---")
                st.markdown("**🚩 Pattern-Based Detection (Optional)**")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    detect_all_zero = st.checkbox(
                        "Flag all-zero patterns",
                        value=False,
                        key="detect_all_zero",
                        help="Identify observations where selected variables are ALL zero"
                    )
                    
                    if detect_all_zero:
                        zero_check_cols = st.multiselect(
                            "Variables to check",
                            df.columns.tolist(),
                            key="zero_check_cols",
                            help="Flag if ALL selected variables = 0 (potential data quality issue)"
                        )
                    else:
                        zero_check_cols = []
                
                with col4:
                    detect_impossible = st.checkbox(
                        "Flag impossible values",
                        value=False,
                        key="detect_impossible",
                        help="Detect values outside plausible ranges"
                    )
                    
                    if detect_impossible:
                        impossible_rules = st.text_area(
                            "Rules (one per line)",
                            value="age < 0\nage > 120\nincome < 0",
                            height=100,
                            key="impossible_rules",
                            help="Format: variable_name < value or variable_name > value"
                        )
                    else:
                        impossible_rules = ""
                
                if st.button("🔧 Apply Data Cleaning", key="apply_cleaning", type="primary"):
                    # Save original data before cleaning
                    if 'original_analysis_data' not in st.session_state:
                        st.session_state.original_analysis_data = df.copy()
                    
                    cleaned_df = df.copy()
                    cleaning_log = []
                    
                    # Apply outlier treatment
                    if outlier_method != "None":
                        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        for col in numeric_cols:
                            if cleaned_df[col].notna().sum() > 10:  # Need sufficient data
                                original_col = f"{col}_original"
                                cleaned_df[original_col] = cleaned_df[col]
                                
                                if outlier_method == "Size-adjusted Winsorization" and strata_var and strata_var in cleaned_df.columns:
                                    # Size-adjusted winsorization
                                    from sklearn.linear_model import LinearRegression
                                    
                                    strata_dummies = pd.get_dummies(cleaned_df[strata_var], prefix='strata', drop_first=False)
                                    valid_mask = cleaned_df[col].notna() & cleaned_df[strata_var].notna()
                                    
                                    if valid_mask.sum() > strata_dummies.shape[1] + 5:
                                        X = strata_dummies[valid_mask]
                                        y = cleaned_df.loc[valid_mask, col]
                                        
                                        model = LinearRegression()
                                        model.fit(X, y)
                                        residuals = y - model.predict(X)
                                        
                                        lower_pct = np.percentile(residuals, winsor_level)
                                        upper_pct = np.percentile(residuals, 100 - winsor_level)
                                        
                                        fitted = model.predict(X)
                                        capped_residuals = np.clip(residuals, lower_pct, upper_pct)
                                        cleaned_df.loc[valid_mask, col] = fitted + capped_residuals
                                        
                                        n_affected = ((residuals < lower_pct) | (residuals > upper_pct)).sum()
                                        cleaning_log.append(f"✓ Size-adjusted winsorized `{col}`: {n_affected} values capped")
                                    else:
                                        # Fall back to simple
                                        lower_val = cleaned_df[col].quantile(winsor_level / 100)
                                        upper_val = cleaned_df[col].quantile(1 - winsor_level / 100)
                                        cleaned_df[col] = cleaned_df[col].clip(lower=lower_val, upper=upper_val)
                                        n_affected = ((cleaned_df[original_col] < lower_val) | (cleaned_df[original_col] > upper_val)).sum()
                                        cleaning_log.append(f"✓ Winsorized `{col}`: {n_affected} values capped at {winsor_level}% tails")
                                
                                elif outlier_method == "Winsorization":
                                    # Simple winsorization
                                    lower_val = cleaned_df[col].quantile(winsor_level / 100)
                                    upper_val = cleaned_df[col].quantile(1 - winsor_level / 100)
                                    cleaned_df[col] = cleaned_df[col].clip(lower=lower_val, upper=upper_val)
                                    n_affected = ((cleaned_df[original_col] < lower_val) | (cleaned_df[original_col] > upper_val)).sum()
                                    cleaning_log.append(f"✓ Winsorized `{col}`: {n_affected} values capped at {winsor_level}% tails")
                                
                                elif outlier_method == "Trimming":
                                    # Remove extreme values
                                    lower_val = cleaned_df[col].quantile(winsor_level / 100)
                                    upper_val = cleaned_df[col].quantile(1 - winsor_level / 100)
                                    n_before = len(cleaned_df)
                                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_val) & (cleaned_df[col] <= upper_val)]
                                    n_removed = n_before - len(cleaned_df)
                                    if n_removed > 0:
                                        cleaning_log.append(f"✓ Trimmed `{col}`: {n_removed} observations removed")
                    
                    # Handle missing values
                    if missing_method == "Listwise deletion":
                        n_before = len(cleaned_df)
                        cleaned_df = cleaned_df.dropna()
                        n_removed = n_before - len(cleaned_df)
                        cleaning_log.append(f"✓ Listwise deletion: {n_removed} observations with missing values removed")
                    
                    elif missing_method == "Indicator + Imputation":
                        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        for col in numeric_cols:
                            if cleaned_df[col].isna().sum() > 0:
                                indicator_col = f"{col}_missing"
                                cleaned_df[indicator_col] = cleaned_df[col].isna().astype(int)
                                
                                if imputation_value == "Zero":
                                    cleaned_df[col] = cleaned_df[col].fillna(0)
                                elif imputation_value == "Mean":
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                elif imputation_value == "Median":
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                                
                                n_missing = cleaned_df[indicator_col].sum()
                                cleaning_log.append(f"✓ `{col}`: Created indicator + imputed {n_missing} values with {imputation_value.lower()}")
                    
                    elif missing_method == "Mean/Median imputation":
                        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        for col in numeric_cols:
                            n_missing = cleaned_df[col].isna().sum()
                            if n_missing > 0:
                                if imputation_value == "Zero":
                                    cleaned_df[col] = cleaned_df[col].fillna(0)
                                elif imputation_value == "Mean":
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                elif imputation_value == "Median":
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                                
                                cleaning_log.append(f"✓ `{col}`: Imputed {n_missing} values with {imputation_value.lower()}")
                    
                    # Pattern detection: All-zero
                    if detect_all_zero and zero_check_cols and len(zero_check_cols) >= 2:
                        zero_mask = (cleaned_df[zero_check_cols] == 0).all(axis=1)
                        n_flagged = zero_mask.sum()
                        
                        if n_flagged > 0:
                            for col in zero_check_cols:
                                cleaned_df.loc[zero_mask, col] = np.nan
                            cleaning_log.append(f"⚠️ Flagged {n_flagged} observations with all-zero pattern across {len(zero_check_cols)} variables")
                            cleaning_log.append("   Variables set to missing (potential data quality issue)")
                    
                    # Pattern detection: Impossible values
                    if detect_impossible and impossible_rules:
                        for rule in impossible_rules.strip().split('\n'):
                            rule = rule.strip()
                            if not rule:
                                continue
                            
                            try:
                                if '<' in rule and 'or' not in rule.lower():
                                    var, threshold = rule.split('<')
                                    var, threshold = var.strip(), float(threshold.strip())
                                    if var in cleaned_df.columns:
                                        n_flagged = (cleaned_df[var] < threshold).sum()
                                        cleaned_df.loc[cleaned_df[var] < threshold, var] = np.nan
                                        if n_flagged > 0:
                                            cleaning_log.append(f"⚠️ Flagged {n_flagged} values in `{var}` < {threshold} as missing")
                                
                                elif '>' in rule:
                                    var, threshold = rule.split('>')
                                    var, threshold = var.strip(), float(threshold.strip())
                                    if var in cleaned_df.columns:
                                        n_flagged = (cleaned_df[var] > threshold).sum()
                                        cleaned_df.loc[cleaned_df[var] > threshold, var] = np.nan
                                        if n_flagged > 0:
                                            cleaning_log.append(f"⚠️ Flagged {n_flagged} values in `{var}` > {threshold} as missing")
                            except Exception as e:
                                st.warning(f"Could not parse rule: {rule}")
                    
                    # Store cleaned data (this replaces the raw data for analysis)
                    st.session_state.analysis_data = cleaned_df
                    st.session_state.cleaning_applied = True
                    st.session_state.cleaning_log = cleaning_log
                    
                    st.success(f"✅ Data cleaning complete! {len(cleaning_log)} operations applied. Cleaned data will be used for analysis.")
                    
                    st.rerun()
            
            # Show if cleaning was applied
            if st.session_state.get('cleaning_applied', False):
                st.success("✅ **Data cleaning active** - Using cleaned data for all analysis below")
                
                # Show cleaning log
                if st.session_state.get('cleaning_log'):
                    with st.expander("📋 View Cleaning Log", expanded=False):
                        for log_entry in st.session_state.cleaning_log:
                            st.markdown(log_entry)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download cleaned data
                    cleaned_csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cleaned Data (CSV)",
                        data=cleaned_csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Download the cleaned dataset for external use"
                    )
                
                with col2:
                    # Reset to original data
                    if st.button("↩️ Reset to Original Data", key="reset_cleaning", use_container_width=True):
                        if 'original_analysis_data' in st.session_state:
                            st.session_state.analysis_data = st.session_state.original_analysis_data
                            del st.session_state.original_analysis_data
                        st.session_state.cleaning_applied = False
                        if 'cleaning_log' in st.session_state:
                            del st.session_state.cleaning_log
                        st.rerun()
            
            else:
                # Show that raw data is being used
                if 'original_analysis_data' not in st.session_state:
                    st.info("ℹ️ Using raw data. Apply cleaning above if needed before analysis.")
            
            st.markdown("---")
            st.markdown("#### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(df):,} rows • {len(df.columns)} columns")
            
            # Analysis configuration
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Configuration")
            
            available_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            with st.form("analysis_config_form"):
                st.markdown("#### Basic Settings")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    treatment_col = st.selectbox(
                        "Treatment column",
                        available_cols,
                        index=available_cols.index("treatment") if "treatment" in available_cols else 0,
                        key="analysis_treatment_col",
                        help="Column containing treatment assignment"
                    )
                
                with col2:
                    cluster_col = st.selectbox(
                        "Cluster column (optional)",
                        ["None"] + available_cols,
                        key="analysis_cluster_col",
                        help="For cluster-robust standard errors"
                    )
                    cluster_col = None if cluster_col == "None" else cluster_col
                
                with col3:
                    weight_col = st.selectbox(
                        "Weight column (optional)",
                        ["None"] + numeric_cols,
                        key="analysis_weight_col",
                        help="For weighted least squares"
                    )
                    weight_col = None if weight_col == "None" else weight_col
                
                st.markdown("#### Outcome Variables")
                outcomes = st.multiselect(
                    "Select outcome variables",
                    numeric_cols,
                    key="analysis_outcomes",
                    help="Numeric variables to analyze as dependent variables"
                )
                
                st.markdown("#### Control Variables (Optional)")
                potential_controls = [col for col in available_cols if col not in [treatment_col] + (outcomes or [])]
                covariates = st.multiselect(
                    "Select covariates for adjustment",
                    potential_controls,
                    key="analysis_covariates",
                    help="Variables to include as controls in the regression"
                )
                
                st.markdown("#### 🔬 Estimation Method")
                
                # Show all method descriptions upfront
                with st.expander("📖 Method Descriptions - Click to expand", expanded=False):
                    st.markdown("""
                    **ATE/ITT (Average Treatment Effect / Intent-to-Treat)**: Compares all assigned to treatment vs all assigned to control, regardless of actual take-up. Gold standard for causal inference in RCTs. Estimates the average treatment effect of assignment (not receipt).
                    
                    **TOT (Treatment-on-Treated)**: Uses IV/2SLS to estimate the effect on those who actually received treatment. Accounts for imperfect compliance. Reports first-stage F-statistic to assess instrument strength (F>10 is strong).
                    
                    **LATE (Local Average Treatment Effect)**: Estimates the effect specifically for compliers—those who take up treatment when assigned but wouldn't otherwise. Uses Wald estimator (ITT / compliance rate). Useful for understanding treatment effects on the moveable population.
                    
                    **Binary Outcome (Logit/Probit)**: For binary (0/1) outcomes like enrollment, adoption, or participation. Uses maximum likelihood estimation instead of OLS. Reports marginal effects (average partial effects) in probability units—easier to interpret than log-odds.
                    
                    **Panel Fixed Effects**: For panel/longitudinal data with multiple observations per unit over time. Controls for all time-invariant unobservables (e.g., ability, location quality) via entity fixed effects. Identifies treatment effects from within-unit variation over time.
                    
                    **Heterogeneity Analysis**: Tests whether treatment effects differ across subgroups (e.g., by gender, age, baseline income). Uses interaction terms (Treatment × Subgroup) and reports an F-test for joint significance. Helps identify who benefits most from the intervention.
                    """)
                
                analysis_method = st.radio(
                    "Select estimation approach",
                    ["ATE/ITT (Average Treatment Effect)", "TOT (Treatment-on-Treated)", "LATE (Local Average Treatment Effect)", 
                     "Binary Outcome (Logit/Probit)", "Panel Fixed Effects", "Heterogeneity Analysis"],
                    key="analysis_method"
                )
                
                # Method-specific inputs
                baseline_outcome = None
                strata_cols_list = []
                takeup_var = None
                instrument_var = None
                panel_id = None
                time_var = None
                model_type = 'logit'
                moderator = None
                
                if analysis_method == "ATE/ITT (Average Treatment Effect)":
                    baseline_outcome = st.selectbox(
                        "Baseline outcome (optional, for ANCOVA)",
                        ["None"] + [col for col in numeric_cols if col not in outcomes],
                        key="baseline_outcome_itt"
                    )
                    baseline_outcome = None if baseline_outcome == "None" else baseline_outcome
                    
                    strata_input = st.text_input(
                        "Stratification variables (comma-separated, optional)",
                        key="strata_itt",
                        help="e.g., block, sector, size_category"
                    )
                    if strata_input:
                        strata_cols_list = [s.strip() for s in strata_input.split(',')]
                
                elif analysis_method == "TOT (Treatment-on-Treated)":
                    takeup_var = st.selectbox(
                        "Take-up variable (actual treatment received)",
                        potential_controls,
                        key="takeup_var_tot"
                    )
                    instrument_var = st.selectbox(
                        "Instrument (random assignment)",
                        [treatment_col] + [col for col in potential_controls if col != takeup_var],
                        key="instrument_var_tot"
                    )
                
                elif analysis_method == "LATE (Local Average Treatment Effect)":
                    takeup_var = st.selectbox(
                        "Take-up variable",
                        potential_controls,
                        key="takeup_var_late"
                    )
                    instrument_var = st.selectbox(
                        "Instrument (random assignment)",
                        [treatment_col] + [col for col in potential_controls if col != takeup_var],
                        key="instrument_var_late"
                    )
                
                elif analysis_method == "Binary Outcome (Logit/Probit)":
                    model_type = st.radio(
                        "Model type",
                        ["logit", "probit"],
                        key="binary_model_type",
                        horizontal=True
                    )
                
                elif analysis_method == "Panel Fixed Effects":
                    panel_id = st.selectbox(
                        "Panel identifier (individual/household/firm ID)",
                        available_cols,
                        key="panel_id"
                    )
                    time_var = st.selectbox(
                        "Time variable",
                        available_cols,
                        key="time_var"
                    )
                
                elif analysis_method == "Heterogeneity Analysis":
                    moderator = st.selectbox(
                        "Subgroup variable (moderator)",
                        potential_controls,
                        key="moderator_het"
                    )
                
                st.markdown("#### Additional Options")
                col1, col2 = st.columns(2)
                with col1:
                    run_balance = st.checkbox("Run Balance Check", value=True, key="run_balance")
                    show_visualizations = st.checkbox("Show Visualizations", value=True, key="show_viz")
                
                with col2:
                    export_table = st.checkbox("Export Results Table", value=False, key="export_table")
                
                submit = st.form_submit_button("🔬 Run Analysis", type="primary", use_container_width=True)
            
            if submit:
                if not outcomes:
                    st.error("Please select at least one outcome variable.")
                else:
                    # Create analysis config
                    config = AnalysisConfig(
                        treatment_column=treatment_col,
                        weight_column=weight_col,
                        cluster_column=cluster_col
                    )
                    
                    st.markdown("---")
                    st.markdown("## 📈 Results")
                    
                    # Balance table using enhanced generate_balance_table
                    if run_balance and covariates:
                        st.markdown("### ⚖️ Balance Table")
                        st.markdown("Check if covariates are balanced across treatment arms.")
                        
                        try:
                            balance_table = generate_balance_table(
                                df=df,
                                treatment_col=treatment_col,
                                covariate_cols=covariates,
                                cluster_col=cluster_col
                            )
                            
                            # Display formatted balance table
                            st.dataframe(balance_table, use_container_width=True)
                            st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.10 | Joint F-test shown in last row")
                            
                            # Check for imbalances
                            sig_imbalances = balance_table[balance_table['stars'].str.contains(r'\*', na=False)]
                            if len(sig_imbalances) > 0:
                                st.warning(f"⚠️ {len(sig_imbalances)} covariate(s) show significant imbalance")
                            else:
                                st.success("✅ All covariates are balanced across treatment arms")
                            
                            # Visualization
                            if show_visualizations:
                                fig = plot_balance(balance_table)
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error generating balance table: {e}")
                    
                    # Method-specific analysis
                    st.markdown("### 📊 Treatment Effect Estimates")
                    
                    all_results = []
                    
                    for outcome in outcomes:
                        st.markdown(f"#### {outcome}")
                        
                        try:
                            # Run analysis based on selected method
                            if analysis_method == "ATE/ITT (Average Treatment Effect)":
                                results = estimate_itt(
                                    df=df,
                                    outcome_col=outcome,
                                    treatment_col=treatment_col,
                                    covariate_cols=covariates,
                                    cluster_col=cluster_col,
                                    baseline_outcome_col=baseline_outcome,
                                    strata_cols=strata_cols_list
                                )
                                
                                # Display formatted table
                                formatted_table = format_regression_table(results)
                                st.markdown("**ITT Estimates (3 Specifications):**")
                                st.dataframe(formatted_table, use_container_width=True)
                                st.caption("Spec 1: No controls | Spec 2: With controls | Spec 3: ANCOVA (with baseline)")
                                
                                # Visualizations
                                if show_visualizations:
                                    fig = plot_coefficients(results, title=f"ITT Effects on {outcome}")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    fig_dist = plot_distributions(df, outcome, treatment_col)
                                    st.plotly_chart(fig_dist, use_container_width=True)
                            
                            elif analysis_method == "TOT (Treatment-on-Treated)":
                                results = estimate_tot(
                                    df=df,
                                    outcome_col=outcome,
                                    treatment_col=takeup_var,
                                    instrument_col=instrument_var,
                                    covariate_cols=covariates,
                                    cluster_col=cluster_col
                                )
                                
                                # Display results
                                st.markdown("**First Stage:**")
                                st.write(f"F-statistic: {results['first_stage_fstat']:.2f}")
                                st.write(f"Instrument strength: {'Strong' if results['first_stage_fstat'] > 10 else 'Weak'}")
                                
                                st.markdown("**Second Stage (TOT Estimate):**")
                                formatted_table = format_regression_table({'spec1': results['second_stage']})
                                st.dataframe(formatted_table, use_container_width=True)
                                
                                if show_visualizations:
                                    fig = plot_coefficients({'TOT': results['second_stage']}, 
                                                          title=f"TOT Effect on {outcome}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            elif analysis_method == "LATE (Local Average Treatment Effect)":
                                results = estimate_late(
                                    df=df,
                                    outcome_col=outcome,
                                    treatment_col=takeup_var,
                                    instrument_col=instrument_var,
                                    covariate_cols=covariates,
                                    cluster_col=cluster_col
                                )
                                
                                # Display LATE interpretation
                                st.markdown("**LATE (Complier Average Causal Effect):**")
                                st.metric("Compliance Rate", f"{results['compliance_rate']:.1%}")
                                st.metric("LATE Estimate", f"{results['late_estimate']:.4f}")
                                st.metric("Std Error", f"{results['late_se']:.4f}")
                                st.write(results['interpretation'])
                                
                                formatted_table = format_regression_table({'LATE': results['second_stage']})
                                st.dataframe(formatted_table, use_container_width=True)
                            
                            elif analysis_method == "Binary Outcome (Logit/Probit)":
                                results = estimate_binary_outcome(
                                    df=df,
                                    outcome_col=outcome,
                                    treatment_col=treatment_col,
                                    covariate_cols=covariates,
                                    cluster_col=cluster_col,
                                    model_type=model_type
                                )
                                
                                st.markdown(f"**{model_type.title()} Model Results:**")
                                st.write(f"Marginal Effect: {results['marginal_effect']:.4f} ({results['marginal_effect_se']:.4f})")
                                st.write(f"Percentage points: {results['marginal_effect']*100:.2f}pp")
                                
                                formatted_table = format_regression_table({model_type: results['model']})
                                st.dataframe(formatted_table, use_container_width=True)
                            
                            elif analysis_method == "Panel Fixed Effects":
                                results = estimate_panel_fe(
                                    df=df,
                                    outcome_col=outcome,
                                    treatment_col=treatment_col,
                                    panel_id_col=panel_id,
                                    time_col=time_var,
                                    covariate_cols=covariates,
                                    cluster_col=cluster_col
                                )
                                
                                st.markdown("**Panel Fixed Effects Results:**")
                                formatted_table = format_regression_table({'Panel FE': results})
                                st.dataframe(formatted_table, use_container_width=True)
                                st.caption("Controls for time-invariant unobservables via entity FE")
                            
                            elif analysis_method == "Heterogeneity Analysis":
                                results = estimate_heterogeneity(
                                    df=df,
                                    outcome_col=outcome,
                                    treatment_col=treatment_col,
                                    subgroup_col=moderator,
                                    covariate_cols=covariates,
                                    cluster_col=cluster_col
                                )
                                
                                st.markdown("**Heterogeneous Treatment Effects:**")
                                st.write(f"Interaction F-statistic: {results['interaction_fstat']:.2f}")
                                st.write(f"P-value: {results['interaction_pvalue']:.4f}")
                                
                                # Show subgroup effects
                                st.markdown("**Effects by Subgroup:**")
                                for subgroup, effect in results['subgroup_effects'].items():
                                    st.write(f"{subgroup}: {effect['coef']:.4f} (SE: {effect['se']:.4f})")
                                
                                if show_visualizations:
                                    fig = plot_heterogeneity(results, moderator)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Store results for export
                            all_results.append({
                                'outcome': outcome,
                                'method': analysis_method,
                                'results': results
                            })
                            
                        except Exception as e:
                            st.error(f"Error analyzing {outcome}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    # Export results
                    if all_results and export_table:
                        st.markdown("---")
                        st.markdown("### 📥 Download Results")
                        
                        # Create export dataframe (simplified)
                        export_data = []
                        for res in all_results:
                            export_data.append({
                                'Outcome': res['outcome'],
                                'Method': res['method'],
                                'Results': str(res['results'])
                            })
                        
                        results_export = pd.DataFrame(export_data)
                        
                        csv = results_export.to_csv(index=False)
                        st.download_button(
                            "📥 Download as CSV",
                            csv,
                            f"{analysis_method.lower().replace(' ', '_')}_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    

        
        else:
            st.info("👆 Upload endline data or fetch from SurveyCTO to begin analysis.")
    
    with tab2:
        st.markdown("### 📉 Attrition Analysis")
        st.markdown("Compare baseline enrollment with endline completion to calculate attrition rates.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Baseline Data")
            baseline_upload = st.file_uploader(
                "Upload baseline data (CSV)",
                type="csv",
                key="attrition_baseline",
                help="Data with randomized participants"
            )
            if baseline_upload:
                baseline = pd.read_csv(baseline_upload)
                st.session_state.baseline_for_attrition = baseline
                st.success(f"✅ Loaded {len(baseline):,} baseline observations")
        
        with col2:
            st.markdown("#### Endline Data")
            if st.session_state.analysis_data is not None:
                st.success(f"✅ Using loaded endline data ({len(st.session_state.analysis_data):,} obs)")
            else:
                st.info("Load endline data in the first tab")
        
        baseline = st.session_state.baseline_for_attrition
        endline = st.session_state.analysis_data
        
        if baseline is not None and endline is not None:
            st.markdown("---")
            st.markdown("#### Configuration")
            
            baseline_cols = baseline.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                id_col = st.selectbox(
                    "ID column",
                    baseline_cols,
                    index=baseline_cols.index("participant_id") if "participant_id" in baseline_cols else 0,
                    key="attrition_id_col"
                )
            
            with col2:
                treatment_col = st.selectbox(
                    "Treatment column",
                    baseline_cols,
                    index=baseline_cols.index("treatment") if "treatment" in baseline_cols else 0,
                    key="attrition_treatment_col"
                )
            
            if st.button("📊 Calculate Attrition", type="primary"):
                try:
                    attrition_df = attrition_table(baseline, endline, id_col, treatment_col)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Attrition Results")
                    
                    # Format the table
                    attrition_df['attrition_rate'] = (attrition_df['rate'] * 100).round(2).astype(str) + '%'
                    attrition_df['followed_up'] = attrition_df['count'] - (attrition_df['rate'] * attrition_df['count']).astype(int)
                    attrition_df['attrited'] = (attrition_df['rate'] * attrition_df['count']).astype(int)
                    
                    display_df = attrition_df[[treatment_col, 'count', 'followed_up', 'attrited', 'attrition_rate']]
                    display_df.columns = ['Treatment Arm', 'Baseline N', 'Followed Up', 'Attrited', 'Attrition Rate']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Summary statistics
                    overall_attrition = attrition_df['rate'].mean() * 100
                    max_attrition = attrition_df['rate'].max() * 100
                    min_attrition = attrition_df['rate'].min() * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Attrition", f"{overall_attrition:.2f}%")
                    with col2:
                        st.metric("Highest Attrition", f"{max_attrition:.2f}%")
                    with col3:
                        st.metric("Lowest Attrition", f"{min_attrition:.2f}%")
                    
                    # Interpretation
                    if overall_attrition > 20:
                        st.warning(f"⚠️ High attrition rate ({overall_attrition:.1f}%). Consider sensitivity analyses.")
                    elif overall_attrition > 10:
                        st.info(f"ℹ️ Moderate attrition rate ({overall_attrition:.1f}%). Check for differential attrition.")
                    else:
                        st.success(f"✅ Low attrition rate ({overall_attrition:.1f}%).")
                    
                    # Check differential attrition
                    if len(attrition_df) > 1:
                        diff = max_attrition - min_attrition
                        if diff > 5:
                            st.warning(f"⚠️ Differential attrition detected: {diff:.1f} percentage point difference between arms.")
                    
                    # Download
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Attrition Table",
                        csv,
                        "attrition_analysis.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error calculating attrition: {e}")
        else:
            st.info("💡 Upload both baseline and endline data to calculate attrition rates.")
    
    with tab3:
        st.markdown("### ℹ️ Analysis Guide")
        
        st.markdown("""
        #### Average Treatment Effects (ATE)
        
        Estimates the causal effect of treatment assignment on outcomes using OLS regression:
        
        - **Simple ATE**: `outcome ~ treatment`
        - **With covariates**: `outcome ~ treatment + covariates`
        - **Clustered SEs**: Account for within-cluster correlation
        - **Weighted**: Use sampling or inverse probability weights
        
        **Interpretation**: The coefficient on treatment represents the average difference in outcomes 
        between treatment and control groups.
        
        #### Heterogeneity Analysis
        
        Tests whether treatment effects vary by subgroups using interaction terms:
        
        - Model: `outcome ~ treatment * moderator + covariates`
        - Significant interactions indicate effect heterogeneity
        - Common moderators: gender, age groups, baseline levels
        
        #### Balance Table
        
        Verifies that baseline covariates are balanced across treatment arms:
        
        - Tests: `covariate ~ treatment` for each covariate
        - P-value > 0.05 indicates good balance
        - Imbalanced covariates should be included as controls
        
        #### Attrition Analysis
        
        Compares baseline enrollment with endline completion:
        
        - Calculates attrition rate by treatment arm
        - High attrition (>20%) may threaten validity
        - Differential attrition requires sensitivity analysis
        
        #### Tips
        
        - Always check balance before estimating ATEs
        - Include pre-specified covariates to increase precision
        - Use cluster-robust SEs for cluster-randomized designs
        - Report both unadjusted and adjusted estimates
        - Account for multiple testing if analyzing many outcomes
        """)
    
    with tab3:
        st.markdown("### ℹ️ Help & Documentation")
        st.markdown("""
        ### 🚀 Quick Start
        
        1. **Upload Data**: Load your endline/follow-up survey data (CSV or Stata .dta)
        2. **Optional Cleaning**: Apply winsorization, handle missing values, flag data quality issues
        3. **Configure Analysis**: Select treatment variable, outcomes, and estimation method
        4. **Run Analysis**: Execute and review results with visualizations
        
        ### 🧹 Data Cleaning Options
        
        **Winsorization**: Reduces the impact of extreme outliers by capping values at specified percentiles.
        - **Simple winsorization**: Caps at 1st and 99th percentiles
        - **Size-adjusted winsorization** (Bruhn-Karlan method): Regresses outcome on strata dummies, caps residuals, then reconstructs values
        - Useful for: sales, profits, employment, assets
        
        **Missing Value Handling**: Creates indicator variables for missing data and imputes with zero.
        - Pattern: Variable `x` → creates `x_d` (1 if missing, 0 otherwise) + fills `x` with 0
        - Allows inclusion of observations with missing covariates
        - Follows published RCT methodology (Bruhn & Karlan 2018)
        
        **Zero Sales Artifacts**: Detects firms reporting zero sales across multiple months.
        - If all monthly sales = 0 but firm has employees/assets, likely data quality issue
        - Sets sales to missing for these cases (following published replication code)
        - Prevents bias from data entry errors or non-response
        
        ### 📊 Estimation Methods
        
        **ATE/ITT (Average Treatment Effect)**: Gold standard for RCTs
        - Compares all assigned to treatment vs all assigned to control
        - Estimates effect of assignment (not receipt)
        - Three specifications: No controls, With covariates, ANCOVA with baseline
        
        **TOT (Treatment-on-Treated)**: Effect on actual recipients
        - Uses instrumental variables (IV/2SLS) to account for imperfect compliance
        - Reports first-stage F-statistic (>10 indicates strong instrument)
        - Larger than ITT if compliance < 100%
        
        **LATE (Local Average Treatment Effect)**: Effect on compliers
        - Estimates impact specifically for those induced to take treatment by assignment
        - Uses Wald estimator: ITT / compliance rate
        - Interpretation: Effect on the "moveable" population
        
        **Binary Outcome Models**: For 0/1 outcomes
        - Logit or Probit maximum likelihood estimation
        - Reports marginal effects (probability units)
        - Use for: adoption, enrollment, participation
        
        **Panel Fixed Effects**: For longitudinal data
        - Controls for all time-invariant unobservables via entity fixed effects
        - Requires multiple observations per unit over time
        - Identifies treatment effects from within-unit variation
        
        **Heterogeneity Analysis**: Subgroup effects
        - Tests if treatment effects differ by characteristics (gender, age, baseline income)
        - Uses interaction terms: Treatment × Subgroup
        - Reports F-test for joint significance
        
        ### 📈 Best Practices
        
        - **Always check balance** before estimating treatment effects
        - **Include pre-specified covariates** to increase precision and reduce SEs
        - **Use cluster-robust SEs** if randomization was at cluster level (villages, schools)
        - **Report multiple specifications**: Unadjusted, with controls, ANCOVA
        - **Account for multiple testing** when analyzing many outcomes (Bonferroni correction)
        - **Check attrition**: High or differential attrition (>20%) may threaten validity
        
        ### 📚 References
        
        This tool implements methods from published RCT studies:
        - Bruhn, Karlan & Schoar (2018): "The Impact of Consulting Services on Small and Medium Enterprises"
        - Attanasio et al (2011): "Risk pooling, risk preferences, and social networks"
        - Emerick et al (2016): "Technological Innovations, Downside Risk, and the Modernization of Agriculture"
        
        For technical details, see the [RANDOMIZATION.md](docs/RANDOMIZATION.md) documentation.
        """)


# ----------------------------------------------------------------------------- #
# BACKCHECK SELECTION                                                           #
# ----------------------------------------------------------------------------- #


def render_backcheck() -> None:
    st.title("🔍 Backcheck Selection")
    st.markdown("Select cases for quality backcheck interviews using stratified sampling.")
    
    # Initialize session state
    if "backcheck_data" not in st.session_state:
        st.session_state.backcheck_data: pd.DataFrame | None = None
    if "backcheck_flags" not in st.session_state:
        st.session_state.backcheck_flags: pd.DataFrame | None = None
    
    # Data loading
    st.markdown("### 📁 Load Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Submissions Data")
        submissions_upload = st.file_uploader(
            "Upload submissions (CSV)",
            type="csv",
            key="backcheck_submissions_file",
            help="Survey submissions for backcheck selection"
        )
        if submissions_upload:
            df = pd.read_csv(submissions_upload)
            st.session_state.backcheck_data = df
            st.success(f"✅ Loaded {len(df):,} submissions")
    
    with col2:
        st.markdown("#### Quality Flags (Optional)")
        flags_upload = st.file_uploader(
            "Upload quality flags (CSV)",
            type="csv",
            key="backcheck_flags_file",
            help="Output from Quality Checks module (optional but recommended)"
        )
        if flags_upload:
            flags = pd.read_csv(flags_upload)
            st.session_state.backcheck_flags = flags
            st.success(f"✅ Loaded flags for {len(flags):,} submissions")
        else:
            st.info("💡 Upload quality flags to prioritize high-risk cases for backchecks")
    
    df = st.session_state.backcheck_data
    flags = st.session_state.backcheck_flags
    
    if df is None:
        st.info("👆 Upload submissions data to begin backcheck selection.")
        return
    
    st.markdown("---")
    st.markdown("### ⚙️ Backcheck Configuration")
    
    available_cols = df.columns.tolist()
    
    with st.form("backcheck_config_form"):
        st.markdown("#### Sample Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.number_input(
                "Total backcheck sample size",
                min_value=1,
                max_value=len(df),
                value=min(50, len(df)),
                step=1,
                key="backcheck_sample_size",
                help="Number of cases to select for backchecks"
            )
        
        with col2:
            high_risk_quota = st.slider(
                "High-risk quota",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key="backcheck_high_risk_quota",
                help="Proportion of sample from flagged cases (if flags provided)"
            )
        
        with col3:
            random_seed = st.number_input(
                "Random seed",
                min_value=1,
                value=42,
                step=1,
                key="backcheck_seed",
                help="For reproducible selection"
            )
        
        st.markdown("#### Columns to Include")
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_column = st.selectbox(
                "ID column",
                available_cols,
                index=available_cols.index("participant_id") if "participant_id" in available_cols else 0,
                key="backcheck_id_col",
                help="Unique identifier for each case"
            )
        
        with col2:
            contact_columns = st.multiselect(
                "Contact information columns",
                available_cols,
                default=[col for col in ["phone", "phone_number", "contact"] if col in available_cols],
                key="backcheck_contact_cols",
                help="Phone numbers or other contact details"
            )
        
        location_columns = st.multiselect(
            "Location columns",
            available_cols,
            default=[col for col in ["village", "community", "district", "gps_latitude", "gps_longitude"] 
                     if col in available_cols],
            key="backcheck_location_cols",
            help="Geographic information for field logistics"
        )
        
        additional_columns = st.multiselect(
            "Additional columns",
            [col for col in available_cols if col not in [id_column] + contact_columns + location_columns],
            key="backcheck_additional_cols",
            help="Other columns to include (e.g., enumerator, date, treatment)"
        )
        
        submit = st.form_submit_button("🎲 Generate Backcheck Roster", type="primary", use_container_width=True)
    
    if submit:
        # Create configuration
        config = {
            "sample_size": int(sample_size),
            "high_risk_quota": high_risk_quota,
            "random_seed": int(random_seed),
            "id_column": id_column,
            "contact_columns": contact_columns,
            "location_columns": location_columns,
        }
        
        try:
            # If no flags provided, create empty flags dataframe
            if flags is None:
                flags = pd.DataFrame(0, index=df.index, columns=['no_flags'])
            
            # Select backcheck sample
            backcheck_roster = sample_backchecks(df, flags, config)
            
            # Add additional columns if requested
            if additional_columns:
                for col in additional_columns:
                    if col in df.columns and col not in backcheck_roster.columns:
                        backcheck_roster = backcheck_roster.merge(
                            df[[id_column, col]],
                            on=id_column,
                            how='left'
                        )
            
            st.markdown("---")
            st.markdown("### ✅ Backcheck Roster Generated")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Selected", len(backcheck_roster))
            with col2:
                high_risk_count = len(backcheck_roster[backcheck_roster['risk_score'] > 0])
                st.metric("High-Risk Cases", high_risk_count)
            with col3:
                random_count = len(backcheck_roster[backcheck_roster['risk_score'] == 0])
                st.metric("Random Cases", random_count)
            
            # Show roster
            st.markdown("#### 📋 Backcheck Roster")
            st.dataframe(backcheck_roster, use_container_width=True)
            
            # Risk score distribution
            if 'risk_score' in backcheck_roster.columns:
                st.markdown("#### 📊 Risk Score Distribution")
                risk_summary = backcheck_roster['risk_score'].describe()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Risk Score", f"{risk_summary['mean']:.2f}")
                with col2:
                    st.metric("Max Risk Score", f"{risk_summary['max']:.0f}")
                with col3:
                    st.metric("Median Risk Score", f"{risk_summary['50%']:.1f}")
                with col4:
                    pct_flagged = (backcheck_roster['risk_score'] > 0).mean() * 100
                    st.metric("% Flagged", f"{pct_flagged:.1f}%")
            
            # Download options
            st.markdown("---")
            st.markdown("### 📥 Download Roster")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = backcheck_roster.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "backcheck_roster.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel export
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    backcheck_roster.to_excel(writer, index=False, sheet_name='Backcheck Roster')
                st.download_button(
                    "📥 Download as Excel",
                    buffer.getvalue(),
                    "backcheck_roster.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Instructions
            st.markdown("---")
            st.markdown("### 📝 Next Steps")
            st.info("""
            **Using the Backcheck Roster:**
            
            1. **Review the roster** to ensure it includes all necessary contact and location information
            2. **Assign to backcheck team** for re-interviews
            3. **Upload to SurveyCTO** as a case list (if using digital backchecks)
            4. **Compare results** between original and backcheck interviews to assess data quality
            5. **Take action** on enumerators with high discrepancy rates
            
            **High-risk cases** are prioritized based on quality flags (outliers, short duration, etc.).
            **Random cases** provide an unbiased sample for overall quality assessment.
            """)
            
        except Exception as e:
            st.error(f"Error generating backcheck roster: {e}")


# ----------------------------------------------------------------------------- #
# REPORT GENERATION                                                             #
# ----------------------------------------------------------------------------- #


def render_reports() -> None:
    st.title("📄 Report Generation")
    st.markdown("Generate formatted weekly reports with monitoring statistics and quality summaries.")
    
    st.markdown("### 📊 Report Data Sources")
    st.info("""
    **Required Data:**
    - Submissions data (from monitoring or upload)
    - Quality flags (from quality checks module)
    - Backcheck roster (optional)
    
    Reports include:
    - Survey progress by treatment arm
    - Productivity statistics by enumerator
    - Quality flag summaries
    - Backcheck roster (if provided)
    """)
    
    # Data collection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Submissions Data")
        submissions_upload = st.file_uploader(
            "Upload submissions (CSV)",
            type="csv",
            key="report_submissions",
            help="Survey submissions for the reporting period"
        )
        
        if submissions_upload:
            submissions_df = pd.read_csv(submissions_upload)
            st.success(f"✅ Loaded {len(submissions_df):,} submissions")
        else:
            submissions_df = None
            st.info("Upload submissions data")
    
    with col2:
        st.markdown("#### Quality Flags")
        flags_upload = st.file_uploader(
            "Upload quality flags (CSV)",
            type="csv",
            key="report_flags",
            help="Quality check results"
        )
        
        if flags_upload:
            flags_df = pd.read_csv(flags_upload)
            st.success(f"✅ Loaded {len(flags_df):,} flag records")
        else:
            flags_df = None
            st.info("Upload quality flags")
    
    if submissions_df is None:
        st.warning("⚠️ Upload submissions data to continue.")
        return
    
    st.markdown("---")
    st.markdown("### ⚙️ Report Configuration")
    
    with st.form("report_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report title",
                value="Weekly Field Monitoring Report",
                key="report_title"
            )
            
            report_period = st.text_input(
                "Reporting period",
                value="Week of [DATE]",
                key="report_period",
                help="e.g., 'Week of January 15-21, 2025'"
            )
        
        with col2:
            project_name = st.text_input(
                "Project name",
                value="RCT Field Project",
                key="report_project"
            )
            
            format_option = st.radio(
                "Output format",
                ["HTML only", "HTML + PDF"],
                key="report_format",
                help="PDF requires WeasyPrint with native dependencies"
            )
        
        # Column mapping
        st.markdown("#### Column Mapping")
        available_cols = submissions_df.columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox(
                "Date column",
                available_cols,
                index=available_cols.index("SubmissionDate") if "SubmissionDate" in available_cols else 0,
                key="report_date_col"
            )
        
        with col2:
            enumerator_col = st.selectbox(
                "Enumerator column",
                available_cols,
                index=available_cols.index("enumerator") if "enumerator" in available_cols else 0,
                key="report_enum_col"
            )
        
        with col3:
            treatment_col = st.selectbox(
                "Treatment column",
                available_cols,
                index=available_cols.index("treatment") if "treatment" in available_cols else 0,
                key="report_treatment_col"
            )
        
        submit = st.form_submit_button("📄 Generate Report", type="primary", use_container_width=True)
    
    if submit:
        try:
            st.markdown("---")
            st.markdown("### 📄 Report Preview")
            
            # Calculate statistics
            st.markdown("#### Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Submissions", f"{len(submissions_df):,}")
            with col2:
                if date_col in submissions_df.columns:
                    unique_dates = submissions_df[date_col].nunique()
                    st.metric("Survey Days", unique_dates)
            with col3:
                if enumerator_col in submissions_df.columns:
                    unique_enums = submissions_df[enumerator_col].nunique()
                    st.metric("Enumerators", unique_enums)
            with col4:
                if flags_df is not None:
                    total_flags = len(flags_df)
                    st.metric("Quality Flags", f"{total_flags:,}")
            
            # Progress by treatment
            if treatment_col in submissions_df.columns:
                st.markdown("#### Progress by Treatment Arm")
                progress = submissions_df[treatment_col].value_counts().reset_index()
                progress.columns = ['Treatment Arm', 'Count']
                progress['Percentage'] = (progress['Count'] / progress['Count'].sum() * 100).round(1)
                st.dataframe(progress, use_container_width=True)
            
            # Productivity
            if enumerator_col in submissions_df.columns:
                st.markdown("#### Enumerator Productivity")
                productivity = submissions_df[enumerator_col].value_counts().reset_index()
                productivity.columns = ['Enumerator', 'Surveys Completed']
                productivity['Average per Day'] = (
                    productivity['Surveys Completed'] / unique_dates
                ).round(1) if date_col in submissions_df.columns else None
                st.dataframe(productivity.head(10), use_container_width=True)
            
            # Quality flags summary
            if flags_df is not None:
                st.markdown("#### Quality Issues")
                flag_summary = flags_df.sum().sort_values(ascending=False).head(10)
                if not flag_summary.empty:
                    flag_df = pd.DataFrame({
                        'Issue Type': flag_summary.index,
                        'Count': flag_summary.values
                    })
                    st.dataframe(flag_df, use_container_width=True)
            
            # Generate HTML report
            st.markdown("---")
            st.markdown("### 📥 Download Report")
            
            # Create simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{report_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .metric {{ display: inline-block; margin: 10px 20px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                    .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <h1>{report_title}</h1>
                <p><strong>Project:</strong> {project_name}</p>
                <p><strong>Period:</strong> {report_period}</p>
                
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <div class="metric-value">{len(submissions_df):,}</div>
                    <div class="metric-label">Total Submissions</div>
                </div>
                
                <h2>Data Tables</h2>
                {submissions_df.head(20).to_html(index=False)}
                
                <p style="margin-top: 40px; color: #7f8c8d; font-size: 12px;">
                    Generated by RCT Field Flow on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                </p>
            </body>
            </html>
            """
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download HTML Report",
                    html_content,
                    "weekly_report.html",
                    "text/html",
                    use_container_width=True
                )
            
            with col2:
                if format_option == "HTML + PDF":
                    st.info("💡 PDF generation requires WeasyPrint. Download HTML and convert externally if needed.")
            
            st.success("✅ Report generated successfully!")
            
            # Add button to proceed to randomization
            st.markdown("---")
            st.markdown("### 🎲 Next Step: Randomization")
            st.info("With your design sprint complete and report generated, you're ready to randomize your baseline data.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("▶️ Proceed to Randomization", type="primary", use_container_width=True, key="proceed_to_random"):
                    st.session_state.current_page = "random"
                    st.session_state.selected_page = "random"
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error generating report: {e}")


# ----------------------------------------------------------------------------- #
# MONITORING DASHBOARD                                                          #
# ----------------------------------------------------------------------------- #


def render_monitoring() -> None:
    st.title("📈 Monitoring Dashboard")
    st.markdown("Real-time progress monitoring with interactive configuration.")
    
    # Initialize session state
    if "monitor_data" not in st.session_state:
        st.session_state.monitor_data: pd.DataFrame | None = None
    if "monitor_config" not in st.session_state:
        st.session_state.monitor_config = {}
    
    cfg = mon_load_config()

    # Data source selection
    st.markdown("### 📁 Data Source")
    source = st.radio(
        "Load data from",
        ["Use project config", "Upload CSV", "SurveyCTO API"],
        key="monitor_data_source",
        horizontal=True
    )

    data: pd.DataFrame | None = None

    if source == "Use project config":
        try:
            submissions = mon_load_submissions(cfg)
            data = mon_prepare_data(submissions, cfg)
            st.session_state.monitor_data = data
        except Exception as exc:  # pragma: no cover
            st.error(f"Couldn't load monitoring components using project config: {exc}")
            return
    elif source == "Upload CSV":
        upload = st.file_uploader("Upload submissions CSV", type="csv", key="monitor_csv_upload")
        if upload:
            data = pd.read_csv(upload, sep=None, engine="python")
            st.session_state.monitor_data = data
            st.success(f"✅ Loaded {len(data):,} submissions")
        else:
            data = st.session_state.get("monitor_data")
        if data is None:
            st.info("📤 Upload a CSV file to continue.")
            return
    else:  # SurveyCTO API
        col1, col2 = st.columns(2)
        surveycto_cfg = cfg.get("surveycto", {})
        with col1:
            server_default = surveycto_cfg.get("server", "")
            server = st.text_input(
                "SurveyCTO server (without https://)",
                value="",
                placeholder=server_default,
                key="monitor_api_server",
            )
            username_default = surveycto_cfg.get("username", "")
            username = st.text_input(
                "Username",
                value="",
                placeholder=username_default,
                key="monitor_api_user",
            )
        with col2:
            password = st.text_input("Password", type="password", key="monitor_api_pass")
            form_default = surveycto_cfg.get("form_id", "")
            form_id = st.text_input(
                "Form ID",
                value="",
                placeholder=form_default,
                key="monitor_api_form",
            )

        if st.button("📥 Fetch SurveyCTO submissions", key="monitor_fetch_api"):
            if not all([server, username, password, form_id]):
                st.error("Server, username, password, and form ID are required.")
            else:
                try:
                    client = SurveyCTO(server=server, username=username, password=password)
                    api_df = client.get_submissions(form_id)
                    st.session_state.monitor_data = api_df
                    st.success(f"✅ Fetched {len(api_df):,} submissions from SurveyCTO.")
                except Exception as exc:
                    st.error(f"Failed to fetch SurveyCTO submissions: {exc}")
        data = st.session_state.get("monitor_data")
        if data is None:
            st.info("💡 Enter credentials and click the fetch button to load live data.")
            return

    if data is None or data.empty:
        st.warning("⚠️ No submissions available. Check your data source.")
        return

    # Data preview
    st.markdown("---")
    st.markdown("#### 📊 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.caption(f"Showing 10 of {len(data):,} rows")

    # Interactive configuration
    st.markdown("---")
    st.markdown("### ⚙️ Dashboard Configuration")
    
    available_cols = data.columns.tolist()
    
    with st.form("monitor_config_form"):
        st.markdown("#### Column Mapping")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = st.selectbox(
                "Date column",
                available_cols,
                index=available_cols.index("SubmissionDate") if "SubmissionDate" in available_cols else 0,
                key="monitor_date_col",
                help="When the submission was made"
            )
        
        with col2:
            enumerator_col = st.selectbox(
                "Enumerator column",
                available_cols,
                index=available_cols.index("enumerator") if "enumerator" in available_cols else 0,
                key="monitor_enum_col",
                help="Who conducted the survey"
            )
        
        with col3:
            treatment_col = st.selectbox(
                "Treatment column",
                available_cols,
                index=available_cols.index("treatment") if "treatment" in available_cols else 0,
                key="monitor_treatment_col",
                help="Treatment assignment"
            )
        
        st.markdown("#### Work Week Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            work_start = st.time_input(
                "Work start time",
                value=pd.Timestamp("08:00").time(),
                key="monitor_work_start",
                help="Time enumerators start work each day"
            )
        
        with col2:
            work_end = st.time_input(
                "Work end time",
                value=pd.Timestamp("17:00").time(),
                key="monitor_work_end",
                help="Time enumerators end work each day"
            )
        
        with col3:
            rest_days = st.multiselect(
                "Rest days",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=["Sunday"],
                key="monitor_rest_days",
                help="Days when no surveying occurs"
            )
        
        st.markdown("#### Dashboard Display Options")
        col1, col2 = st.columns(2)
        
        with col1:
            show_productivity = st.checkbox("Show productivity table", value=True, key="monitor_show_prod")
            show_by_arm = st.checkbox("Show progress by treatment arm", value=True, key="monitor_show_arm")
        
        with col2:
            show_projections = st.checkbox("Show completion projections", value=True, key="monitor_show_proj")
            show_enumerators = st.checkbox("Show enumerator details", value=True, key="monitor_show_enum")
        
        st.markdown("#### Target Configuration (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            target_total = st.number_input(
                "Target total surveys",
                min_value=0,
                value=0,
                step=10,
                key="monitor_target_total",
                help="Leave as 0 to skip target visualization"
            )
        
        with col2:
            if target_total > 0:
                target_per_arm = st.text_input(
                    "Target per treatment arm (comma-separated)",
                    value="",
                    key="monitor_target_per_arm",
                    help="e.g., '100,100,100' for three equal arms"
                )
        
        submit = st.form_submit_button("📊 Generate Dashboard", type="primary", use_container_width=True)
    
    if submit:
        st.markdown("---")
        st.markdown("## 📊 Monitoring Dashboard")
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Submissions", f"{len(data):,}")
        with col2:
            if date_col in data.columns:
                unique_dates = data[date_col].nunique()
                st.metric("Survey Days", unique_dates)
        with col3:
            if enumerator_col in data.columns:
                unique_enums = data[enumerator_col].nunique()
                st.metric("Active Enumerators", unique_enums)
        with col4:
            if treatment_col in data.columns:
                unique_arms = data[treatment_col].nunique()
                st.metric("Treatment Arms", unique_arms)
        
        # Productivity table
        if show_productivity and enumerator_col in data.columns:
            st.markdown("### 👤 Enumerator Productivity")
            productivity = data[enumerator_col].value_counts().reset_index()
            productivity.columns = ['Enumerator', 'Surveys']
            
            if date_col in data.columns:
                unique_dates = data[date_col].nunique()
                productivity['Avg/Day'] = (productivity['Surveys'] / unique_dates).round(1)
            
            st.dataframe(productivity.head(20), use_container_width=True)
        
        # Progress by treatment arm
        if show_by_arm and treatment_col in data.columns:
            st.markdown("### 🎯 Progress by Treatment Arm")
            
            arm_progress = data[treatment_col].value_counts().reset_index()
            arm_progress.columns = ['Treatment Arm', 'Count']
            arm_progress['Percentage'] = (arm_progress['Count'] / arm_progress['Count'].sum() * 100).round(1)
            
            st.dataframe(arm_progress, use_container_width=True)
            
            # Visualize as bar chart
            chart_data = arm_progress.set_index('Treatment Arm')['Count']
            st.bar_chart(chart_data)
        
        # Projections
        if show_projections and date_col in data.columns and target_total > 0:
            st.markdown("### 📅 Completion Projection")
            
            # Calculate daily submission rate
            if date_col in data.columns:
                try:
                    # Convert to datetime if needed
                    dates = pd.to_datetime(data[date_col], errors='coerce')
                    daily_counts = dates.dt.date.value_counts().sort_index()
                    
                    if len(daily_counts) > 1:
                        avg_daily_rate = daily_counts.mean()
                        current_total = len(data)
                        remaining = target_total - current_total
                        
                        if remaining > 0 and avg_daily_rate > 0:
                            days_needed = remaining / avg_daily_rate
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Progress", f"{current_total:,} / {target_total:,}")
                            with col2:
                                pct_complete = (current_total / target_total * 100)
                                st.metric("% Complete", f"{pct_complete:.1f}%")
                            with col3:
                                st.metric("Daily Average", f"{avg_daily_rate:.1f}")
                            with col4:
                                st.metric("Days to Target", f"{days_needed:.0f}")
                        elif current_total >= target_total:
                            st.success(f"✅ Target reached! ({current_total:,}/{target_total:,})")
                except Exception:
                    st.info("Could not calculate projections for this date format.")
        
        # Enumerator details
        if show_enumerators and enumerator_col in data.columns:
            st.markdown("### 📋 Enumerator Details")
            
            selected_enum = st.selectbox(
                "Select enumerator to view details",
                sorted(data[enumerator_col].unique()),
                key="monitor_selected_enum"
            )
            
            enum_data = data[data[enumerator_col] == selected_enum]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Surveys", len(enum_data))
            with col2:
                if date_col in data.columns:
                    enum_dates = enum_data[date_col].nunique()
                    st.metric("Days Active", enum_dates)
            with col3:
                if treatment_col in data.columns and date_col in data.columns:
                    try:
                        dates = pd.to_datetime(enum_data[date_col], errors='coerce')
                        last_submission = dates.max()
                        st.metric("Last Submission", last_submission.strftime("%Y-%m-%d") if pd.notna(last_submission) else "N/A")
                    except Exception:
                        pass
            
            if treatment_col in data.columns:
                st.markdown("#### Surveys by Arm")
                arm_dist = enum_data[treatment_col].value_counts().reset_index()
                arm_dist.columns = ['Treatment Arm', 'Count']
                st.dataframe(arm_dist, use_container_width=True)
        
        # Export data
        st.markdown("---")
        st.markdown("### 📥 Download Data")
        
        csv = data.to_csv(index=False)
        st.download_button(
            "📥 Download submissions",
            csv,
            "submissions_export.csv",
            "text/csv",
            use_container_width=True
        )


# ----------------------------------------------------------------------------- #
# FACILITATOR DASHBOARD                                                         #
# ----------------------------------------------------------------------------- #


def render_facilitator_dashboard() -> None:
    """Render the facilitator dashboard for monitoring team progress."""
    st.title("👨‍🏫 Facilitator Dashboard")
    st.markdown("Monitor team progress and provide guidance during the RCT Design Activity workshop.")
    
    # Password protection
    if 'facilitator_authenticated' not in st.session_state:
        st.session_state.facilitator_authenticated = False
    
    if not st.session_state.facilitator_authenticated:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        with st.form("facilitator_password_form"):
            st.markdown("### Enter Facilitator Password")
            password = st.text_input("Password", type="password", key="facilitator_pwd", label_visibility="collapsed", placeholder="Enter facilitator password")
            submit = st.form_submit_button("Unlock", use_container_width=True, type="primary")
            
            if submit:
                if password == "facilitator2025":  # Default password
                    st.session_state.facilitator_authenticated = True
                    st.success("✅ Access granted!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        return
    
    # Authenticated - show dashboard
    st.success("✅ Authenticated as facilitator")
    
    if st.button("🔒 Logout", key="facilitator_logout"):
        st.session_state.facilitator_authenticated = False
        st.rerun()
    
    st.markdown("---")
    
    # Load team progress data
    progress_file = Path(__file__).parent / "rct-design" / "data" / "team_progress.json"
    
    if not progress_file.exists():
        st.warning("📊 No team progress data yet. Teams will appear here once they start the activity.")
        st.info("**Instructions:**\n- Teams enter their names in the sidebar\n- Progress is automatically tracked\n- Refresh this page to see updates")
        return
    
    try:
        with open(progress_file, "r") as f:
            team_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        st.error("Error loading team progress data")
        return
    
    if not team_data:
        st.warning("📊 No teams have started yet.")
        return
    
    # Summary metrics
    st.markdown("### 📊 Workshop Summary")
    
    teams_started = sum(1 for team in team_data.values() if team.get("started"))
    teams_completed = sum(1 for team in team_data.values() if team.get("completed"))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Teams", len(team_data))
    with col2:
        st.metric("Teams Started", teams_started)
    with col3:
        st.metric("Teams Completed", teams_completed)
    with col4:
        completion_rate = (teams_completed / len(team_data) * 100) if len(team_data) > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    st.markdown("---")
    
    # Team progress table
    st.markdown("### 📋 Team Progress")
    
    # Create progress dataframe
    progress_list = []
    for team_name, data in team_data.items():
        progress_list.append({
            "Team": team_name,
            "Program Card": data.get("program_card", "N/A"),
            "Started": "✅" if data.get("started") else "❌",
            "Current Step": data.get("current_step", 0),
            "Completed": "✅" if data.get("completed") else "⏳",
            "Started At": data.get("started_at", "N/A"),
            "Completed At": data.get("completed_at", "N/A")
        })
    
    progress_df = pd.DataFrame(progress_list)
    st.dataframe(progress_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Individual team details
    st.markdown("### 🔍 Team Details")
    
    selected_team = st.selectbox("Select team to view details:", list(team_data.keys()))
    
    if selected_team:
        team_info = team_data[selected_team]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Team Information:**")
            st.write(f"- **Program Card:** {team_info.get('program_card', 'N/A')}")
            st.write(f"- **Current Step:** {team_info.get('current_step', 0)}")
            st.write(f"- **Started:** {team_info.get('started_at', 'N/A')}")
            st.write(f"- **Completed:** {team_info.get('completed_at', 'N/A')}")
        
        with col2:
            # Load workbook responses if available
            workbook_dir = Path(__file__).parent / "rct-design" / "data" / "workbooks"
            safe_name = "".join(c for c in selected_team if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            workbook_file = workbook_dir / f"{safe_name}_workbook.json"
            
            if workbook_file.exists():
                try:
                    with open(workbook_file, "r") as f:
                        responses = json.load(f)
                    
                    st.markdown("**Workbook Progress:**")
                    filled_fields = sum(1 for v in responses.values() if v)
                    st.progress(filled_fields / max(len(responses), 1))
                    st.caption(f"{filled_fields}/{len(responses)} fields completed")
                except Exception:
                    st.warning("Could not load workbook data")
            else:
                st.info("No workbook data saved yet")
    
    st.markdown("---")
    
    # Coaching tips
    st.markdown("### 💡 Facilitator Tips")
    
    with st.expander("🎯 Common Challenges & Interventions"):
        st.markdown("""
        **If teams are stuck on Step 1 (Frame the Challenge):**
        - Ask: "Who exactly will benefit from this program?"
        - Prompt: "What does success look like in 6 months?"
        
        **If teams struggle with Theory of Change (Step 2):**
        - Use the "If-Then" framework: "If we do X, then Y will happen because..."
        - Draw the connection visually on a whiteboard
        
        **For Measurement issues (Step 3):**
        - Start with outcomes, work backwards to indicators
        - Ask: "How will you know if it's working?"
        
        **Randomization confusion (Step 4):**
        - Use simple examples (coin flip, lottery)
        - Emphasize fairness and statistical power
        """)
    
    # Download progress report
    st.markdown("---")
    st.markdown("### 📥 Export Progress Report")
    
    csv_data = progress_df.to_csv(index=False)
    st.download_button(
        "📥 Download Progress Report (CSV)",
        csv_data,
        f"workshop_progress_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True
    )


# ----------------------------------------------------------------------------- #
# USER INFORMATION PAGE (ADMIN ONLY)                                            #
# ----------------------------------------------------------------------------- #


def render_user_information():
    """Render the User Information page - admin only."""
    
    st.title("👥 User Information")
    st.markdown("---")
    
    # Check if current user is admin
    is_admin = st.session_state.get("username") == "aj-admin"
    
    # Password protection (fallback for non-logged in admin or other users trying to access)
    if 'userinfo_authenticated' not in st.session_state:
        st.session_state.userinfo_authenticated = False
    
    if not is_admin and not st.session_state.userinfo_authenticated:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        with st.form("userinfo_password_form"):
            st.markdown("### Enter Admin Password")
            password = st.text_input("Password", type="password", key="userinfo_pwd", label_visibility="collapsed", placeholder="Enter admin password")
            submit = st.form_submit_button("Unlock", use_container_width=True, type="primary")
            
            if submit:
                if password == "admin2025":  # Admin password
                    st.session_state.userinfo_authenticated = True
                    st.success("✅ Access granted!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        return
    
    # Logout button for admin
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔓 Lock Page", use_container_width=True):
            st.session_state.userinfo_authenticated = False
            st.rerun()
    
    st.markdown("---")
    
    # Persistent user listing (historical)
    st.markdown("### 🗂️ Historical Users")
    try:
        from .persistence import (
            fetch_all_users,
            fetch_user_session,
            fetch_user_activity,
            fetch_user_design,
            fetch_user_randomization,
            delete_user,
            anonymize_user,
            prune_activities,
            vacuum_db,
        )
        users_list = fetch_all_users()
    except Exception:
        users_list = []
    if users_list:
        # Display summary table of users with consent + security indicators
        user_df = pd.DataFrame([
            {
                "Username": u["username"],
                "Org": u.get("organization") or "",
                "Consent": "✅" if u.get("consent") else "❌",
                "Hashed": "🔒" if u.get("hashed") else "—",
                "Encrypted": "🛡️" if u.get("encrypted") else "—",
                "User ID": (u.get("user_id") or "")[0:10] + ("…" if u.get("user_id") and len(u.get("user_id")) > 10 else ""),
                "First Access": u.get("first_access"),
                "Last Access": u.get("last_access"),
            }
            for u in users_list
        ])
        st.dataframe(user_df, use_container_width=True, hide_index=True)
        usernames = [u["username"] for u in users_list]
        selected_history_user = st.selectbox(
            "Select user to view persisted data", usernames, key="userinfo_selected_history"
        )
    else:
        st.info("No persisted users found yet.")
        selected_history_user = None
    st.markdown("---")

    has_active_session = st.session_state.get("authentication_status")
    if has_active_session:
        username = st.session_state.get("username")
        org_type = "See DB" # We don't have this in session state currently
        # Calculate duration if we have access time, otherwise just show now
        access_time = st.session_state.get('access_time', datetime.now())
        if isinstance(access_time, str):
             try:
                 from dateutil import parser
                 access_time = parser.parse(access_time)
             except:
                 access_time = datetime.now()
        duration = datetime.now() - access_time
    else:
        username = None
        org_type = None
        duration = None
    
    # Session Summary
    if has_active_session:
        st.markdown("### 👤 Current Browser Session Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Username", username)
        with col2:
            st.metric("Organization", org_type if org_type else "Not specified")
        with col3:
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            st.metric("Session Duration", duration_str)
        st.markdown("---")
    else:
        st.info("No live session loaded – use historical selector above to inspect persisted data.")
        st.markdown("---")
    
    st.markdown("---")
    
    # Activity Log (live session) & historical persisted activity (selected user)
    st.markdown("### 📊 Activity Log (Current Session)")
    activity_log = st.session_state.get('activity_log', []) if has_active_session else []
    
    if activity_log:
        st.info(f"📝 Total activities logged: **{len(activity_log)}**")
        
        # Convert to DataFrame for display
        df_activity = pd.DataFrame(activity_log)
        
        # Format timestamp
        if 'timestamp' in df_activity.columns:
            df_activity['timestamp'] = pd.to_datetime(df_activity['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Show in expander
        with st.expander("View Activity Log Details", expanded=False):
            st.dataframe(df_activity, use_container_width=True)
        
        # Activity summary
        if 'action' in df_activity.columns:
            action_counts = df_activity['action'].value_counts()
            st.markdown("#### Activity Summary")
            for action, count in action_counts.items():
                st.markdown(f"- **{action}**: {count} time(s)")
    else:
        st.info("No activities logged yet.")
    
    st.markdown("---")
    # Historical persisted data for selected user
    if selected_history_user:
        st.markdown("### 🗄️ Persisted Data for Selected User")
        try:
            persisted_session = fetch_user_session(selected_history_user)
            persisted_activity = fetch_user_activity(selected_history_user)
            persisted_design = fetch_user_design(selected_history_user)
            persisted_randomization = fetch_user_randomization(selected_history_user)
        except Exception as e:
            st.error(f"Error loading persisted data: {e}")
            persisted_session = {}
            persisted_activity = []
            persisted_design = {}
            persisted_randomization = {}

        if persisted_session:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Username", persisted_session.get('username'))
            with col2:
                st.metric("Organization", persisted_session.get('organization') or 'N/A')
            with col3:
                st.metric("Last Access", persisted_session.get('last_access') or 'N/A')
        else:
            st.info("No session metadata persisted.")

        # Persisted activity summary
        if persisted_activity:
            st.markdown("#### Activity Summary (Persisted)")
            df_hist = pd.DataFrame(persisted_activity)
            with st.expander("View Persisted Activity Log", expanded=False):
                st.dataframe(df_hist, use_container_width=True)
            if 'action' in df_hist.columns:
                counts = df_hist['action'].value_counts()
                for act, ct in counts.items():
                    st.markdown(f"- **{act}**: {ct} time(s)")
        else:
            st.info("No persisted activity entries.")

        # Persisted design data
        if persisted_design:
            st.markdown("#### Design Data (Persisted)")
            st.write({k: v for k, v in persisted_design.items() if k != 'workbook_responses'})
            if persisted_design.get('workbook_responses'):
                with st.expander("Workbook Responses (Persisted)"):
                    for step, resp in persisted_design['workbook_responses'].items():
                        st.markdown(f"**{step}:**")
                        st.write(resp)
                        st.markdown("---")
        else:
            st.info("No persisted design data.")

        # Persisted randomization data
        if persisted_randomization:
            st.markdown("#### Randomization Data (Persisted)")
            st.write(persisted_randomization)
        else:
            st.info("No persisted randomization data.")
        st.markdown("---")

        # Downloads of persisted data
        st.markdown("### 💾 Download Persisted Data")
        export_payload = {
            'session': persisted_session,
            'activity': persisted_activity,
            'design': persisted_design,
            'randomization': persisted_randomization,
        }
        json_hist = json.dumps(export_payload, indent=2)
        st.download_button(
            "📥 Download Persisted JSON",
            json_hist,
            file_name=f"persisted_{selected_history_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
        if persisted_activity:
            df_hist_csv = pd.DataFrame(persisted_activity)
            csv_buf = BytesIO()
            df_hist_csv.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button(
                "📥 Download Persisted Activity CSV",
                csv_buf.getvalue(),
                file_name=f"persisted_activity_{selected_history_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("---")
    
    # RCT Design Data
    st.markdown("### 🎯 RCT Design Data")
    has_design_data = False
    
    if 'design_team_name' in st.session_state:
        has_design_data = True
        st.markdown(f"**Team Name:** {st.session_state.design_team_name}")
    
    if 'design_program_card' in st.session_state:
        has_design_data = True
        st.markdown(f"**Program Card:** {st.session_state.design_program_card}")
    
    if 'design_workbook_responses' in st.session_state:
        has_design_data = True
        responses = st.session_state.design_workbook_responses
        completed_steps = sum(1 for v in responses.values() if v)
        st.markdown(f"**Workbook Progress:** {completed_steps}/6 steps completed")
        
        with st.expander("View Workbook Responses"):
            for step, response in responses.items():
                if response:
                    st.markdown(f"**{step}:**")
                    st.write(response)
                    st.markdown("---")
    
    if not has_design_data:
        st.info("No RCT Design data available.")
    
    st.markdown("---")
    
    # Download Options
    st.markdown("### 💾 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 Complete Session Data (JSON)")
        st.caption("Includes all session info, activity log, and RCT design data")
        
        session_data = get_session_data()
        json_str = json.dumps(session_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"session_data_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 📊 Activity Log (CSV)")
        st.caption("Spreadsheet-friendly format of your activities")
        
        if activity_log:
            # Flatten activity log for CSV
            csv_data_list = []
            for log in activity_log:
                row = {
                    'timestamp': log['timestamp'],
                    'action': log['action'],
                    'page': log.get('page', ''),
                    'username': log.get('username', ''),
                    'organization': log.get('organization', '')
                }
                # Add details if present
                if 'details' in log:
                    for key, value in log['details'].items():
                        row[f'detail_{key}'] = value
                csv_data_list.append(row)
            
            df_csv = pd.DataFrame(csv_data_list)
            csv_buffer = BytesIO()
            df_csv.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"activity_log_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📥 Download CSV", disabled=True, use_container_width=True)
            st.caption("No activity data to download")
    
    st.markdown("---")
    # ------------------------------------------------------------------
    # Maintenance & Privacy Actions
    # ------------------------------------------------------------------
    st.markdown("### 🛠️ Maintenance & Privacy Actions")
    with st.expander("Admin Maintenance Tools", expanded=False):
        st.caption("Perform irreversible maintenance tasks. Proceed with care.")
        # User-specific actions
        if selected_history_user:
            st.markdown(f"#### Selected User: `{selected_history_user}`")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Anonymize User**")
                if st.button("🌀 Anonymize", key="btn_anonymize_user", use_container_width=True):
                    try:
                        new_name = anonymize_user(selected_history_user)
                        if new_name:
                            st.success(f"User anonymized. New handle: `{new_name}`")
                            st.rerun()
                        else:
                            st.error("User not found; cannot anonymize.")
                    except Exception as e:  # pragma: no cover
                        st.error(f"Error anonymizing: {e}")
            with col_b:
                st.markdown("**Delete User**")
                confirm_text = st.text_input(
                    "Type username to confirm deletion", key="delete_confirm_input"
                )
                if st.button("🗑️ Delete", key="btn_delete_user", use_container_width=True, disabled=not confirm_text):
                    if confirm_text == selected_history_user:
                        try:
                            delete_user(selected_history_user)
                            st.success("User and related data deleted.")
                            st.rerun()
                        except Exception as e:  # pragma: no cover
                            st.error(f"Deletion failed: {e}")
                    else:
                        st.error("Confirmation text does not match username.")
        else:
            st.info("Select a user above to enable user-specific actions.")

        st.markdown("---")
        # Prune activities before a given date/time
        st.markdown("#### Prune Old Activities")
        prune_col1, prune_col2, prune_col3 = st.columns([2, 2, 1])
        with prune_col1:
            prune_date = st.date_input("Delete entries before date", key="prune_date_input")
        with prune_col2:
            prune_time = st.time_input("Time (UTC)", key="prune_time_input")
        with prune_col3:
            if st.button("🧹 Prune", key="btn_prune", use_container_width=True):
                if prune_date:
                    from datetime import datetime
                    dt = datetime.combine(prune_date, prune_time)
                    iso_ts = dt.isoformat()
                    try:
                        deleted = prune_activities(iso_ts)
                        st.success(f"Pruned {deleted} activity rows before {iso_ts}.")
                    except Exception as e:  # pragma: no cover
                        st.error(f"Prune failed: {e}")
                else:
                    st.error("Select a date to prune.")

        st.markdown("---")
        # Vacuum database
        st.markdown("#### Optimize Database")
        if st.button("🧪 VACUUM", key="btn_vacuum_db", use_container_width=True):
            try:
                vacuum_db()
                st.success("Database vacuum completed.")
            except Exception as e:  # pragma: no cover
                st.error(f"VACUUM failed: {e}")

        st.caption(
            "Anonymization preserves user_id linkage; deletion removes all rows. "
            "Pruning only affects activities table."
        )

    st.markdown("---")
    
    # Data Privacy Notice
    with st.expander("ℹ️ About User Data"):
        st.markdown("""
        **What's included:**
        - Live session data (in-memory) AND persisted historical records
        - Persisted records include: session metadata, activities, design data, randomization summaries
        
        **Persistence model:**
        - Data written to local SQLite database `persistent_data/rct_field_flow.db`
        - Each activity and design/randomization update is appended or upserted
        - Historical data survives browser/session termination
        
        **Privacy & Compliance:**
        - Only username + organization type stored (no emails unless entered as username)
        - Delete a user by removing their rows from the database (manual admin action)
        - Consider adding an explicit consent checkbox for production use
        
        **Administration:**
        - Use this page to audit usage and export historical datasets
        - Password protects access (`admin2025`) – change for production
        """)
    
    st.markdown("---")
    st.caption("🔒 Administrator Access Only | Password: admin2025")


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# AUTHENTICATION & ACTIVITY LOGGING                                             #
# ----------------------------------------------------------------------------- #


# Pages that don't require access credentials (if any - currently all protected)
PUBLIC_PAGES = []

# Protected pages (require authentication)
PROTECTED_PAGES = ["design", "power", "random", "cases", "visualize", "quality", "analysis", 
                   "backcheck", "reports", "monitor", "home"]

# Admin pages
ADMIN_PAGES = ["facilitator", "userinfo"]


def init_auth():
    """Initialize authentication state and return authenticator object."""
    users = fetch_users_for_auth()
    
    # Convert to credentials dict format expected by streamlit-authenticator
    credentials = {"usernames": {}}
    for username, data in users.items():
        credentials["usernames"][username] = {
            "name": data["name"],
            "password": data["password"],
            "email": data.get("email", "")
        }

    # Add admin user if not exists (seed)
    if "aj-admin" not in credentials["usernames"]:
        # We should probably create it in DB if missing, but for now let's just handle it via registration
        pass

    authenticator = stauth.Authenticate(
        credentials,
        "rct_field_flow_auth",
        "rct_field_flow_key",
        cookie_expiry_days=30,
    )
    return authenticator


def render_login_page(authenticator):
    """Render the login page and registration form."""
    st.markdown("""
    <style>
    .auth-container {
        max-width: 500px;
        margin: 50px auto;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔐 RCT Field Flow")
    
    # Initialize mode in session state
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'register'  # Start with register mode
    
    # Registration Form (shown first)
    if st.session_state.auth_mode == 'register':
        st.subheader("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_password_confirm = st.text_input("Confirm Password", type="password")
            new_org = st.selectbox("Organization Type", [
                "University/Academic Institution",
                "Research Organization",
                "NGO/Non-Profit",
                "Government Agency",
                "Private Company",
                "Independent Researcher",
                "Student",
                "Other"
            ])
            submit_reg = st.form_submit_button("Register", use_container_width=True, type="primary")
            
            if submit_reg:
                if new_password != new_password_confirm:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif not new_username:
                    st.error("Username is required")
                else:
                    success = create_user(new_username, new_password, new_username, new_org)
                    if success:
                        st.success("Registration successful! Please login below.")
                        st.session_state.auth_mode = 'login'
                        st.rerun()
                    else:
                        st.error("Username already exists or error creating account.")
        
        # Toggle to login
        st.markdown("---")
        if st.button("Already registered? Log in", use_container_width=True):
            st.session_state.auth_mode = 'login'
            st.rerun()
    
    # Login Form
    else:
        st.subheader("Welcome Back")
        try:
            authenticator.login(location='main')
        except Exception as e:
            st.error(f"Error initializing login: {e}")
        
        if st.session_state.get("authentication_status"):
            st.rerun()
        elif st.session_state.get("authentication_status") is False:
            st.error("Username/password is incorrect")
        elif st.session_state.get("authentication_status") is None:
            pass  # Don't show warning, let the form speak for itself
        
        # Toggle to register
        st.markdown("---")
        if st.button("Need an account? Register", use_container_width=True):
            st.session_state.auth_mode = 'register'
            st.rerun()


def init_activity_log():
    """Initialize activity log in session state"""
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []


def log_activity(action: str, details: dict = None):
    """Log user activity with timestamp"""
    init_activity_log()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'page': st.session_state.get('current_page', 'unknown'),
    }
    
    # Add user info if available
    if st.session_state.get("authentication_status"):
        log_entry['username'] = st.session_state.get("username")
        log_entry['name'] = st.session_state.get("name")
    
    # Add additional details
    if details:
        log_entry['details'] = details
    
    st.session_state.activity_log.append(log_entry)
    # Persist activity if persistence available
    try:
        if log_entry.get('username'):
            record_activity(log_entry.get('username'), log_entry.get('page', 'unknown'), action, details)
    except Exception:  # pragma: no cover
        pass


def save_session_snapshot():
    """Save a snapshot of current session data to session state for persistence"""
    if not st.session_state.get('authentication_status'):
        return
    
    # Create a comprehensive snapshot
    snapshot = {
        'user_info': {
            'username': st.session_state.get('username'),
            'name': st.session_state.get('name'),
            'access_time': st.session_state.get('access_time', datetime.now()).isoformat() if hasattr(st.session_state.get('access_time', None), 'isoformat') else str(st.session_state.get('access_time')),
            'last_activity': datetime.now().isoformat(),
        },
        'design_data': {
            'team_name': st.session_state.get('design_team_name'),
            'program_card': st.session_state.get('design_program_card'),
            'current_step': st.session_state.get('design_current_step', 1),
            'workbook_responses': dict(st.session_state.get('design_workbook_responses', {})),
        },
        'activity_summary': {
            'total_actions': len(st.session_state.get('activity_log', [])),
            'last_page': st.session_state.get('current_page', 'home'),
        }
    }
    
    # Store snapshot in session state (persists during browser session)
    st.session_state.session_snapshot = snapshot
    st.session_state.last_save_time = datetime.now()
    # Persist design & randomization data if available
    try:
        if st.session_state.get('username'):
            upsert_design_data(
                st.session_state.username,
                st.session_state.get('design_team_name'),
                st.session_state.get('design_program_card'),
                st.session_state.get('design_current_step', 1),
                dict(st.session_state.get('design_workbook_responses', {})),
            )
            if 'randomization_result' in st.session_state and st.session_state.randomization_result:
                rr = st.session_state.randomization_result
                arms = [{'name': arm.name, 'size': arm.size} for arm in rr.treatment_arms]
                upsert_randomization(
                    st.session_state.username,
                    rr.total_units,
                    arms,
                    rr.timestamp.isoformat() if hasattr(rr, 'timestamp') else None,
                )
    except Exception:  # pragma: no cover
        pass


def restore_session_if_available():
    """Restore session data from snapshot if available (after page reload)"""
    # Check if we have a saved snapshot and user is authenticated but session state is not fully populated
    if 'session_snapshot' in st.session_state and st.session_state.get('authentication_status') and 'username' not in st.session_state:
        snapshot = st.session_state.session_snapshot
        
        # Restore user info
        if 'user_info' in snapshot:
            st.session_state.username = snapshot['user_info'].get('username')
            st.session_state.name = snapshot['user_info'].get('name')
            # Restore access time if available
            try:
                from dateutil import parser
                st.session_state.access_time = parser.parse(snapshot['user_info'].get('access_time'))
            except Exception:
                st.session_state.access_time = datetime.now()
        
        # Restore design data
        if 'design_data' in snapshot:
            design_data = snapshot['design_data']
            if design_data.get('team_name'):
                st.session_state.design_team_name = design_data.get('team_name')
            if design_data.get('program_card'):
                st.session_state.design_program_card = design_data.get('program_card')
            if design_data.get('current_step'):
                st.session_state.design_current_step = design_data.get('current_step')
            if design_data.get('workbook_responses'):
                st.session_state.design_workbook_responses = design_data.get('workbook_responses')
        
        # Log session restoration
        init_activity_log()
        st.session_state.activity_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'session_restored_from_snapshot',
            'page': 'system',
            'username': st.session_state.get('username'),
        })


def get_session_data() -> dict:
    """Compile all session data for download"""
    data = {
        'session_info': {
            'username': st.session_state.get('username', 'Anonymous'),
            'name': st.session_state.get('name', 'Not specified'),
            'access_time': st.session_state.get('access_time', datetime.now()).isoformat(),
            'export_time': datetime.now().isoformat(),
        },
        'activity_log': st.session_state.get('activity_log', []),
        'rct_design_data': {},
        'randomization_data': {},
        'other_data': {}
    }
    
    # Collect RCT Design data
    if 'design_team_name' in st.session_state:
        data['rct_design_data']['team_name'] = st.session_state.design_team_name
    if 'design_program_card' in st.session_state:
        data['rct_design_data']['program_card'] = st.session_state.design_program_card
    if 'design_workbook_responses' in st.session_state:
        data['rct_design_data']['workbook_responses'] = st.session_state.design_workbook_responses
    if 'design_current_step' in st.session_state:
        data['rct_design_data']['current_step'] = st.session_state.design_current_step
    
    # Collect randomization data
    if 'randomization_result' in st.session_state:
        result = st.session_state.randomization_result
        data['randomization_data'] = {
            'total_units': result.total_units,
            'treatment_arms': [{'name': arm.name, 'size': arm.size} for arm in result.treatment_arms],
            'timestamp': result.timestamp.isoformat() if hasattr(result, 'timestamp') else None
        }
    
    # Collect other relevant data
    data['other_data'] = {
        'current_page': st.session_state.get('current_page', 'unknown'),
        'pages_visited': list(set([log['page'] for log in st.session_state.get('activity_log', [])]))
    }
    
    return data


def require_auth(page_name: str) -> bool:
    """
    Check if page requires authentication and if user has it.
    Returns True if access is granted, False if login form should be shown.
    """
    # Public pages don't need access
    if page_name in PUBLIC_PAGES:
        return True
    
    # Admin pages have their own password protection, but still require a logged-in user
    if page_name in ADMIN_PAGES:
        return st.session_state.get("authentication_status")
    
    # All other pages require authentication
    return st.session_state.get("authentication_status")


def show_user_info_sidebar(authenticator):
    """Display current user info and logout button in sidebar."""
    if st.session_state.get("authentication_status"):
        st.sidebar.markdown("---")
        
        # Show current user
        user_name = st.session_state.get("name")
        username = st.session_state.get("username")
        
        st.sidebar.markdown("### 👤 Current Session")
        st.sidebar.markdown(f"**Name:** {user_name}")
        st.sidebar.markdown(f"**User:** {username}")
        
        # Logout button
        authenticator.logout("Logout", "sidebar")


# ----------------------------------------------------------------------------- #
# MAIN                                                                          #
# ----------------------------------------------------------------------------- #


def main() -> None:
    # Initialize persistence
    try:
        init_db()
    except Exception:  # pragma: no cover
        pass

    # Initialize Authentication
    authenticator = init_auth()
    
    # Check authentication status (this renders the login widget if needed)
    # But we want to control the login page rendering separately if not logged in
    # authenticator.login() is called inside render_login_page
    
    # We need to check if we are logged in. 
    # streamlit-authenticator manages state in st.session_state["authentication_status"]
    # But we need to call login() at least once to process the form submission?
    # Actually, calling login() renders the form.
    # If we are already logged in (cookie), login() returns the name/status without rendering form (usually).
    # Let's use render_login_page which calls login().
    
    # However, if we are logged in, we don't want to show the login page tabs.
    # So we call login() first to check status.
    
    # Hack: We call login() invisibly or check cookie? 
    # authenticator.login() renders the widget.
    # If we want a custom page, we render it.
    
    if not st.session_state.get("authentication_status"):
        render_login_page(authenticator)
        return

    # If we are here, we are logged in
    restore_session_if_available()
    
    # Visible navigation menu for users
    nav = {
        "home": "🏠 Home",
        "design": "🎯 RCT Design",
        "power": "⚡ Power Calculations",
        "random": "🎲 Randomization",
        "cases": "📋 Case Assignment",
        "visualize": "📊 Data Visualization",
        "quality": "✅ Quality Checks",
        "analysis": "📊 Analysis & Results",
        "backcheck": "🔍 Backcheck Selection",
        "reports": "📄 Report Generation",
        "monitor": "📈 Monitoring Dashboard",
    }
    
    # Hidden admin pages (not in menu, accessible via URL):
    # - ?page=userinfo - User Information (admin password: admin2025)
    # - ?page=facilitator - Facilitator Dashboard (facilitator password: facilitator2025)
    
    # All valid pages (including hidden ones)
    all_valid_pages = list(nav.keys()) + ["userinfo", "facilitator"]
    
    # Determine which page to show
    # Priority: URL parameter > programmatically set current_page > sidebar selection
    
    # Check URL parameters first
    query_params = st.query_params
    url_page = query_params.get("page", None)
    
    if url_page and url_page in all_valid_pages:
        # URL parameter takes highest priority
        page = url_page
    elif 'current_page' in st.session_state and st.session_state.current_page:
        # Programmatically set page
        page = st.session_state.current_page
    else:
        # Use sidebar for navigation
        nav_keys = list(nav.keys())
        
        # Get the default selection (use 'home' if nothing is set)
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "home"
        
        default_index = nav_keys.index(st.session_state.selected_page) if st.session_state.selected_page in nav_keys else 0
        
        page = st.sidebar.radio(
            "Navigation",
            options=nav_keys,
            format_func=lambda key: nav[key],
            label_visibility="collapsed",
            index=default_index,
            key="nav_radio"
        )
        
        # Update selected_page
        st.session_state.selected_page = page
    
    # Auto-save session data periodically
    if 'temp_user' in st.session_state:
        save_session_snapshot()

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear cached data"):
        for key in ["baseline_data", "randomization_result", "case_data", "quality_data", 
                    "analysis_data", "baseline_for_attrition", "backcheck_data", "backcheck_flags"]:
            st.session_state.pop(key, None)
        st.rerun()
    
    # Show user info and session controls
    show_user_info_sidebar(authenticator)

    # Add developer watermark to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.75rem; padding: 0.5rem 0;">
        <p style="margin: 0.3rem 0;"><strong>RCT Field Flow</strong></p>
        <p style="margin: 0.3rem 0; font-size: 0.7rem;">by <strong>Aubrey Jolex</strong></p>
        <p style="margin: 0.3rem 0; font-size: 0.65rem;">
        <a href="mailto:aubreyjolex@gmail.com" style="color: #888; text-decoration: none;">📧 Email</a> | 
        <a href="https://github.com/ajolex/rct_field_flow" target="_blank" style="color: #888; text-decoration: none;">GitHub</a>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Check authentication for protected pages (redundant as we return early, but good for safety)
    if page in PROTECTED_PAGES and page not in ADMIN_PAGES:
        if not st.session_state.get("authentication_status"):
             st.rerun()
    
    # Render the selected page
    if page == "home":
        render_home()
    elif page == "design":
        render_rct_design()
    elif page == "power":
        render_power_calculations()
    elif page == "random":
        render_randomization()
    elif page == "cases":
        render_case_assignment()
    elif page == "visualize":
        render_data_visualization()
    elif page == "quality":
        render_quality_checks()
    elif page == "analysis":
        render_analysis()
    elif page == "backcheck":
        render_backcheck()
    elif page == "reports":
        render_reports()
    elif page == "monitor":
        render_monitoring()
    elif page == "userinfo":
        render_user_information()
    elif page == "facilitator":
        render_facilitator_dashboard()
    
    # Clear the programmatic navigation flag after rendering
    if 'current_page' in st.session_state and st.session_state.current_page:
        st.session_state.current_page = None


def render_footer():
    """Render app footer with developer info"""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem 0;">
            <p style="margin: 0;">
            <strong>RCT Field Flow</strong> | Developed by <strong>Aubrey Jolex</strong><br>
            📧 <a href="mailto:aubreyjolex@gmail.com">aubreyjolex@gmail.com</a><br>
            <a href="https://github.com/ajolex/rct_field_flow" target="_blank">View on GitHub</a>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
    render_footer()
