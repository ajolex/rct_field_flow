from __future__ import annotations

import importlib.util
import io
import math
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

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
    from .analyze import AnalysisConfig, estimate_ate, heterogeneity_analysis, attrition_table
    from .backcheck import BackcheckConfig, sample_backchecks
    from .report import generate_weekly_report
    from .surveycto import SurveyCTO
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
    )
    from rct_field_flow.backcheck import BackcheckConfig, sample_backchecks  # type: ignore
    from rct_field_flow.report import generate_weekly_report  # type: ignore
    from rct_field_flow.surveycto import SurveyCTO  # type: ignore

# ----------------------------------------------------------------------------- #
# Page configuration & session state                                            #
# ----------------------------------------------------------------------------- #

st.set_page_config(page_title="RCT Field Flow", page_icon=":bar_chart:", layout="wide")

if "baseline_data" not in st.session_state:
    st.session_state.baseline_data: pd.DataFrame | None = None
if "randomization_result" not in st.session_state:
    st.session_state.randomization_result: RandomizationResult | None = None
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
        di "  Iteration " %6.0f `i' " / {config.iterations:,} (best p-value so far: " %6.4f `bestp' ")"
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
di "Best min p-value: " %6.4f `bestp'
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
    di "  `arm': " r(N) " observations (" %4.1f `pct_`arm'' "%)"
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
        di as error "  ⚠ WARNING: Significant imbalance detected (p = " %6.4f r(p) ")"
    }}
    else {{
        di as text "  ✓ Acceptable balance (p = " %6.4f r(p) ")"
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
        **Integrated toolkit** for designing RCTs, conducting randomization, managing cases, 
        quality assurance, analysis, and live monitoring.
        """
    )

    # New workflow overview with RCT Design Wizard
    st.markdown("## 📋 Complete RCT Workflow")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Phase 1: Design & Planning
        1. **🎯 RCT Design** – Build your concept note with 15 sections
           - Create comprehensive designs for education, health, agriculture projects
           - View realistic sample concept notes from different sectors
           - Export in multiple formats (Markdown, DOCX, PDF)
        
        ### Phase 2: Technical Setup
        2. **⚡ Power Calculations** – Determine sample size and power
           - Calculate minimum detectable effects (MDE)
           - Run power simulations with custom assumptions
           - Generate analysis code (Stata/Python)
        
        3. **🎲 Randomization** – Configure arms, strata, rerandomization
           - Set up treatment arms and stratification variables
           - Support for clustered and cross-clustered designs
           - Real-time balance checking
        
        ### Phase 3: Implementation
        4. **📋 Case Assignment** – Build SurveyCTO-ready cases dataset
           - Assign treatment groups to beneficiaries
           - Create tracking spreadsheets
           - Prepare for field deployment
        
        5. **🔍 Quality Checks** – Apply speed, outlier, duplicate checks
           - Monitor data quality during collection
           - Flag and resolve issues in real-time
           - Generate quality reports
        
        6. **📈 Monitoring Dashboard** – Track live productivity
           - Monitor survey completion rates
           - Track supervisor performance
           - Project timelines and resource needs
        
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
        "💡 **Pro Tip:** All features can be driven from the CLI. Run `rct-field-flow --help` "
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
        import sys
        from pathlib import Path
        
        # Add rct-design to path if needed
        rct_design_path = Path(__file__).parent / "rct-design"
        if str(rct_design_path) not in sys.path:
            sys.path.insert(0, str(rct_design_path))
        
        # Import wizard modules
        from wizard import main as wizard_main
        
        # Run the wizard
        wizard_main()
        
    except ImportError as e:
        st.error(f"Could not load RCT Design Wizard: {str(e)}")
        st.info("Please ensure the rct-design module is properly installed.")
        with st.expander("📋 Debug Info"):
            st.code(f"Import error: {str(e)}", language="python")
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

    if not submitted:
        return

    total_prop = sum(a.proportion for a in arms)
    if abs(total_prop - 1.0) > 0.01:
        st.error(f"Arm proportions must sum to 1.0 (current total: {total_prop:.2f}).")
        return

    if int(seed) <= 0:
        st.error("Random seed is required and must be a positive integer.")
        return

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
        return

    st.session_state.randomization_result = result
    st.session_state.case_data = result.assignments.copy()

    st.success(
        f"Randomization complete! Iterations: {result.iterations}. Best min p-value: {result.best_min_pvalue:.4f}"
    )

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

    csv_buffer = io.StringIO()
    result.assignments.to_csv(csv_buffer, index=False)
    
    # Generate code upfront (before any download buttons)
    python_code = generate_python_randomization_code(rand_config, method)
    stata_code = generate_stata_randomization_code(rand_config, method)
    
    # Downloads section with proper keys to prevent page refresh issues
    st.markdown("#### 📥 Download Results & Code")
    st.markdown("Download assignments, code, or analysis files.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📊 Download Assignments CSV",
            data=csv_buffer.getvalue(),
            file_name="randomized_assignments.csv",
            mime="text/csv",
            key="download_csv_assignments",
        )
    
    with col2:
        st.download_button(
            "🐍 Download Python Code",
            data=python_code,
            file_name="randomization_code.py",
            mime="text/x-python",
            key="download_python_code",
            help="Python script with your exact randomization parameters"
        )
    
    with col3:
        st.download_button(
            "📈 Download Stata Code",
            data=stata_code,
            file_name="randomization_code.do",
            mime="text/x-stata",
            key="download_stata_code",
            help="Stata do-file with your exact randomization parameters"
        )
    
    st.markdown("---")

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
                    key="power_compliance"
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
                        key="power_compliance"
                    )
            
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
                    compliance_rate=final_compliance
                )
            else:
                # For sample size calculation, set mde in assumptions
                # For cluster designs in sample_size mode, we need dummy num_clusters for initialization
                assumptions = PowerAssumptions(
                    alpha=alpha,
                    power=power,
                    baseline_mean=baseline_mean,
                    baseline_sd=baseline_sd,
                    outcome_type=outcome_type,
                    treatment_share=treatment_share,
                    design_type=design_type,
                    mde=target_mde,
                    num_clusters=int(cluster_size) if design_type == "cluster" else None,  # Dummy value, will be calculated
                    cluster_size=int(cluster_size) if design_type == "cluster" else None,
                    icc=icc if design_type == "cluster" else 0.0,
                    r_squared=final_r_squared,
                    compliance_rate=final_compliance
                )
            
            # Calculate results
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            if outcome_type == "binary":
                st.info("📌 **Binary Outcome**: MDE represents change in proportion (e.g., from 50% to 55% = MDE of 0.05 or 5 percentage points)")
            
            if calculation_mode == "mde":
                mde_result = calculate_mde(assumptions, sample_size=sample_size)
                mde = mde_result['mde_absolute']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if outcome_type == "binary":
                        st.metric("MDE (Percentage Points)", f"{mde*100:.1f}pp")
                        st.caption(f"Change from {baseline_mean*100:.1f}% to {(baseline_mean+mde)*100:.1f}%")
                    else:
                        st.metric("Minimum Detectable Effect", f"{mde:.3f}")
                        st.caption(f"{(mde / baseline_mean * 100):.2f}% of baseline mean")
                
                with col2:
                    st.metric("Sample Size", f"{int(sample_size):,}")
                    if design_type == "cluster":
                        st.caption(f"{int(num_clusters)} clusters × {int(cluster_size)} individuals")
                
                with col3:
                    st.metric("Power", f"{power*100:.0f}%")
                    st.caption(f"at α = {alpha}")
                
                # Summary
                with st.container():
                    st.info(
                        f"With **{int(sample_size):,} individuals** "
                        f"({'in ' + str(int(num_clusters)) + ' clusters' if design_type == 'cluster' else ''}), "
                        f"you can detect an effect of **{mde:.3f}** ({(mde / baseline_mean * 100):.2f}% of baseline) "
                        f"with **{power*100:.0f}% power** at **α = {alpha}**."
                    )
            
            else:  # sample_size mode
                sample_result = calculate_sample_size(assumptions)
                
                # Initialize variables
                required_n = None
                required_clusters = None
                total_n = None
                
                if design_type == "individual":
                    required_n = sample_result['sample_size']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Required Sample Size", f"{int(required_n):,}")
                    
                    with col2:
                        st.metric("Target MDE", f"{target_mde:.3f}")
                        st.caption(f"{(target_mde / baseline_mean * 100):.2f}% of baseline mean")
                    
                    with col3:
                        st.metric("Power", f"{power*100:.0f}%")
                        st.caption(f"at α = {alpha}")
                    
                    with st.container():
                        st.info(
                            f"You need **{int(required_n):,} individuals** to detect an effect of "
                            f"**{target_mde:.3f}** ({(target_mde / baseline_mean * 100):.2f}% of baseline) "
                            f"with **{power*100:.0f}% power** at **α = {alpha}**."
                        )
                
                else:  # cluster
                    required_clusters = sample_result['num_clusters']
                    total_n = sample_result['total_individuals']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Required Clusters", f"{required_clusters:,}")
                        st.caption(f"Total N = {total_n:,}")
                    
                    with col2:
                        st.metric("Target MDE", f"{target_mde:.3f}")
                        st.caption(f"{(target_mde / baseline_mean * 100):.2f}% of baseline mean")
                    
                    with col3:
                        st.metric("Cluster Size", f"{int(cluster_size):,}")
                        st.caption(f"ICC = {icc:.3f}")
                    
                    with st.container():
                        st.info(
                            f"You need **{required_clusters:,} clusters** ({total_n:,} individuals) "
                            f"to detect an effect of **{target_mde:.3f}** "
                            f"({(target_mde / baseline_mean * 100):.2f}% of baseline) "
                            f"with **{power*100:.0f}% power** at **α = {alpha}**."
                        )
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📈 Power Curves")
            
            tab1, tab2 = st.tabs(["Power vs Sample Size", "Cluster Analysis" if design_type == "cluster" else "Power Analysis"])
            
            with tab1:
                # Generate power curve
                if calculation_mode == "mde":
                    curve_assumptions = PowerAssumptions(
                        alpha=alpha,
                        power=power,
                        baseline_mean=baseline_mean,
                        baseline_sd=baseline_sd,
                        outcome_type=outcome_type,
                        treatment_share=treatment_share,
                        design_type=design_type,
                        effect_size=mde,
                        num_clusters=num_clusters if design_type == "cluster" else None,
                        cluster_size=cluster_size if design_type == "cluster" else None,
                        icc=icc if design_type == "cluster" else 0.0,
                        r_squared=final_r_squared,
                        compliance_rate=final_compliance
                    )
                else:
                    curve_assumptions = PowerAssumptions(
                        alpha=alpha,
                        power=power,
                        baseline_mean=baseline_mean,
                        baseline_sd=baseline_sd,
                        outcome_type=outcome_type,
                        treatment_share=treatment_share,
                        design_type=design_type,
                        effect_size=target_mde,
                        num_clusters=num_clusters if design_type == "cluster" else None,
                        cluster_size=cluster_size if design_type == "cluster" else None,
                        icc=icc if design_type == "cluster" else 0.0,
                        r_squared=final_r_squared,
                        compliance_rate=final_compliance
                    )
                
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
                
                # Add reference lines
                fig.add_hline(y=0.8, line_dash="dash", line_color="gray", 
                             annotation_text="80% Power", annotation_position="right")
                
                if calculation_mode == "mde":
                    if design_type == "cluster":
                        x_val = num_clusters
                    else:
                        x_val = sample_size
                    if x_val:
                        fig.add_vline(x=x_val, line_dash="dash", line_color="red",
                                     annotation_text=f"Current: {int(x_val):,}")
                else:
                    if design_type == "cluster":
                        x_val = required_clusters
                    else:
                        x_val = required_n
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
                if design_type == "cluster":
                    # Generate MDE table for different cluster sizes
                    st.markdown("#### MDE Table: Different Cluster Configurations")
                    
                    # Create assumptions for table generation
                    table_num_clusters = num_clusters if calculation_mode == "mde" else (required_clusters if required_clusters else 50)
                    table_assumptions = PowerAssumptions(
                        alpha=alpha,
                        power=power,
                        baseline_mean=baseline_mean,
                        baseline_sd=baseline_sd,
                        outcome_type=outcome_type,
                        treatment_share=treatment_share,
                        design_type="cluster",
                        num_clusters=table_num_clusters,
                        cluster_size=cluster_size,
                        icc=icc,
                        r_squared=final_r_squared,
                        compliance_rate=final_compliance
                    )
                    
                    mde_table_df = generate_cluster_size_table(
                        table_assumptions,
                        cluster_sizes=[10, 15, 20, 25, 30, 40, 50],
                        num_clusters_options=[20, 30, 40, 50, 60, 80, 100]
                    )
                    
                    # Pivot for better display
                    mde_table = mde_table_df.pivot(
                        index='cluster_size',
                        columns='num_clusters',
                        values='mde_absolute'
                    )
                    
                    st.dataframe(
                        mde_table.style.format("{:.3f}").background_gradient(cmap='RdYlGn_r', axis=None),
                        use_container_width=True
                    )
                    st.caption(
                        "Each cell shows the MDE for that cluster configuration. "
                        "Smaller MDEs (greener) are better."
                    )
                else:
                    st.info("Cluster size analysis is only available for cluster randomization designs.")
            
            # Code generation
            st.markdown("---")
            st.markdown("### 💻 Downloadable Code")
            
            code_tab1, code_tab2 = st.tabs(["Python", "Stata"])
            
            # Prepare results dict for code generation
            if calculation_mode == "mde":
                code_results = mde_result
            else:
                code_results = sample_result
            
            with code_tab1:
                python_code = generate_python_code(
                    curve_assumptions if calculation_mode == "mde" else assumptions,
                    code_results
                )
                st.code(python_code, language="python")
                st.download_button(
                    "Download Python Script",
                    data=python_code,
                    file_name="power_calculation.py",
                    mime="text/x-python"
                )
            
            with code_tab2:
                stata_code = generate_stata_code(
                    curve_assumptions if calculation_mode == "mde" else assumptions,
                    code_results
                )
                st.code(stata_code, language="stata")
                st.download_button(
                    "Download Stata Script",
                    data=stata_code,
                    file_name="power_calculation.do",
                    mime="text/plain"
                )
        
        except ValueError as e:
            st.error(f"❌ Calculation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")


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
                ["Upload CSV", "SurveyCTO API"],
                key="analysis_data_source",
                horizontal=True
            )
        
        if data_source == "Upload CSV":
            upload = st.file_uploader(
                "Upload endline data (CSV)",
                type="csv",
                key="analysis_upload",
                help="Upload your follow-up/endline survey data"
            )
            if upload:
                df = pd.read_csv(upload)
                st.session_state.analysis_data = df
                st.success(f"✅ Loaded {len(df):,} observations with {len(df.columns)} variables.")
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
                
                st.markdown("#### Analysis Options")
                col1, col2 = st.columns(2)
                with col1:
                    run_ate = st.checkbox("Average Treatment Effects (ATE)", value=True, key="run_ate")
                    run_balance = st.checkbox("Balance Table", value=True, key="run_balance")
                
                with col2:
                    run_heterogeneity = st.checkbox("Heterogeneity Analysis", value=False, key="run_het")
                    if run_heterogeneity:
                        moderator = st.selectbox(
                            "Moderator variable",
                            potential_controls,
                            key="analysis_moderator",
                            help="Variable to interact with treatment"
                        )
                    else:
                        moderator = None
                
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
                    
                    # Balance table
                    if run_balance and covariates:
                        st.markdown("### ⚖️ Balance Table")
                        st.markdown("Check if covariates are balanced across treatment arms.")
                        
                        balance_results = []
                        for cov in covariates:
                            if cov in numeric_cols:
                                try:
                                    # Run regression of covariate on treatment
                                    result = estimate_ate(df, cov, config=config)
                                    
                                    # Extract treatment arms and p-values
                                    for param in result.params.index:
                                        if param.startswith(f"C({treatment_col})"):
                                            p_value = result.pvalues[param]
                                            balance_results.append({
                                                "Covariate": cov,
                                                "Parameter": param,
                                                "Coefficient": result.params[param],
                                                "P-value": p_value,
                                                "Balanced": "✓" if p_value > 0.05 else "✗"
                                            })
                                except Exception as e:
                                    st.warning(f"Could not test balance for {cov}: {e}")
                        
                        if balance_results:
                            balance_df = pd.DataFrame(balance_results)
                            st.dataframe(balance_df, use_container_width=True)
                            
                            imbalanced = balance_df[balance_df["Balanced"] == "✗"]
                            if len(imbalanced) > 0:
                                st.warning(f"⚠️ {len(imbalanced)} covariate(s) show significant imbalance (p < 0.05)")
                            else:
                                st.success("✅ All covariates are balanced across treatment arms")
                    
                    # Average Treatment Effects
                    if run_ate:
                        st.markdown("### 📊 Average Treatment Effects (ATE)")
                        
                        all_results = []
                        
                        for outcome in outcomes:
                            st.markdown(f"#### {outcome}")
                            
                            try:
                                result = estimate_ate(df, outcome, covariates=covariates, config=config)
                                
                                # Extract results
                                st.markdown("**Regression Output:**")
                                
                                # Create results table
                                results_data = []
                                for param in result.params.index:
                                    results_data.append({
                                        "Parameter": param,
                                        "Coefficient": f"{result.params[param]:.4f}",
                                        "Std Error": f"{result.bse[param]:.4f}",
                                        "t-statistic": f"{result.tvalues[param]:.3f}",
                                        "P-value": f"{result.pvalues[param]:.4f}",
                                        "Significance": "***" if result.pvalues[param] < 0.01 else 
                                                       "**" if result.pvalues[param] < 0.05 else 
                                                       "*" if result.pvalues[param] < 0.10 else ""
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df, use_container_width=True)
                                st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.10")
                                
                                # Model statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R-squared", f"{result.rsquared:.4f}")
                                with col2:
                                    st.metric("Observations", f"{int(result.nobs):,}")
                                with col3:
                                    st.metric("F-statistic", f"{result.fvalue:.2f}")
                                
                                # Store for export
                                for param in result.params.index:
                                    if param.startswith(f"C({treatment_col})"):
                                        all_results.append({
                                            "Outcome": outcome,
                                            "Treatment": param,
                                            "Coefficient": result.params[param],
                                            "Std_Error": result.bse[param],
                                            "P_value": result.pvalues[param],
                                            "CI_Lower": result.conf_int().loc[param, 0],
                                            "CI_Upper": result.conf_int().loc[param, 1],
                                            "R_squared": result.rsquared,
                                            "N_obs": int(result.nobs)
                                        })
                                
                            except Exception as e:
                                st.error(f"Error analyzing {outcome}: {e}")
                        
                        # Export all results
                        if all_results:
                            st.markdown("---")
                            st.markdown("### 📥 Download Results")
                            
                            results_export = pd.DataFrame(all_results)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = results_export.to_csv(index=False)
                                st.download_button(
                                    "📥 Download as CSV",
                                    csv,
                                    "ate_results.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Excel export
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    results_export.to_excel(writer, index=False, sheet_name='ATE Results')
                                st.download_button(
                                    "📥 Download as Excel",
                                    buffer.getvalue(),
                                    "ate_results.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                    
                    # Heterogeneity Analysis
                    if run_heterogeneity and moderator:
                        st.markdown("### 🔍 Heterogeneity Analysis")
                        st.markdown(f"Treatment effects by **{moderator}**")
                        
                        for outcome in outcomes:
                            st.markdown(f"#### {outcome}")
                            
                            try:
                                result = heterogeneity_analysis(
                                    df, outcome, moderator, covariates=covariates, config=config
                                )
                                
                                # Show interaction effects
                                interaction_params = [p for p in result.params.index 
                                                     if f"C({treatment_col})" in p and f"C({moderator})" in p]
                                
                                if interaction_params:
                                    st.markdown("**Interaction Effects:**")
                                    
                                    interaction_data = []
                                    for param in interaction_params:
                                        interaction_data.append({
                                            "Interaction": param,
                                            "Coefficient": f"{result.params[param]:.4f}",
                                            "Std Error": f"{result.bse[param]:.4f}",
                                            "P-value": f"{result.pvalues[param]:.4f}",
                                            "Significance": "***" if result.pvalues[param] < 0.01 else 
                                                           "**" if result.pvalues[param] < 0.05 else 
                                                           "*" if result.pvalues[param] < 0.10 else ""
                                        })
                                    
                                    interaction_df = pd.DataFrame(interaction_data)
                                    st.dataframe(interaction_df, use_container_width=True)
                                    
                                    # Interpretation
                                    sig_interactions = [d for d in interaction_data if d["Significance"]]
                                    if sig_interactions:
                                        st.info(f"💡 Found {len(sig_interactions)} significant interaction effect(s). "
                                               f"Treatment effects differ by {moderator}.")
                                    else:
                                        st.info(f"No significant heterogeneity found by {moderator}.")
                                else:
                                    st.warning("No interaction terms found in the model.")
                                
                            except Exception as e:
                                st.error(f"Error in heterogeneity analysis for {outcome}: {e}")
        
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
                    except:
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
                except:
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
    """Render the User Information page - admin only with password protection."""
    
    st.title("👥 User Information")
    st.markdown("---")
    
    # Password protection
    if 'userinfo_authenticated' not in st.session_state:
        st.session_state.userinfo_authenticated = False
    
    if not st.session_state.userinfo_authenticated:
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
    
    # Check if user has an active session
    if 'temp_user' not in st.session_state:
        st.warning("⚠️ No active session data found. Users must be logged in for their data to appear here.")
        return
    
    username = st.session_state.temp_user
    org_type = st.session_state.get('temp_organization', 'Not specified')
    access_time = st.session_state.get('temp_access_time', datetime.now())
    duration = datetime.now() - access_time
    
    # Session Summary
    st.markdown("### 👤 Session Summary")
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
    
    # Activity Log
    st.markdown("### 📊 Activity Log")
    activity_log = st.session_state.get('activity_log', [])
    
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
    
    # Data Privacy Notice
    with st.expander("ℹ️ About User Data"):
        st.markdown("""
        **What's included in downloads:**
        - Session information (username, organization type, timestamps)
        - Activity log (pages visited, actions taken)
        - RCT Design responses (if user completed the design feature)
        - Randomization data (if user performed randomization)
        
        **Data retention:**
        - All data is stored only in the user's current browser session
        - Data is automatically cleared when user ends session or closes browser
        - No data is stored on servers permanently
        - Downloads are generated on-demand from session state
        
        **Administrator access:**
        - This page is password-protected for administrators only
        - Used for monitoring user activity and generating reports
        - All user data remains private and temporary
        """)
    
    st.markdown("---")
    st.caption("🔒 Administrator Access Only | Password: admin2025")


# ----------------------------------------------------------------------------- #
# TEMPORARY ACCESS SYSTEM & ACTIVITY LOGGING                                    #
# ----------------------------------------------------------------------------- #

import json
from io import BytesIO

# Pages that don't require access credentials
PUBLIC_PAGES = ["home"]

# Pages that require temporary access (excludes admin pages with their own auth)
PROTECTED_PAGES = ["design", "power", "random", "cases", "quality", "analysis", 
                   "backcheck", "reports", "monitor"]

# Admin pages with their own password protection (skip temporary access)
ADMIN_PAGES = ["facilitator", "userinfo"]


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
    if 'temp_user' in st.session_state:
        log_entry['username'] = st.session_state.temp_user
        log_entry['organization'] = st.session_state.temp_organization
    
    # Add additional details
    if details:
        log_entry['details'] = details
    
    st.session_state.activity_log.append(log_entry)
    
    # Auto-save session data after logging activity
    save_session_snapshot()


def save_session_snapshot():
    """Save a snapshot of current session data to session state for persistence"""
    if 'temp_user' not in st.session_state:
        return
    
    # Create a comprehensive snapshot
    snapshot = {
        'user_info': {
            'username': st.session_state.get('temp_user'),
            'organization': st.session_state.get('temp_organization'),
            'access_time': st.session_state.get('temp_access_time', datetime.now()).isoformat() if hasattr(st.session_state.get('temp_access_time', None), 'isoformat') else str(st.session_state.get('temp_access_time')),
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


def restore_session_if_available():
    """Restore session data from snapshot if available (after page reload)"""
    # Check if we have a saved snapshot but missing active session keys
    if 'session_snapshot' in st.session_state and 'temp_user' not in st.session_state:
        snapshot = st.session_state.session_snapshot
        
        # Restore user info
        if 'user_info' in snapshot:
            st.session_state.temp_user = snapshot['user_info'].get('username')
            st.session_state.temp_organization = snapshot['user_info'].get('organization')
            # Restore access time if available
            try:
                from dateutil import parser
                st.session_state.temp_access_time = parser.parse(snapshot['user_info'].get('access_time'))
            except:
                st.session_state.temp_access_time = datetime.now()
        
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
            'username': st.session_state.get('temp_user'),
        })


def get_session_data() -> dict:
    """Compile all session data for download"""
    data = {
        'session_info': {
            'username': st.session_state.get('temp_user', 'Anonymous'),
            'organization': st.session_state.get('temp_organization', 'Not specified'),
            'access_time': st.session_state.get('temp_access_time', datetime.now()).isoformat(),
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


def require_temp_access(page_name: str) -> bool:
    """
    Check if page requires temporary access and if user has it.
    Returns True if access is granted, False if access form should be shown.
    """
    # Public pages don't need access
    if page_name in PUBLIC_PAGES:
        return True
    
    # Check if user already has temporary access
    if 'temp_user' in st.session_state and st.session_state.temp_user:
        # Log page visit (except admin pages)
        if page_name not in ['userinfo', 'facilitator']:
            log_activity(f'visited_{page_name}_page')
        return True
    
    # Show temporary access form
    show_temp_access_form(page_name)
    return False


def show_temp_access_form(page_name: str):
    """Display temporary access form for protected pages."""
    
    st.markdown("""
    <style>
    .access-container {
        max-width: 500px;
        margin: 80px auto;
        padding: 40px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .access-title {
        color: #164a7f;
        text-align: center;
        margin-bottom: 10px;
        font-size: 2em;
    }
    .access-subtitle {
        color: #2fa6dc;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="access-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="access-title">🎯 RCT Field Flow</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="access-subtitle">Access Required: {page_name.replace("_", " ").title()}</p>', 
                unsafe_allow_html=True)
    
    st.info("👤 Please provide your details to access this feature. No registration required!")
    
    # Temporary access form
    with st.form("temp_access_form", clear_on_submit=False):
        st.markdown("##### 📝 Your Information")
        
        username = st.text_input(
            "Username *",
            placeholder="e.g., johndoe or jane.smith",
            help="Enter a username to identify your session",
            key="temp_name_input"
        )
        
        # Organization type selector
        org_type = st.selectbox(
            "Organization Type",
            options=[
                "Select...",
                "University/Academic Institution",
                "Research Organization",
                "NGO/Non-Profit",
                "Government Agency",
                "Private Company",
                "Independent Researcher",
                "Student",
                "Other"
            ],
            help="Select your organization type",
            key="temp_org_input"
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button(
                "✅ Continue to App",
                use_container_width=True,
                type="primary"
            )
        
        if submit:
            if username and username.strip():
                # Store temporary access credentials
                st.session_state.temp_user = username.strip()
                st.session_state.temp_organization = org_type if org_type != "Select..." else None
                st.session_state.temp_access_time = datetime.now()
                
                # Initialize activity log and log access
                init_activity_log()
                log_activity('user_access_granted', {
                    'username': username.strip(),
                    'organization': org_type,
                    'target_page': page_name
                })
                
                st.success(f"✅ Welcome, {username}! Redirecting...")
                st.rerun()
            else:
                st.error("⚠️ Please enter a username to continue")
    
    st.markdown("---")
    
    # Information box
    with st.expander("ℹ️ About Temporary Access"):
        st.markdown("""
        **How it works:**
        - No registration or account creation required
        - Enter a username to create a temporary session
        - Your session lasts until you close your browser
        - No passwords required or stored
        
        **Your privacy:**
        - We only store your username and organization type for the current session
        - No personal data is saved permanently
        - Session data is cleared when you end session or close browser
        - Organization type helps us understand our user community
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #666; font-size: 0.85rem;'>
        <p>RCT Field Flow | Developed by <strong>Aubrey Jolex</strong></p>
        <p>📧 <a href='mailto:aubreyjolex@gmail.com'>aubreyjolex@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)


def show_user_info_sidebar():
    """Display current user info and logout button in sidebar."""
    if 'temp_user' in st.session_state and st.session_state.temp_user:
        st.sidebar.markdown("---")
        
        # Show current user
        user_name = st.session_state.temp_user
        user_org = st.session_state.get('temp_organization', None)
        
        st.sidebar.markdown("### 👤 Current Session")
        st.sidebar.markdown(f"**Name:** {user_name}")
        if user_org:
            st.sidebar.markdown(f"**Org:** {user_org}")
        
        # Show session time
        if 'temp_access_time' in st.session_state:
            access_time = st.session_state.temp_access_time
            duration = datetime.now() - access_time
            minutes = int(duration.total_seconds() / 60)
            st.sidebar.caption(f"⏱️ Session: {minutes} min")
        
        # Show last save time
        if 'last_save_time' in st.session_state:
            last_save = st.session_state.last_save_time
            seconds_ago = int((datetime.now() - last_save).total_seconds())
            if seconds_ago < 60:
                st.sidebar.caption(f"💾 Auto-saved: {seconds_ago}s ago")
            else:
                minutes_ago = seconds_ago // 60
                st.sidebar.caption(f"💾 Auto-saved: {minutes_ago}m ago")
        
        # Logout button
        if st.sidebar.button("� End Session", use_container_width=True):
            # Log session end before clearing
            if 'activity_log' in st.session_state:
                log_activity('user_session_ended')
            
            # Clear temporary access and all session data
            keys_to_clear = [
                'temp_user', 'temp_organization', 'temp_code', 'temp_access_time',
                'baseline_data', 'randomization_result', 'case_data', 
                'quality_data', 'analysis_data', 'design_data',
                'design_workbook_responses', 'design_program_card', 'activity_log',
                'session_snapshot', 'last_save_time'
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            
            st.success("👋 Session ended. Redirecting to home...")
            st.rerun()


# ----------------------------------------------------------------------------- #
# MAIN                                                                          #
# ----------------------------------------------------------------------------- #


def main() -> None:
    # Initialize and recover session if available
    restore_session_if_available()
    
    # Visible navigation menu for users
    nav = {
        "home": "🏠 Home",
        "design": "🎯 RCT Design",
        "power": "⚡ Power Calculations",
        "random": "🎲 Randomization",
        "cases": "📋 Case Assignment",
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
    show_user_info_sidebar()

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

    # Check temporary access for protected pages (skip admin pages)
    if page in PROTECTED_PAGES and page not in ADMIN_PAGES:
        if not require_temp_access(page):
            return  # Show access form, don't render page
    
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
