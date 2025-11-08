from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
import yaml

try:
    from .assign_cases import assign_cases
    from .flag_quality import QualityResults, flag_all
    from .randomize import RandomizationConfig, RandomizationResult, Randomizer, TreatmentArm
    from .monitor import (
        load_config as mon_load_config,
        load_submissions as mon_load_submissions,
        prepare_data as mon_prepare_data,
        render_dashboard as mon_render_dashboard,
    )
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
    from rct_field_flow.monitor import (  # type: ignore
        load_config as mon_load_config,
        load_submissions as mon_load_submissions,
        prepare_data as mon_prepare_data,
        render_dashboard as mon_render_dashboard,
    )
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

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


def load_default_config() -> Dict:
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def yaml_dump(data: Dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def yaml_load(text: str) -> Dict:
    return yaml.safe_load(text) if text.strip() else {}


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
        Integrated toolkit for randomization, SurveyCTO cases assignement,
        data quality checks, and live monitoring.
        """
    )

    st.markdown(
        """
        **Workflow overview**

        1. 🎲 Randomization – configure arms, strata, and rerandomization.
        2. 📋 Case Assignment – build SurveyCTO-ready case rosters.
        3. ✅ Quality Checks – apply speed/outlier/duplicate checks.
        4. 📈 Monitoring Dashboard – track productivity, supervisor roll-ups, and projected timelines.
        """
    )

    st.markdown("---")
    st.info(
        "Tip: All features can also be driven from the CLI. Run `rct-field-flow --help` "
        "to explore commands and options."
    )


# ----------------------------------------------------------------------------- #
# RANDOMIZATION                                                                 #
# ----------------------------------------------------------------------------- #


def render_randomization() -> None:
    st.title("🎲 Randomization")
    st.markdown(
        "Upload randomization data, configure treatment arms, and run rerandomization with balance checks."
    )

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
    st.download_button(
        "Download assignments CSV",
        data=csv_buffer.getvalue(),
        file_name="randomized_assignments.csv",
        mime="text/csv",
    )

    # Generate and offer code downloads for transparency
    st.markdown("#### 📥 Download Randomization Code")
    st.markdown("Download the actual code that ran your randomization to share with PIs and collaborators.")
    
    col1, col2 = st.columns(2)
    with col1:
        python_code = generate_python_randomization_code(rand_config, method)
        st.download_button(
            "📄 Download Python Code",
            data=python_code,
            file_name="randomization_code.py",
            mime="text/x-python",
            help="Python script with your exact randomization parameters"
        )
    with col2:
        stata_code = generate_stata_randomization_code(rand_config, method)
        st.download_button(
            "📄 Download Stata Code",
            data=stata_code,
            file_name="randomization_code.do",
            mime="text/x-stata",
            help="Stata do-file with your exact randomization parameters"
        )

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
# CASE ASSIGNMENT                                                               #
# ----------------------------------------------------------------------------- #


def render_case_assignment() -> None:
    st.title("📋 Case Assignment")
    st.markdown("Assign interview cases to SurveyCTO teams and produce upload-ready rosters.")

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

        if st.button("Generate SurveyCTO roster", type="primary"):
            try:
                config = yaml_load(config_text)
                roster = assign_cases(df, config)
            except Exception as exc:
                st.error(f"Assignment failed: {exc}")
                return

            st.success(f"✅ Generated roster with {len(roster):,} cases.")
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
            
            # Form IDs by treatment arm
            st.markdown("##### SurveyCTO Form IDs")
            st.markdown("Specify which form(s) each treatment arm should use.")
            
            treatment_arms = df[treatment_column].dropna().unique().tolist()
            form_ids = {}
            
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                default_forms = st.text_input(
                    "Default form ID(s)",
                    value="follow_up",
                    key="case_default_forms",
                    help="Comma-separated form IDs for cases not matching specific treatments"
                )
                form_ids["default"] = [f.strip() for f in default_forms.split(",") if f.strip()]
            
            with form_col2:
                form_separator = st.text_input(
                    "Form ID separator",
                    value=",",
                    key="case_form_separator",
                    help="Character to separate multiple form IDs"
                )
            
            for arm in treatment_arms:
                arm_forms = st.text_input(
                    f"Form ID(s) for '{arm}'",
                    key=f"case_forms_{arm}",
                    placeholder="leave blank to use default",
                    help=f"Comma-separated form IDs for {arm} cases"
                )
                if arm_forms:
                    form_ids[str(arm)] = [f.strip() for f in arm_forms.split(",") if f.strip()]
            
            # Additional columns
            st.markdown("##### Additional Roster Columns")
            additional_columns = st.multiselect(
                "Include these columns in the roster",
                [col for col in available_cols if col not in [case_id_column, treatment_column]],
                key="case_additional_columns",
                help="Extra columns to include in the SurveyCTO case roster"
            )
            
            # Submit button
            submitted = st.form_submit_button("Generate SurveyCTO Roster", type="primary")
        
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
                
                st.success(f"✅ Generated roster with {len(roster):,} cases!")
                
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
                
                st.markdown("#### Roster Preview")
                st.dataframe(roster.head(20), use_container_width=True)
                
                # Store roster in session state for upload
                st.session_state['generated_roster'] = roster
                
                # Download and Upload options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_buffer = io.StringIO()
                    roster.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📥 Download Roster CSV",
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
            st.error(f"❌ Roster missing required columns: {missing_cols}")
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
# MONITORING DASHBOARD                                                          #
# ----------------------------------------------------------------------------- #


def render_monitoring() -> None:
    st.title("📈 Monitoring Dashboard")
    cfg = mon_load_config()

    source = st.radio(
        "Data source",
        ["Use project config", "Upload CSV", "SurveyCTO API"],
        key="monitor_data_source",
    )

    data: pd.DataFrame | None = None

    if source == "Use project config":
        try:
            submissions = mon_load_submissions(cfg)
            data = mon_prepare_data(submissions, cfg)
        except Exception as exc:  # pragma: no cover
            st.error(f"Couldn't load monitoring components using project config: {exc}")
            return
    elif source == "Upload CSV":
        upload = st.file_uploader("Upload submissions CSV", type="csv", key="monitor_csv_upload")
        if upload:
            data = pd.read_csv(upload, sep=None, engine="python")
            st.session_state["monitor_upload_df"] = data
        else:
            data = st.session_state.get("monitor_upload_df")
        if data is None:
            st.info("Upload a CSV file to continue.")
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

        if st.button("Fetch SurveyCTO submissions", key="monitor_fetch_api"):
            if not all([server, username, password, form_id]):
                st.error("Server, username, password, and form ID are required.")
            else:
                try:
                    client = SurveyCTO(server=server, username=username, password=password)
                    api_df = client.get_submissions(form_id)
                    st.session_state["monitor_api_df"] = api_df
                    st.success(f"Fetched {len(api_df):,} submissions from SurveyCTO.")
                except Exception as exc:
                    st.error(f"Failed to fetch SurveyCTO submissions: {exc}")
        data = st.session_state.get("monitor_api_df")
        if data is None:
            st.info("Enter credentials and click the fetch button to load live data.")
            return

    if data is None or data.empty:
        st.warning("No submissions available. Check your data source.")
        return

    mon_render_dashboard(data, cfg)


# ----------------------------------------------------------------------------- #
# MAIN                                                                          #
# ----------------------------------------------------------------------------- #


def main() -> None:
    nav = {
        "home": "🏠 Home",
        "random": "🎲 Randomization",
        "cases": "📋 Case Assignment",
        "quality": "✅ Quality Checks",
        "monitor": "📈 Monitoring Dashboard",
    }
    page = st.sidebar.radio(
        "Navigation",
        options=list(nav.keys()),
        format_func=lambda key: nav[key],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear cached data"):
        for key in ["baseline_data", "randomization_result", "case_data", "quality_data"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

    if page == "home":
        render_home()
    elif page == "random":
        render_randomization()
    elif page == "cases":
        render_case_assignment()
    elif page == "quality":
        render_quality_checks()
    elif page == "monitor":
        render_monitoring()


if __name__ == "__main__":
    main()
