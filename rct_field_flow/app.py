from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
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


# ----------------------------------------------------------------------------- #
# HOME                                                                          #
# ----------------------------------------------------------------------------- #


def render_home() -> None:
    st.title("📊 RCT Field Flow")
    st.markdown(
        """
        Integrated toolkit for randomization, SurveyCTO case management,
        quality checks, and live monitoring.
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
        "Upload baseline data, configure treatment arms, and run rerandomization with balance checks."
    )

    default_config = load_default_config().get("randomization", {})
    df = st.session_state.baseline_data

    upload = st.file_uploader("Upload baseline data (CSV)", type="csv", key="randomization_upload")
    if upload:
        df = pd.read_csv(upload)
        st.session_state.baseline_data = df
        st.success(f"Loaded {len(df):,} observations • {len(df.columns)} columns.")

    if df is None:
        st.info("Please upload a baseline CSV to configure randomization.")
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
        strata=strata if strata else None,
        cluster=cluster_col,
        balance_covariates=balance_covariates if balance_covariates else None,
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
    st.markdown("Assign randomized cases to SurveyCTO teams and produce upload-ready rosters.")

    df = st.session_state.case_data
    upload = st.file_uploader("Upload randomized data (CSV)", type="csv", key="case_upload")
    if upload:
        df = pd.read_csv(upload)
        st.session_state.case_data = df
        st.success(f"Loaded {len(df):,} rows for case assignment.")

    if df is None:
        st.info("Provide randomized data via the previous tab or upload a CSV here.")
        return

    st.dataframe(df.head(10), use_container_width=True)

    default_case_cfg = load_default_config().get("case_assignment", {})
    config_text = st.text_area(
        "Case assignment configuration (YAML)",
        value=yaml_dump(default_case_cfg),
        height=260,
        key="case_config_text",
    )

    if st.button("Generate SurveyCTO roster", type="primary"):
        try:
            config = yaml_load(config_text)
            roster = assign_cases(df, config)
        except Exception as exc:
            st.error(f"Assignment failed: {exc}")
            return

        st.success(f"Generated roster with {len(roster):,} cases.")
        st.dataframe(roster.head(20), use_container_width=True)

        csv_buffer = io.StringIO()
        roster.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download roster CSV",
            data=csv_buffer.getvalue(),
            file_name="surveycto_case_roster.csv",
            mime="text/csv",
        )


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
            "🔢 Outlier Detection",
            "⏱️ Duration Checks",
            "👥 Duplicate Detection",
            "✅ Intervention Fidelity"
        ])
        
        # Tab 1: Outlier Detection
        with tab1:
            st.markdown("**Detect outliers in numeric variables using IQR or standard deviation methods**")
            
            if not numeric_cols:
                st.warning("⚠️ No numeric columns automatically detected. Select columns manually below.")
                st.info("💡 **Tip:** Columns may need to be converted to numeric. The tool will attempt automatic conversion.")
            
            outlier_vars = st.multiselect(
                "Select variables to check for outliers",
                options=all_cols if not numeric_cols else numeric_cols,
                help="Choose variables to analyze. Must contain numeric values.",
                key="outlier_vars"
            )
            
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
            flagged_records = []
            
            # 1. Outlier detection
            if outlier_vars:
                with st.spinner("Detecting outliers..."):
                    for var in outlier_vars:
                        # Try to convert to numeric
                        try:
                            df[var] = pd.to_numeric(df[var], errors='coerce')
                        except Exception:
                            st.warning(f"⚠️ Could not convert '{var}' to numeric. Skipping...")
                            continue
                        
                        # Check if we have numeric values after conversion
                        if not pd.api.types.is_numeric_dtype(df[var]):
                            st.warning(f"⚠️ Column '{var}' does not contain numeric values. Skipping...")
                            continue
                        
                        if group_by_outlier != "None":
                            for group_val, group_data in df.groupby(group_by_outlier):
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
                            values = df[var].dropna()
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
                                
                                outliers = df[(df[var] < lower) | (df[var] > upper)]
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
                    # Try to convert to numeric
                    try:
                        df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
                    except Exception:
                        st.error(f"❌ Could not convert '{duration_col}' to numeric. Please select a numeric column.")
                        st.stop()
                    
                    if not pd.api.types.is_numeric_dtype(df[duration_col]):
                        st.error(f"❌ Column '{duration_col}' does not contain numeric values. Please select a different column.")
                        st.stop()
                    
                    durations = df[duration_col].copy()
                    
                    if duration_unit == "Minutes":
                        durations = durations * 60
                    
                    if check_method == "Quantile-based":
                        lower_thresh = durations.quantile(speed_quantile / 100)
                        upper_thresh = durations.quantile(slow_quantile / 100)
                    else:
                        lower_thresh = min_duration if duration_unit == "Seconds" else min_duration * 60
                        upper_thresh = max_duration if duration_unit == "Seconds" else max_duration * 60
                    
                    fast_surveys = df[df[duration_col] < lower_thresh]
                    slow_surveys = df[df[duration_col] > upper_thresh]
                    
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
                    duplicates = df[df.duplicated(subset=duplicate_keys, keep=False)]
                    
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
                    invalid_treatments = df[~df[treatment_col].isin(expected_vals)]
                    
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
                if group_results_by != "None":
                    flagged_df = flagged_df.merge(
                        df[[group_results_by]],
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
