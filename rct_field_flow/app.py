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
            method_options = ["simple", "stratified", "cluster"]
            default_method = default_config.get("method", "simple")
            method_index = method_options.index(default_method) if default_method in method_options else 0
            method = st.selectbox(
                "Method",
                method_options,
                index=method_index,
                key="rand_method",
            )
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
            if method == "stratified":
                strata = st.multiselect(
                    "Strata columns",
                    available_cols,
                    key="rand_strata",
                )
            if method == "cluster":
                cluster_col = st.selectbox(
                    "Cluster column",
                    available_cols,
                    key="rand_cluster",
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

    rand_config = RandomizationConfig(
        id_column=id_column,
        treatment_column=treatment_column or "treatment",
        method=method,  # type: ignore[arg-type]
        arms=arms,
        strata=strata,
        cluster=cluster_col,
        balance_covariates=balance_covariates,
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
    st.markdown("Apply duration, duplicate, and intervention checks to submission data.")

    df = st.session_state.quality_data
    upload = st.file_uploader("Upload submissions data (CSV)", type="csv", key="quality_upload")
    if upload:
        df = pd.read_csv(upload)
        st.session_state.quality_data = df
        st.success(f"Loaded {len(df):,} submissions.")

    if df is None:
        st.info("Upload a submissions CSV to continue.")
        return

    st.dataframe(df.head(10), use_container_width=True)

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
