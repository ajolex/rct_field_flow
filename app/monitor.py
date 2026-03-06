from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from .surveycto import SurveyCTO

try:
    st.set_page_config(page_title="RCT Field Flow", layout="wide")
except Exception:
    # Page config may already be set when embedded inside another Streamlit app
    pass


@st.cache_data(ttl=1800)
def load_config() -> dict:
    with open("rct_field_flow/config/default.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand_env(raw: str | None) -> str | None:
    if raw is None:
        return None
    expanded = os.path.expandvars(raw)
    if isinstance(expanded, str) and expanded.startswith("${") and expanded.endswith("}"):
        return None
    expanded = expanded.strip() if isinstance(expanded, str) else expanded
    return expanded or None


def get_surveycto_client(cfg: dict) -> SurveyCTO | None:
    creds = cfg.get("surveycto", {})
    server = expand_env(os.getenv("SCTO_SERVER", creds.get("server")))
    username = expand_env(os.getenv("SCTO_USER", creds.get("username")))
    password = expand_env(os.getenv("SCTO_PASS", creds.get("password")))
    if not all([server, username, password]):
        return None
    return SurveyCTO(server=server, username=username, password=password)


@st.cache_data(ttl=1200, show_spinner=True)
def load_submissions(cfg: dict) -> pd.DataFrame:
    monitor_cfg = cfg.get("monitoring", {})
    submissions_path = expand_env(monitor_cfg.get("submissions_path"))
    if submissions_path and os.path.exists(submissions_path):
        return pd.read_csv(submissions_path, sep=None, engine="python")

    client = get_surveycto_client(cfg)
    form_id = expand_env(cfg.get("surveycto", {}).get("form_id"))
    if client and form_id:
        return client.get_submissions(form_id)
    return pd.DataFrame()


def prepare_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df
    monitor_cfg = cfg.get("monitoring", {})
    ts_col = monitor_cfg.get("submission_timestamp_column", "submission_date")
    
    # Check if the timestamp column exists, if not, try to find a suitable column
    if ts_col not in df.columns:
        # Look for common timestamp column names
        possible_cols = [
            "submission_date", "SubmissionDate", "submissiondate",
            "starttime", "start", "endtime", "end",
            "CompletedDate", "completed_date", "timestamp"
        ]
        found_col = None
        for col in possible_cols:
            if col in df.columns:
                found_col = col
                break
        if not found_col:
            # If no suitable column found, raise an error with helpful message
            raise KeyError(
                f"Column '{ts_col}' not found in data. "
                f"Available columns: {', '.join(df.columns)}. "
                f"Please set 'monitoring.submission_timestamp_column' in your config to match your data."
            )
        ts_col = found_col
    
    if ts_col not in df.columns:
        # Leave data unchanged; column mapping will allow manual selection
        return df.copy()

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[df[ts_col].notna()]
    df["submission_date_only"] = df[ts_col].dt.date
    return df


def projected_end_date(total_completed: int, target: int | None, daily_rate: float, work_days_per_week: int) -> str:
    if not target:
        return "n/a"
    remaining = max(0, target - total_completed)
    if remaining == 0 or daily_rate <= 0:
        return "complete"
    work_factor = work_days_per_week / 7 if work_days_per_week else 1
    adjusted_rate = daily_rate * work_factor
    if adjusted_rate <= 0:
        return "n/a"
    days_left = remaining / adjusted_rate
    end_date = date.today() + timedelta(days=round(days_left))
    return end_date.strftime("%Y-%m-%d")


def render_dashboard(df: pd.DataFrame, cfg: dict) -> None:
    monitor_cfg = cfg.get("monitoring", {})

    # Column mapping
    st.markdown("---")
    st.markdown("### Column Mapping")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        date_col = st.selectbox("Submission Date Column", df.columns)
    with col2:
        treatment_col = st.selectbox("Treatment Column", df.columns)
    with col3:
        enumerator_col = st.selectbox("Enumerator Column", df.columns)
    with col4:
        community_col = st.selectbox("Community Column", df.columns)

    opt1, opt2 = st.columns(2)
    with opt1:
        duration_candidates = [c for c in df.columns if "dur" in c.lower() or "time" in c.lower()] or list(df.columns)
        duration_col = st.selectbox("Duration Column (optional)", ["<none>"] + duration_candidates)
        if duration_col == "<none>":
            duration_col = None
    with opt2:
        supervisor_candidates = [c for c in df.columns if "super" in c.lower() or "team" in c.lower() or "lead" in c.lower()]
        # If no matching columns found, show all columns
        if not supervisor_candidates:
            supervisor_candidates = list(df.columns)
        supervisor_col = st.selectbox("Supervisor Column (optional)", ["<none>"] + supervisor_candidates)
        if supervisor_col == "<none>":
            supervisor_col = None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()]
    df["date_only"] = df[date_col].dt.date
    if df.empty:
        st.warning("No records after parsing the date column.")
        return

    total_completed = len(df)
    today = date.today()
    todays = df[df["date_only"] == today]
    daily_counts = df.groupby("date_only").size()
    daily_rate = daily_counts.mean() if not daily_counts.empty else 0
    target_n = cfg.get("target_n")
    work_days_default = monitor_cfg.get("work_days_per_week", 6)

    st.markdown("---")
    st.markdown("### 📈 Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Completes", f"{total_completed:,}")
    col2.metric("Today", f"{len(todays):,}")
    col3.metric("Daily Average", f"{daily_rate:.1f}")
    col4.metric("Active Days", df["date_only"].nunique())

    # Time series
    st.markdown("#### Submissions Over Time")
    daily_counts_df = daily_counts.reset_index(name="count")
    fig = px.line(
        daily_counts_df,
        x="date_only",
        y="count",
        title="Daily Submissions",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Enumerator productivity table
    st.markdown("#### Enumerator Productivity")
    pivot = df.groupby([enumerator_col, "date_only"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    totals = pivot.sum(axis=1)

    avg_minutes = None
    if duration_col and duration_col in df.columns:
        dur = pd.to_numeric(df[duration_col], errors="coerce")
        if dur.dropna().mean() > 120:
            dur = dur / 60.0
        avg_minutes = (
            df.assign(_dur=dur)
            .groupby(enumerator_col)["_dur"]
            .mean()
            .round(0)
        )

    table = pivot.copy()
    table.insert(0, "Total submissions", totals)
    if avg_minutes is not None:
        table.insert(1, "Avg duration (min)", avg_minutes.reindex(table.index))

    avg_per_enum = pivot.mean(axis=0).round(2)
    total_per_day = pivot.sum(axis=0)
    date_cols = list(pivot.columns)

    def _top_row(template: dict[str, object]) -> pd.Series:
        return pd.Series(template, index=table.columns)

    row_avg = {col: "" for col in table.columns}
    for d in date_cols:
        row_avg[d] = avg_per_enum.get(d, "")
    top_avg = _top_row(row_avg)
    top_avg.name = "Avg per Enumerator"

    row_total = {col: "" for col in table.columns}
    row_total["Total submissions"] = int(totals.sum())
    for d in date_cols:
        row_total[d] = int(total_per_day.get(d, 0))
    top_total = _top_row(row_total)
    top_total.name = "Total"

    styled_supervisor_rows: set[str] = set()
    table_body = table
    if supervisor_col and supervisor_col in df.columns:
        enum_sup = (
            df[[enumerator_col, supervisor_col]]
            .dropna()
            .drop_duplicates(subset=[enumerator_col])
        )
        sup_groups = enum_sup.groupby(supervisor_col)[enumerator_col].apply(list)
        ordered_rows = []
        for sup, enums in sup_groups.items():
            sup_counts = pivot.loc[pivot.index.isin(enums)].sum(axis=0)
            sup_total = sup_counts.sum()
            if avg_minutes is not None:
                weights = df[df[enumerator_col].isin(enums)].groupby(enumerator_col).size()
                if not weights.empty:
                    weighted = (
                        avg_minutes.reindex(weights.index).fillna(0) * weights
                    ).sum()
                    sup_avg = round(weighted / weights.sum(), 0)
                else:
                    sup_avg = ""
            else:
                sup_avg = ""
            row = pd.Series(
                {"Total submissions": sup_total},
                name=f"SUP: {sup}",
            )
            if avg_minutes is not None:
                row["Avg duration (min)"] = sup_avg
            for d in date_cols:
                row[d] = sup_counts.get(d, 0)
            ordered_rows.append(row)
            styled_supervisor_rows.add(row.name)
            for e in enums:
                if e in table.index:
                    ordered_rows.append(table.loc[e])
        if ordered_rows:
            table_body = pd.DataFrame(ordered_rows)

    full_table = pd.concat([top_avg.to_frame().T, top_total.to_frame().T, table_body])

    def fmt_int(x: object) -> object:
        try:
            if pd.isna(x):
                return ""
            return f"{int(round(float(x))):d}"
        except Exception:
            return x

    styler = full_table.style
    fmt_map: dict[str, object] = {}
    if "Total submissions" in full_table.columns:
        fmt_map["Total submissions"] = fmt_int
    if "Avg duration (min)" in full_table.columns:
        fmt_map["Avg duration (min)"] = fmt_int
    for d in date_cols:
        if d in full_table.columns:
            fmt_map[d] = fmt_int
    if fmt_map:
        styler = styler.format(fmt_map)

    if "Avg per Enumerator" in full_table.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Avg per Enumerator"], :],
            **{"background-color": "#2e7d32", "color": "white"},
        )
        styler = styler.format(
            lambda x: f"{float(x):.2f}" if isinstance(x, (int, float, np.floating)) else x,
            subset=pd.IndexSlice[["Avg per Enumerator"], date_cols],
        )

    if styled_supervisor_rows:
        styler = styler.set_properties(
            subset=pd.IndexSlice[list(styled_supervisor_rows), :],
            **{"background-color": "#fff7ae"},
        )

    if "Avg duration (min)" in full_table.columns:
        styler = styler.set_properties(
            subset=pd.IndexSlice[:, ["Avg duration (min)"]],
            **{"background-color": "#dbeafe"},
        )
        try:
            col_vals = pd.to_numeric(full_table["Avg duration (min)"], errors="coerce")
            mask = col_vals < 60
            idx = full_table.index[mask.fillna(False)]
            styler = styler.set_properties(
                subset=pd.IndexSlice[idx, ["Avg duration (min)"]],
                **{"background-color": "#fecaca"},
            )
        except Exception:
            pass

    st.dataframe(styler, use_container_width=True)
    csv_prod = full_table.reset_index().rename(columns={"index": "enumerator"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Productivity CSV",
        data=csv_prod,
        file_name="enumerator_productivity.csv",
        mime="text/csv",
    )

    # Targets and timeline summary
    st.markdown("---")
    st.markdown("### Targets and Timeline")

    arms = list(df[treatment_col].dropna().unique())
    if "targets_by_arm" not in st.session_state:
        st.session_state.targets_by_arm = {}
    for arm in arms:
        st.session_state.targets_by_arm.setdefault(arm, 0)

    with st.expander("Set targets and dates", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            start_default = df["date_only"].min()
            start_date = st.date_input("Start date of data collection", value=start_default)
        with c2:
            today_display = date.today()
            st.date_input("Date today", value=today_display, disabled=True)

        weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        default_work = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] if work_days_default == 6 else weekday_order[:-1]
        work_days = st.multiselect(
            "Working days (weekmask)",
            options=weekday_order,
            default=default_work,
        )
        weekmask = " ".join(work_days) if work_days else "Mon Tue Wed Thu Fri Sat"

        cols = st.columns(max(1, len(arms))) if arms else []
        for idx, arm in enumerate(arms):
            with cols[idx % len(cols)]:
                st.session_state.targets_by_arm[arm] = st.number_input(
                    f"Target – {arm}",
                    min_value=0,
                    value=int(st.session_state.targets_by_arm.get(arm, 0)),
                    step=1,
                )

    completed = df[treatment_col].value_counts().reindex(arms, fill_value=0)
    targets = pd.Series({arm: int(st.session_state.targets_by_arm.get(arm, 0)) for arm in arms})
    pct = (completed / targets.replace(0, np.nan) * 100).round(2)
    comp_table = pd.DataFrame(
        {"Target": targets, "Completed": completed, "%age Completed": pct}
    )
    total_row = pd.DataFrame(
        {
            "Target": [targets.sum()],
            "Completed": [completed.sum()],
            "%age Completed": [
                (completed.sum() / targets.sum() * 100) if targets.sum() else np.nan
            ],
        },
        index=["Total"],
    )
    comp_table = pd.concat([comp_table, total_row])
    st.dataframe(
        comp_table.style.format({"%age Completed": lambda x: f"{x:.2f}%" if pd.notna(x) else ""}),
        use_container_width=True,
    )
    targets_csv = comp_table.reset_index().rename(columns={"index": "arm"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Targets CSV",
        data=targets_csv,
        file_name="targets_summary.csv",
        mime="text/csv",
    )

    weekmask_np = weekmask if weekmask else "Mon Tue Wed Thu Fri Sat"
    try:
        start_np = np.datetime64(start_date)
        end_np = np.datetime64(today) + np.timedelta64(1, "D")
        field_days = int(np.busday_count(start_np, end_np, weekmask=weekmask_np))
    except Exception:
        field_days = 0

    colA, colB = st.columns(2)
    with colA:
        info_table = pd.DataFrame(
            {
                "": [
                    "Start date of data collection",
                    "Date today",
                    "Field collection days",
                    "TOTAL WORK DAYS",
                ],
                "Value": [
                    start_date.isoformat(),
                    today.isoformat(),
                    field_days,
                    field_days,
                ],
            }
        )
        st.dataframe(info_table, hide_index=True, use_container_width=True)

    with colB:
        remaining = max(0, int(targets.sum()) - int(completed.sum())) if len(arms) else 0
        days_to_finish = remaining / daily_rate if daily_rate > 0 else np.nan
        weeks_to_finish = days_to_finish / len(work_days) if work_days and not np.isnan(days_to_finish) else np.nan
        if days_to_finish and not np.isnan(days_to_finish):
            try:
                projected_np = np.busday_offset(
                    np.datetime64(today),
                    int(np.ceil(days_to_finish)),
                    weekmask=weekmask_np,
                )
                projected_end = pd.Timestamp(projected_np).date().isoformat()
            except Exception:
                projected_end = ""
        else:
            projected_end = ""
        summary_tbl = pd.DataFrame(
            {
                "": [
                    "Average productivity per day",
                    "Days to finish from today",
                    "Weeks to finish from today",
                    "Projected end date",
                ],
                "Value": [
                    round(daily_rate, 2),
                    round(days_to_finish, 2) if days_to_finish and not np.isnan(days_to_finish) else "",
                    round(weeks_to_finish, 2) if weeks_to_finish and not np.isnan(weeks_to_finish) else "",
                    projected_end,
                ],
            }
        )
        st.dataframe(summary_tbl, hide_index=True, use_container_width=True)

    # Additional charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 By Treatment Arm")
        treatment_counts = df[treatment_col].value_counts()
        fig = px.pie(
            values=treatment_counts.values,
            names=treatment_counts.index,
            title="Distribution by Treatment",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### 👥 Top Enumerators")
        enum_counts = df[enumerator_col].value_counts().head(10)
        fig = px.bar(
            x=enum_counts.values,
            y=enum_counts.index,
            orientation="h",
            title="Top 10 Enumerators",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Enumerator Productivity (Last 7 Days)")
    recent_cutoff = today - timedelta(days=6)
    recent = df[df["date_only"] >= recent_cutoff]
    if recent.empty:
        st.info("No submissions in the past 7 days.")
    else:
        prod_recent = (
            recent.groupby([enumerator_col, "date_only"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            prod_recent,
            x="date_only",
            y="count",
            color=enumerator_col,
            barmode="group",
            title="Daily Submissions by Enumerator",
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    cfg = load_config()
    submissions = load_submissions(cfg)
    data = prepare_data(submissions, cfg)

    if data.empty:
        st.warning("No submissions available. Check SurveyCTO credentials or data path.")
        return

    render_dashboard(data, cfg)


if __name__ == "__main__":
    main()
