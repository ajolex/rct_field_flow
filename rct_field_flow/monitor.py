from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from .surveycto import SurveyCTO

st.set_page_config(page_title="RCT Field Flow", layout="wide")


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
        return pd.read_csv(submissions_path)

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
    treatment_col = monitor_cfg.get("treatment_column", "treatment")
    enumerator_col = monitor_cfg.get("enumerator_column", "enumerator")
    community_col = monitor_cfg.get("community_column")
    target_n = cfg.get("target_n")
    work_days = monitor_cfg.get("work_days_per_week", 6)

    total_completed = len(df)
    today = date.today()
    todays = df[df["submission_date_only"] == today]
    daily_counts = df.groupby("submission_date_only").size()
    daily_rate = daily_counts.mean() if not daily_counts.empty else 0

    st.title("RCT Field Flow — Live Monitoring")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total completes", f"{total_completed:,}")
    col2.metric("Today", f"{len(todays):,}")
    col3.metric(
        "Projected end date",
        projected_end_date(total_completed, target_n, daily_rate, work_days),
    )

    st.subheader("Enumerator productivity (past 7 days)")
    recent_cutoff = today - timedelta(days=6)
    recent = df[df["submission_date_only"] >= recent_cutoff]
    if recent.empty:
        st.info("No submissions in the past 7 days.")
    else:
        prod = (
            recent.groupby([enumerator_col, "submission_date_only"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            prod,
            x="submission_date_only",
            y="count",
            color=enumerator_col,
            barmode="group",
            title="Daily submissions by enumerator",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Progress by treatment arm")
    if treatment_col in df.columns:
        progress = (
            df.groupby([treatment_col, "submission_date_only"])
            .size()
            .groupby(level=0)
            .cumsum()
            .reset_index(name="cumulative")
        )
        fig = px.line(
            progress,
            x="submission_date_only",
            y="cumulative",
            color=treatment_col,
            title="Cumulative completes",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Column '{treatment_col}' not found for treatment progress chart.")

    if community_col and community_col in df.columns:
        st.subheader("Completions by community")
        community_counts = (
            df.groupby([community_col, treatment_col])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            community_counts,
            x=community_col,
            y="count",
            color=treatment_col if treatment_col in df.columns else None,
            title="Completed surveys by community",
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
