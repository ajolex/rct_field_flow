from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
import yaml

from .analyze import one_click_analysis
from .assign_cases import assign_cases
from .backcheck import sample_backchecks
from .flag_quality import flag_all
from .randomize import RandomizationConfig, Randomizer, TreatmentArm
from .report import generate_weekly_report
from .upload_cases import upload_to_surveycto

app = typer.Typer(help="RCT Field Flow CLI")


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _resolve_placeholder(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value.startswith("${") and value.endswith("}"):
        return None
    value = value.strip()
    return value or None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _expand_env(raw)


def build_randomization_config(cfg: dict) -> RandomizationConfig:
    arms = [
        TreatmentArm(name=arm["name"], proportion=arm["proportion"])
        for arm in cfg.get("arms", [])
    ]
    if not arms:
        arms = [
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ]
    return RandomizationConfig(
        id_column=cfg["id_column"],
        treatment_column=cfg.get("treatment_column", "treatment"),
        method=cfg.get("method", "simple"),
        arms=arms,
        strata=cfg.get("strata", []),
        cluster=cfg.get("cluster"),
        balance_covariates=cfg.get("balance_covariates", []),
        iterations=cfg.get("iterations", 1),
        seed=cfg.get("seed"),
        use_existing_assignment=cfg.get("use_existing_assignment", True),
    )


@app.command()
def randomize(
    baseline: Path = typer.Option(..., exists=True, readable=True, help="Baseline dataset (CSV)."),
    output: Path = typer.Option(Path("randomized_cases.csv"), help="Output CSV for assignments."),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True, help="Project config."),
) -> None:
    """Run randomization with rerandomization balance checks."""
    app_config = load_config(str(config_path))
    rand_cfg = build_randomization_config(app_config.get("randomization", {}))
    df = pd.read_csv(baseline)
    result = Randomizer(rand_cfg).run(df)
    result.assignments.to_csv(output, index=False)
    typer.echo(f"Saved assignments to {output}")
    typer.echo(f"Best min p-value: {result.best_min_pvalue:.4f}")


@app.command("assign-cases")
def assign_cases_cmd(
    randomized: Path = typer.Option(..., exists=True, readable=True, help="Randomized dataset."),
    output: Path = typer.Option(Path("cases_upload.csv"), help="SurveyCTO case CSV."),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True),
) -> None:
    """Build SurveyCTO case roster from randomized data."""
    app_config = load_config(str(config_path))
    case_cfg = app_config.get("case_assignment", {})
    df = pd.read_csv(randomized)
    roster = assign_cases(df, case_cfg)
    roster.to_csv(output, index=False)
    typer.echo(f"Prepared {len(roster)} cases at {output}")


@app.command("quality-check")
def quality_check_cmd(
    submissions: Path = typer.Option(..., exists=True, readable=True, help="Submission data CSV."),
    flags_output: Optional[Path] = typer.Option(None, help="Optional CSV path for detailed flags."),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True),
) -> None:
    """Run field quality checks and print summary."""
    app_config = load_config(str(config_path))
    quality_cfg = app_config.get("quality_checks", {})
    df = pd.read_csv(submissions)
    results = flag_all(df, quality_cfg)
    typer.echo("Flag counts:")
    typer.echo(results.flag_counts.to_string())
    if flags_output:
        results.flags.to_csv(flags_output, index=False)
        typer.echo(f"Flag details saved to {flags_output}")


@app.command()
def backcheck(
    submissions: Path = typer.Option(..., exists=True, readable=True, help="Submission data CSV."),
    output: Path = typer.Option(Path("backcheck_cases.csv"), help="Output CSV for backcheck sample."),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True),
) -> None:
    """Select high-risk and random cases for backchecking."""
    app_config = load_config(str(config_path))
    quality_cfg = app_config.get("quality_checks", {})
    backcheck_cfg = app_config.get("backcheck", {})
    df = pd.read_csv(submissions)
    flags = flag_all(df, quality_cfg).flags
    sample = sample_backchecks(df, flags, backcheck_cfg)
    sample.to_csv(output, index=False)
    typer.echo(f"Generated backcheck list with {len(sample)} cases at {output}")


@app.command()
def report(
    submissions: Path = typer.Option(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("reports"), help="Directory for report outputs."),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True),
) -> None:
    """Render the weekly monitoring report."""
    app_config = load_config(str(config_path))
    quality_cfg = app_config.get("quality_checks", {})
    backcheck_cfg = app_config.get("backcheck", {})
    df = pd.read_csv(submissions)
    totals = len(df)
    enumerator_col = app_config.get("monitoring", {}).get("enumerator_column", "enumerator")
    treatment_col = app_config.get("monitoring", {}).get("treatment_column", "treatment")
    ts_col = app_config.get("monitoring", {}).get("submission_timestamp_column", "submission_date")
    work_days = app_config.get("monitoring", {}).get("work_days_per_week", 6)
    enumerator_summary = df[enumerator_col].value_counts().to_dict() if enumerator_col in df.columns else {}
    treatment_summary = df[treatment_col].value_counts().to_dict() if treatment_col in df.columns else {}
    quality = flag_all(df, quality_cfg)
    flag_counts = quality.flag_counts.to_dict()
    backchecks = sample_backchecks(df, quality.flags, backcheck_cfg).head(10)
    backcheck_html = backchecks.to_html(index=False) if not backchecks.empty else ""
    if ts_col in df.columns:
        dates = pd.to_datetime(df[ts_col], errors="coerce").dt.date
        daily_rate = dates.value_counts().mean()
    else:
        daily_rate = 0
    target = app_config.get("target_n")

    def _project_end_date(total: int, target_n: Optional[int], rate: float, work: int) -> str:
        if not target_n or rate is None or rate <= 0:
            return ""
        remaining = max(0, target_n - total)
        if remaining == 0:
            return date.today().isoformat()
        work_factor = work / 7 if work else 1
        adjusted = rate * work_factor
        if adjusted <= 0:
            return ""
        days_left = remaining / adjusted
        return (date.today() + timedelta(days=round(days_left))).isoformat()

    projected_end = _project_end_date(totals, target, daily_rate, work_days)

    context = {
        "project": app_config.get("project", "RCT Field Flow"),
        "total_completes": totals,
        "enumerator_summary": enumerator_summary,
        "treatment_summary": treatment_summary,
        "flag_counts": flag_counts,
        "backcheck_list": backcheck_html,
        "generated_on": date.today().isoformat(),
        "target_n": target,
        "projected_end": projected_end,
    }
    outputs = generate_weekly_report(context, {**app_config.get("reports", {}), "output_dir": str(output_dir)})
    for kind, path in outputs.items():
        typer.echo(f"{kind.upper()} report: {path}")


@app.command()
def analyze(
    data: Path = typer.Option(..., exists=True, readable=True),
    outcomes: Optional[List[str]] = typer.Option(None, help="Outcome columns to analyze."),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True),
) -> None:
    """Run one-click ATEs for selected outcomes."""
    app_config = load_config(str(config_path))
    df = pd.read_csv(data)
    analysis_cfg = app_config.get("analysis", {})
    outcome_list = outcomes or analysis_cfg.get("outcome_columns", [])
    if not outcome_list:
        typer.echo("No outcomes specified.", err=True)
        raise typer.Exit(code=1)
    results = one_click_analysis(df, outcome_list, analysis_cfg)
    for outcome, arms in results.items():
        typer.echo(f"\nOutcome: {outcome}")
        for arm, stats in arms.items():
            typer.echo(f"  {arm}: estimate={stats['estimate']:.3f}, p-value={stats['p_value']:.3f}")


@app.command("upload-cases")
def upload_cases_cmd(
    csv: Path = typer.Option(..., exists=True, readable=True),
    config_path: Path = typer.Option(Path("rct_field_flow/config/default.yaml"), exists=True),
) -> None:
    """Upload a case CSV to SurveyCTO."""
    app_config = load_config(str(config_path))
    scto_cfg = app_config.get("surveycto", {})
    server = _resolve_placeholder(scto_cfg.get("server", ""))
    username = _resolve_placeholder(scto_cfg.get("username", ""))
    password = _resolve_placeholder(scto_cfg.get("password", ""))
    form_id = _resolve_placeholder(scto_cfg.get("form_id", ""))
    if not all([server, username, password]):
        typer.echo("Missing SurveyCTO credentials. Set environment variables or update config.", err=True)
        raise typer.Exit(code=1)
    response = upload_to_surveycto(
        csv_path=str(csv),
        server=server,
        username=username,
        password=password,
        form_id=form_id,
    )
    typer.echo(f"SurveyCTO response: {response}")


def main():
    app()


if __name__ == "__main__":
    main()
