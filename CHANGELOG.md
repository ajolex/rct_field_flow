# Changelog

All notable changes to this project will be documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

### Added
- Comprehensive randomization engine supporting simple, stratified, and cluster
  assignment with rerandomization balance checks, exposed via
  `RandomizationConfig`, `TreatmentArm`, `Randomizer`, and `RandomizationResult`
  (`rct_field_flow/randomize.py`).
- Configurable SurveyCTO case-assignment pipeline with rule-based team routing
  and form routing (`rct_field_flow/assign_cases.py`).
- Field quality module covering speed checks, numeric outliers, duplicates,
  fidelity checks, enumerator risk summaries, and helper dataclasses
  (`rct_field_flow/flag_quality.py`).
- Risk-based backcheck sampler with configurable quotas and roster columns
  (`rct_field_flow/backcheck.py`).
- Reporting utilities with HTML/PDF output (PDF optional) and templated weekly
  report (`rct_field_flow/report.py`, `templates/weekly_report.html`).
- Analysis helpers for ATE, heterogeneity, attrition, and batch outcome runs
  (`rct_field_flow/analyze.py`).
- Streamlit monitoring dashboard with SurveyCTO integration, CSV fallback, and
  projected end-date logic (`rct_field_flow/monitor.py`).
- Typer-based CLI exposing `randomize`, `assign-cases`, `quality-check`,
  `backcheck`, `report`, `analyze`, and `upload-cases` commands with env-aware
  configuration loading (`rct_field_flow/cli.py`).
- Example configuration/dataset updates showing end-to-end workflow
  (`rct_field_flow/config/default.yaml`, `examples/config_example.yaml`,
  `examples/dummy_*.csv`).
- Pytest coverage for randomization behaviour (`tests/test_randomize.py`).
- Project documentation detailing workflow, CLI usage, and WeasyPrint guidance
  (`README.md`).
- `CHANGELOG.md` to track future changes.

### Changed
- Requirements now pin `numpy>=2.0,<3.0` to ensure wheels are available for
  Python 3.13 (`pyproject.toml`, `requirements.txt`).
- Removed `linearmodels` from core dependencies to avoid requiring Microsoft C++
  build tools on Windows; install manually when advanced panel estimators are
  needed (`pyproject.toml`, `requirements.txt`, `README.md`).
- PDF generation gracefully degrades when WeasyPrint system dependencies are
  missing, surfacing a clear error message only when PDF output is requested
  (`rct_field_flow/report.py`).
- Config loader and dashboard utilities now expand environment variables so
  `.yaml` files can reference `${ENV_VAR}` placeholders.
- README updated with workflow overview, CLI instructions, WeasyPrint notes, and
  installation guidance.
