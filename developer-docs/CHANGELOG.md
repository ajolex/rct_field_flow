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
- Comprehensive randomization documentation guide covering all methods, best
  practices, validation checks, and troubleshooting (`docs/RANDOMIZATION.md`).
- Sample baseline dataset with 8,210 observations across 450 clusters for
  testing the full pipeline (`examples/sample_baseline.csv`).
- Test scripts validating all randomization types: simple, stratified, cluster,
  and multi-arm (`test_randomization.py`, `test_randomize_direct.py`).
- **Integrated Streamlit UI** (`rct_field_flow/app.py`) providing a comprehensive
  web interface for all pipeline operations without requiring YAML configuration:
  - Interactive randomization configuration with real-time validation
  - Visual case assignment with team management
  - Quality check dashboard with automated flagging
  - Integrated monitoring dashboard with real-time metrics
  - Session state management for seamless workflow
  - CSV upload/download for all operations
  - SurveyCTO API integration for live data
- UI Quick Start Guide with detailed walkthroughs for each feature (`docs/UI_GUIDE.md`).
- `CHANGELOG.md` to track future changes.

### Changed

- Enhanced randomization engine with comprehensive validation and diagnostics
  following IPA/J-PAL best practices and Bruhn & McKenzie (2009) guidelines:
  - Added duplicate ID detection to catch data quality issues early.
  - Treatment distribution validation ensures actual group sizes match expected
    proportions (strict tolerance for individual randomization, relaxed 10%
    tolerance for cluster randomization to account for varying cluster sizes).
  - Assignment probability tracking for rerandomization (>100 iterations) detects
    systematic bias and warns if units are systematically favored.
  - Diagnostic output includes p-value history, mean/median p-values, and
    assignment probability statistics.
  - Verbose mode (`--verbose` flag) provides detailed progress reporting and
    validation results.
  - Warning system alerts users when high-iteration rerandomization (>1000) may
    affect randomization inference assumptions.
  - Fixed cluster randomization to properly handle pure cluster designs (no
    strata) and cluster + stratified combinations.
- CLI `randomize` command enhanced with `--verbose` and `--balance-table` flags
  for better visibility into randomization process and results.
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
