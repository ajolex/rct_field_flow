# RCT Field Flow

Python toolkit for managing field operations in randomized controlled trials—covering randomization, SurveyCTO case management, live monitoring, quality control, backchecks, weekly reporting, and one-click analysis.

## Workflow Overview

- **Randomize participants** using simple, stratified, or cluster designs with rerandomization (up to 10,000 iterations) to maximise joint balance p-values across covariates. Existing treatment assignments (e.g., follow-up rounds) are ingested for balance checks without reassigning. See [Randomization Guide](docs/RANDOMIZATION.md) for detailed documentation.
- **Assign SurveyCTO cases** to enumerator teams based on configurable rules (community, strata, quotas) and produce upload-ready CSVs.
- **Upload cases** directly to SurveyCTO via the API.
- **Monitor progress** in Streamlit with productivity, progress-by-arm, and projected end dates that respect rest days.
- **Automate data quality checks** such as speeding, outliers, duplicates, intervention fidelity, and enumerator risk scoring.
- **Select backchecks** with configurable high-risk quotas and random draws.
- **Generate weekly reports** (HTML/PDF) summarising key metrics, flag counts, and backcheck rosters.
- **Run one-click analytics** for ATEs, heterogeneity, and attrition summaries.


## Installation

```bash
pip install -e .
```

Ensure SurveyCTO credentials are available via environment variables (`SCTO_SERVER`, `SCTO_USER`, `SCTO_PASS`) or populated in the config file.

## Configuration

Edit `rct_field_flow/config/default.yaml` (or supply an alternate file) to customise the pipeline:

- `randomization`: id column, arms & proportions, method (`simple`, `stratified`, `cluster`), strata, cluster column, covariates for balance checks, iteration count, and base seed.
- `case_assignment`: case ID, label template, team rules, form IDs, and additional columns for the SurveyCTO case upload.
- `monitoring`: column names for submissions, rest days, and work-week assumptions used in projections.
- `quality_checks`: thresholds for duration, duplicates, GPS, other-specify merging, and intervention fidelity columns.
- `backcheck`: sample size, high-risk quota, and columns to include in the roster.
- `reports`: Jinja template path plus output directory/format options.
- `analysis`: default outcome list, heterogeneity columns, and attrition flag reference.

Use `${ENV_VAR}` placeholders in the YAML to pull values from the environment.

## CLI Commands

```bash
rct-field-flow randomize --baseline path/to/baseline.csv
rct-field-flow assign-cases --randomized randomized_cases.csv
rct-field-flow upload-cases --csv cases_upload.csv
rct-field-flow quality-check --submissions submissions.csv --flags-output flags.csv
rct-field-flow backcheck --submissions submissions.csv
rct-field-flow report --submissions submissions.csv
rct-field-flow analyze --data analysis_ready.csv --outcomes outcome1 --outcomes outcome2
```

Use `--config-path` to point to a different YAML file when running multiple projects.

## Monitoring Dashboard

Launch the Streamlit app (after setting up config/credentials):

```bash
streamlit run rct_field_flow/monitor.py
```

The dashboard automatically pulls from SurveyCTO when credentials are available; otherwise it falls back to the CSV path defined in the config.

## Examples

The `examples/` directory contains miniature baseline and submission datasets plus a sample configuration (`config_example.yaml`) showing how to adapt the pipeline to a new RCT.
