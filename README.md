# RCT Field Flow

Python toolkit for managing field operations in randomized controlled trials—covering randomization, SurveyCTO case assignment, live monitoring, quality control, backchecks, and one-click analysis.

## Workflow Overview

- **Randomize participants** using simple, stratified, or cluster designs with optional rerandomization (up to 10,000 iterations). A positive integer seed is required so results are fully reproducible. Existing treatment assignments (e.g., follow-up rounds) are ingested for balance checks without reassigning. See [Randomization Guide](docs/RANDOMIZATION.md) for details.
- **Assign SurveyCTO cases** to enumerator teams based on configurable rules (community, strata, quotas) and produce upload-ready CSVs.
- **Upload cases** directly to SurveyCTO via the API.
- **Monitor progress** in Streamlit with productivity tables, progress-by-arm, per-arm targets/completion, and projected end dates that use configurable workdays.
- **Automate data quality checks** for speeding, outliers, duplicates, intervention fidelity, and enumerator risk scoring.
- **Select backchecks** with configurable high-risk quotas and random draws.
- **Generate weekly reports** (HTML/PDF) summarising key metrics, flag counts, and backcheck rosters.
- **Run one-click analytics** for ATEs, heterogeneity, and attrition summaries.

## Installation

```bash
pip install -e .
```

Ensure SurveyCTO credentials are available via environment variables (`SCTO_SERVER`, `SCTO_USER`, `SCTO_PASS`) or populated in the config file.

PDF generation is optional. WeasyPrint is used only when exporting PDF reports. On Windows, native GTK/Pango libraries are required by WeasyPrint. By default, the config ships with `reports.render_pdf: false` so everything else works out of the box. To enable PDFs later, install GTK/Pango as per WeasyPrint’s Windows guide and set `render_pdf: true`.

## Quick Start: Integrated UI

The easiest way to use RCT Field Flow is through the integrated web interface:

```bash
python -m streamlit run rct_field_flow/app.py
```

Or simply double-click `launch_ui.bat` (Windows) or `launch_ui.sh` (Mac/Linux).

This launches a comprehensive interface where you can:

- 🎲 **Configure and run randomization** interactively (seed required for reproducibility; defaults to 12345 but editable)
- 📋 **Assign cases to teams** with configuration helpers
- ✅ **Run quality checks** with visual feedback
- 📈 **Monitor real-time progress** with interactive dashboards
- 💾 **Download results** at each step

No configuration file editing required—the UI provides forms for all settings. Sample defaults are loaded from `rct_field_flow/config/default.yaml`.

## Configuration

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

You can run the dashboard on its own if desired:

```bash
streamlit run rct_field_flow/monitor.py
```

The dashboard automatically pulls from SurveyCTO when credentials are available; otherwise it falls back to the CSV path defined in the config.

### Key Features

- Enumerator Productivity table  
  - Columns: total submissions per enumerator, running average duration (minutes, auto-converts from seconds), and one column per submission date with that day’s counts.  
  - Summary rows: “Avg per Enumerator” (green, two-decimal averages across enumerators) and “Total” (integer counts per day).  
  - Optional supervisor roll-ups (yellow “SUP: …” rows) plus per-enumerator detail lines.  
  - Styling highlights: average-duration column in blue, <60 minute averages highlighted red.  
  - CSV export button: “Download Productivity CSV”.

- Targets and Timeline panel  
  - Manually enter per-arm targets; the dashboard computes completed counts, %age completed, and a “Total” summary row.  
  - Adjustable timeline inputs: start date, today’s date, and a weekday multiselect to exclude rest days (defaults Mon–Sat, i.e., Sunday off).  
  - Outputs: field collection days, projected end date (business-day aware), average productivity per day, days/weeks remaining.  
  - CSV export button: “Download Targets CSV”.

### Tips

- SurveyCTO server value can be `yourserver`, `yourserver.surveycto.com`, or a full URL. The app normalizes this automatically and warns if it detects the common typo `surveycto.com.surveycto.com`.
- If you don’t need PDFs, leave `reports.render_pdf: false` (default). To generate PDFs, install GTK/Pango (WeasyPrint requirement on Windows) and flip it to `true`.

## Examples

The `examples/` directory contains miniature baseline and submission datasets plus a sample configuration (`config_example.yaml`) showing how to adapt the pipeline to a new RCT.

## Documentation

- **[UI Quick Start Guide](docs/UI_GUIDE.md)** – Step-by-step guide for the integrated web interface  
- **[Randomization Guide](docs/RANDOMIZATION.md)** – Detailed randomization methodology and best practices  
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** – Common issues and solutions
