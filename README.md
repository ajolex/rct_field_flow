# RCT Field Flow

A Python based toolkit for managing the entire field operations in randomized controlled trials survey projects—covering randomization, SurveyCTO case assignment, live monitoring, quality control, backchecks, and one-click analysis.

## Workflow Overview

- **Randomize participants** using four methods: simple, stratified, cluster, or combined stratified+cluster designs with optional rerandomization (up to 10,000 iterations) to pick a random assigment that achieves the most balance on the selected covariates. A positive integer seed is required so results are fully reproducible. Existing treatment assignments (e.g., follow-up rounds) are retained for balance checks without reassigning. See [Randomization Guide](docs/RANDOMIZATION.md) for details.
- **Assign SurveyCTO cases** to enumerator teams based on configurable rules (community, strata, quotas) and produce upload-ready CSVs.
- **Upload cases** directly to SurveyCTO via the API.
- **Monitor progress** in Streamlit with productivity tables, progress-by-arm, per-arm targets/completion, and projected end dates that use configurable workdays.
- **Automate advanced quality checks** with interactive configuration:
  - Flexible outlier detection (IQR or Standard Deviation methods)
  - Duration checks with quantile or absolute thresholds
  - Duplicate detection by keys or GPS proximity
  - Intervention fidelity verification
  - Wide-to-long reshaping for repeated measures (e.g., SurveyCTO repeat groups)
  - Group-based analysis (by enumerator, date, treatment, etc.)
  - Visual results with filtering and export
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

- 🎲 **Configure and run randomization** interactively with 4 methods:
  - Simple randomization
  - Stratified randomization
  - Cluster randomization
  - Stratified + Cluster randomization (combined)
  - Seed required for reproducibility; defaults to 12345 but editable
  - Balance checks with automatic diagnostics and treatment distribution tables
  - **📥 Download randomization code**: Export Python and Stata code with your exact parameters for full transparency and sharing with PIs
- 📋 **Assign cases to teams** with configuration helpers
- ✅ **Run advanced quality checks** with interactive configuration:
  - **Outlier Detection**: IQR or Standard Deviation methods with adjustable thresholds
  - **Duration Checks**: Flag surveys that are too fast/slow (quantile or absolute thresholds)
  - **Duplicate Detection**: By key columns or GPS proximity
  - **Intervention Fidelity**: Verify treatment assignments and expected values
  - **Wide-to-Long Reshaping**: Handle repeated measures from SurveyCTO (e.g., `var_1`, `var_2`, `var_3` → `var`)
  - Group-based analysis (by enumerator, date, treatment, etc.)
  - Interactive results with filtering and export
- 📈 **Monitor real-time progress** with interactive dashboards
- 💾 **Download results** at each step

No configuration file editing required—the UI provides forms for all settings. Sample defaults are loaded from `rct_field_flow/config/default.yaml`.

## Configuration

### Configuration Methods

**Interactive UI (Recommended)**: Configure all settings through the web interface with no YAML editing required. The UI provides:

- Form-based input with validation
- Real-time preview of settings
- Helpful tooltips and examples
- Automatic error checking

**YAML Configuration (Advanced)**: For power users and automation pipelines:

- `randomization`: id column, arms & proportions, method (`simple`, `stratified`, `cluster`, `stratified_cluster`), strata, cluster column, covariates for balance checks, iteration count, and base seed.
- `case_assignment`: case ID, label template, team rules, form IDs, and additional columns for the SurveyCTO case upload.
- `monitoring`: column names for submissions, rest days, and work-week assumptions used in projections.
- `quality_checks`: Configuration for interactive mode (flexible) or YAML mode (fixed thresholds):
  - Duration thresholds (seconds or minutes, quantile or absolute)
  - Duplicate detection keys and GPS proximity
  - Outlier detection method (IQR or SD), thresholds, and grouping
  - Intervention fidelity checks
  - Wide-to-long reshaping patterns for repeated measures
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
- If you don't need PDFs, leave `reports.render_pdf: false` (default). To generate PDFs, install GTK/Pango (WeasyPrint requirement on Windows) and flip it to `true`.

## Randomization: Transparency & Reproducibility

### Download Randomization Code

For complete transparency and reproducibility, RCT Field Flow automatically generates downloadable code (both Python and Stata) that exactly replicates your randomization with all parameters embedded.

**Why This Matters:**

- **Transparency**: Share the exact randomization procedure with Principal Investigators, collaborators, and reviewers
- **Reproducibility**: Anyone can verify and replicate your randomization using the provided code
- **Documentation**: Keep a permanent record of the exact randomization method and parameters used
- **Compliance**: Meet pre-registration and reporting requirements that mandate sharing randomization code

**What's Included:**

- Complete Python script using the `rct_field_flow` package
- Equivalent Stata do-file with identical logic
- All your specific parameters embedded:
  - Random seed for exact replication
  - Treatment arms with exact proportions
  - Stratification variables (if used)
  - Cluster column (if used)
  - Balance covariates (if specified)
  - Rerandomization iterations
- Comments explaining each step
- Balance check code (for Stata)
- Instructions for running the code

**How to Use:**

1. Configure and run your randomization in the UI
2. After successful randomization, click "📄 Download Python Code" or "📄 Download Stata Code"
3. Share the code file with collaborators or include in your project repository
4. The code is ready to run - just update the data file path

**Example Output:**

*Python code includes:*

```python
config = RandomizationConfig(
    id_column="caseid",
    treatment_column="treatment",
    method="stratified",
    arms=[
        TreatmentArm(name="control", proportion=0.5),
        TreatmentArm(name="treatment", proportion=0.5)
    ],
    strata=['region', 'gender'],
    balance_covariates=['age', 'income'],
    iterations=100,
    seed=12345
)
```

*Stata code includes:*

```stata
set seed 12345
bysort region gender: gen double random_draw = runiform()
* ... treatment assignment logic ...
* ... balance checks ...
```

This feature ensures your randomization is fully documented and can be independently verified, meeting the highest standards for RCT research.

## Quality Checks: Interactive High-Frequency Checks

The quality checks module provides comprehensive field data validation with an intuitive interface designed for RCT research teams.

### Key Features

**🔄 Data Reshaping for Repeated Measures**

- Automatically detect and reshape wide-format repeated measures (e.g., `icm_hr_worked_7d_1`, `icm_hr_worked_7d_2`, `icm_hr_worked_7d_3`)
- Convert to long format (`icm_hr_worked_7d`) for proper analysis
- Handles SurveyCTO repeat groups seamlessly
- Supports multiple reshape patterns simultaneously
- Preview reshaped data before running checks

**🔢 Outlier Detection**

- Two methods: IQR (Interquartile Range) or Standard Deviation
- Adjustable thresholds (IQR: 0.5-3.0 multiplier, SD: 1.0-5.0 standard deviations)
- Group-based detection (analyze by enumerator, treatment, date, etc.)
- Automatic numeric conversion with helpful error messages
- Highlights reshaped variables in selection dropdown

**⏱️ Duration Checks**

- Flag surveys that are too fast or too slow
- Two detection methods:
  - Quantile-based: Flag fastest/slowest X percentile
  - Absolute thresholds: Set min/max duration limits
- Support for seconds or minutes
- Useful for identifying speeders or incomplete surveys

**👥 Duplicate Detection**

- Identify duplicates by key columns (caseid, enumerator, etc.)
- Optional GPS-based duplicate detection with proximity threshold
- Helpful for catching data entry errors or fraudulent submissions

**✅ Intervention Fidelity**

- Verify treatment assignments match expected values
- Flag unexpected treatment codes
- Ensure intervention delivery as designed

### Usage Workflow

1. **Load Data**: Choose from project config, CSV upload, or SurveyCTO API
2. **Configure Reshape** (if needed):
   - Expand "Reshape repeated measures" section
   - Select patterns or enter manually (e.g., `icm_hr_worked_7d_*`)
   - Choose ID columns
   - Click "Apply Reshape"
3. **Configure Checks**: Use tabs to set up outlier detection, duration checks, etc.
4. **Run Checks**: Click "Run Quality Checks" to analyze data
5. **Review Results**:
   - See summary statistics by check type
   - Group results by enumerator, date, or custom variables
   - View detailed flagged cases
   - Download results as CSV

### Configuration Modes

**Interactive Mode (Recommended)**

- No YAML editing required
- Visual configuration with forms and dropdowns
- Real-time validation and helpful tooltips
- Preview results before downloading

**YAML Mode (Advanced)**

- For automation pipelines and batch processing
- Backward compatible with existing configurations
- Suitable for scheduled quality checks

### Example Use Cases

- **Repeated Measures**: Check outliers in hours worked across 7 days (reshaped from wide format)
- **Speed Checks**: Flag surveys completed in under 10 minutes
- **Enumerator Monitoring**: Detect outliers by enumerator to identify training needs
- **GPS Validation**: Find duplicate GPS coordinates within 10 meters
- **Treatment Verification**: Ensure all treatment codes are valid (control/treatment)

## Examples

The `examples/` directory contains miniature baseline and submission datasets plus a sample configuration (`config_example.yaml`) showing how to adapt the pipeline to a new RCT.

## Documentation

- **[UI Quick Start Guide](docs/UI_GUIDE.md)** – Step-by-step guide for the integrated web interface
- **[Randomization Guide](docs/RANDOMIZATION.md)** – Detailed randomization methodology and best practices
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** – Common issues and solutions
