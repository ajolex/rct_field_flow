# Installation Guide

This guide covers installation and setup for RCT Field Flow on your local machine or server.

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ajolex/rct_field_flow.git
cd rct_field_flow
```

### 2. Install Dependencies

```bash
pip install -e .
```

This installs RCT Field Flow in editable mode along with all required dependencies.

### 3. Configure SurveyCTO Credentials (Optional)

If you plan to use SurveyCTO integration features, provide your credentials via environment variables or configuration file.

**Option A: Environment Variables**

```bash
export SCTO_SERVER="yourserver"
export SCTO_USER="your_username"
export SCTO_PASS="your_password"
```

On Windows (PowerShell):
```powershell
$env:SCTO_SERVER="yourserver"
$env:SCTO_USER="your_username"
$env:SCTO_PASS="your_password"
```

**Option B: Configuration File**

Edit `rct_field_flow/config/default.yaml` and add your credentials:

```yaml
surveycto:
  server: yourserver
  username: your_username
  password: your_password
```

**Note:** The server value can be:
- Just the server name: `yourserver`
- Full subdomain: `yourserver.surveycto.com`
- Complete URL: `https://yourserver.surveycto.com`

The app normalizes these automatically.

### 4. PDF Report Generation (Optional)

PDF generation requires WeasyPrint, which has additional system dependencies.

**Windows:**
- Install GTK and Pango libraries following [WeasyPrint's Windows guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows)
- In your config, set `reports.render_pdf: true`

**Mac:**
- WeasyPrint dependencies usually install automatically via pip

**Linux:**
- Install required system packages:
  ```bash
  sudo apt-get install python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
  ```

**Default Configuration:**
By default, `reports.render_pdf: false` so HTML reports work out of the box. Enable PDF generation only when you need it.

## Running the Application

### Web Interface (Recommended)

Launch the integrated web interface:

```bash
python -m streamlit run rct_field_flow/app.py
```

Or use the convenience scripts:

**Windows:**
```
launch_ui.bat
```

**Mac/Linux:**
```bash
./launch_ui.sh
```

The application will open in your default browser at `http://localhost:8501`.

### Monitoring Dashboard Only

To run just the monitoring dashboard:

```bash
streamlit run rct_field_flow/monitor.py
```

### Command Line Interface

RCT Field Flow also provides CLI commands for scripting and automation:

```bash
# Randomization
rct-field-flow randomize --baseline path/to/baseline.csv

# Case assignment
rct-field-flow assign-cases --randomized randomized_cases.csv

# Upload to SurveyCTO
rct-field-flow upload-cases --csv cases_upload.csv

# Quality checks
rct-field-flow quality-check --submissions submissions.csv --flags-output flags.csv

# Backcheck selection
rct-field-flow backcheck --submissions submissions.csv

# Report generation
rct-field-flow report --submissions submissions.csv

# Statistical analysis
rct-field-flow analyze --data analysis_ready.csv --outcomes outcome1 --outcomes outcome2
```

Use `--config-path` to specify a custom configuration file:

```bash
rct-field-flow randomize --baseline baseline.csv --config-path my_project/config.yaml
```

## Configuration

### Using the Web Interface (Recommended)

All modules support visual configuration through the web interface:
- No YAML editing required
- Form-based input with validation
- Real-time preview and error checking
- Guided workflows with helpful tooltips

### YAML Configuration (Advanced)

For automation pipelines and batch processing, create or edit YAML configuration files.

**Example Project Structure:**
```
my_rct_project/
├── config.yaml          # Project configuration
├── data/
│   ├── baseline.csv
│   └── submissions.csv
└── output/
```

**Sample config.yaml:**

```yaml
randomization:
  id_column: caseid
  treatment_column: treatment
  method: stratified
  arms:
    - name: control
      proportion: 0.5
    - name: treatment
      proportion: 0.5
  strata:
    - region
    - gender
  balance_covariates:
    - age
    - income
  iterations: 100
  seed: 12345

case_assignment:
  case_id_column: caseid
  label_template: "{caseid} - {name}"
  teams:
    - name: Team_A
      size: 10
    - name: Team_B
      size: 10

monitoring:
  submission_date_column: submissiondate
  enumerator_column: enumerator_id
  rest_days:
    - Sunday

quality_checks:
  duration:
    column: duration_seconds
    min_threshold: 600
    max_threshold: 7200
  outliers:
    method: iqr
    threshold: 1.5
    variables:
      - variable1
      - variable2

analysis:
  outcome_variables:
    - outcome1
    - outcome2
  heterogeneity_vars:
    - gender
    - region

reports:
  template_path: templates/weekly_report.html
  output_dir: output/reports
  render_pdf: false
```

**Environment Variable Substitution:**

Use `${ENV_VAR}` placeholders to pull values from environment variables:

```yaml
surveycto:
  server: ${SCTO_SERVER}
  username: ${SCTO_USER}
  password: ${SCTO_PASS}
```

## Troubleshooting

### Port Already in Use

If you see "Port 8501 is in use":

```bash
# Use a different port
streamlit run rct_field_flow/app.py --server.port 8502
```

### Import Errors

If you encounter import errors:

```bash
# Reinstall in editable mode
pip install -e . --force-reinstall
```

### SurveyCTO Connection Issues

Common issues:
- **Wrong server format**: Use just `yourserver`, not `yourserver.surveycto.com.surveycto.com`
- **Invalid credentials**: Verify username and password
- **Form permissions**: Ensure your user has access to the forms

### PDF Generation Issues on Windows

If PDFs aren't generating:
1. Verify GTK/Pango installation
2. Check that WeasyPrint installed successfully: `pip install weasyprint`
3. Set `reports.render_pdf: true` in config
4. Check logs for specific error messages

For more help, see [Troubleshooting Guide](docs/TROUBLESHOOTING.md).

## Updating

To update to the latest version:

```bash
cd rct_field_flow
git pull origin master
pip install -e . --upgrade
```

## Uninstalling

```bash
pip uninstall rct_field_flow
```

## Getting Help

- **Documentation**: See [docs/](docs/) folder
- **Issues**: Report bugs on GitHub
- **Examples**: Check [examples/](examples/) for sample configurations

## Next Steps

After installation, see:
- **[README.md](README.md)** for feature overview and module descriptions
- **[UI Quick Start Guide](docs/UI_GUIDE.md)** for step-by-step walkthrough
- **[Randomization Guide](docs/RANDOMIZATION.md)** for randomization best practices
