# RCT Field Flow

**A comprehensive toolkit for managing randomized controlled trial field operations** — from study design and power calculations through randomization, data collection monitoring, quality control, analysis, and reporting.

## Overview

RCT Field Flow provides an integrated web-based platform supporting the complete lifecycle of RCT field operations. The toolkit offers intuitive interfaces and automated workflows that maintain research rigor while streamlining operations from initial design through final analysis.

---

## Access & Authentication

The toolkit uses basic HTTP authentication for temporary access control during field operations. This lightweight security layer prevents unauthorized access to sensitive research data and operations while keeping setup simple for field teams.

**Important Notes:**
- Default credentials are configured in `config/default.yaml` for quick deployment
- Authentication is designed for **temporary field use** and short-term project access
- For production deployments or long-term use, implement proper authentication (OAuth, SSO, or enterprise identity management)
- SurveyCTO API credentials are stored separately and can be managed via environment variables or secure credential stores

This approach balances immediate usability for field researchers with basic security during active data collection periods.

---

## Toolkit Modules

### 🏠 Home

Quick-start dashboard with navigation guidance, documentation links, and project overview.

### 🎯 RCT Design

Study planning hub for documenting research design, intervention logic, and preparation for power calculations and randomization.

### ⚡ Power Calculations

An interactive dashboard for statistical power analysis and sample size determination, built on J-PAL methodologies.

**Calculation Modes:**

- **MDE given sample size** — fix your sample and find the minimum detectable effect
- **Sample size given MDE** — set your target effect and find the required sample

**Outcome Types:**

- Continuous outcomes (test scores, income, consumption)
- Binary outcomes (enrollment, employment, take-up)

**Design Options:**

- Individual and cluster randomization with ICC and design effect calculations
- Baseline covariate adjustments (R²) to increase precision
- Imperfect compliance modeling
- Interactive power curves and cluster configuration trade-off tables

**Attrition Adjustment:**

- Specify an expected attrition rate and the module automatically calculates required sample size **with and without attrition** side by side — so you can see exactly how much attrition inflates your required sample

**Reproducibility:**

- Download the complete **Stata or Python code** that replicates your exact results — run it outside the app to verify or share for pre-registration

### 🎲 Randomization

Comprehensive treatment assignment covering all major randomization designs, with built-in optimization and fairness validation.

**Methods:**

- **Simple** — pure random assignment
- **Stratified** — block randomization within strata to ensure balance across key subgroups
- **Cluster** — group-level assignment (e.g., villages, schools)
- **Stratified-cluster** — combined approach for clustered designs with stratification

**Rerandomization:**

A powerful feature for achieving well-balanced treatment arms. Configure the number of iterations (up to 10,000), select balance variables, and the algorithm runs all iterations and picks the one with the **maximum minimum p-value** — the draw that produces the best covariate balance across all arms. A formatted balance table is shown and downloadable.

**Randomization Fairness Validation:**

An underused but important diagnostic: the module re-runs your randomization many times with different random seeds to verify that the procedure is truly fair. Each observation should have approximately equal probability of being assigned to each treatment arm across draws. Results are displayed as a histogram that you can download — a transparent, shareable proof that your randomization is unbiased.

**Reproducibility:**

- Seed-based randomization for exact reproduction at any time
- Download complete **Python and Stata code** replicating the exact procedure
- Preserves existing assignments for follow-up survey rounds
- Visual balance checks with automatic flagging

### ✅ Quality Checks

Automated high-frequency data quality validation for real-time fieldwork monitoring.

**Data Preparation:**

- Wide-to-long reshaping for repeated measures and SurveyCTO repeat groups
- Multiple data sources: config files, CSV uploads, or SurveyCTO API

**Check Types:**

- **Outlier Detection**: IQR or standard deviation methods with group-based analysis
- **Duration Checks**: Flag suspiciously fast or slow surveys
- **Duplicate Detection**: Key-based or GPS proximity identification
- **Intervention Fidelity**: Verify treatment assignment matches records

**Analysis & Export:**

- Summary statistics by check type and grouping variables
- Interactive filtering and visualization
- Download flagged cases as CSV

### 📊 Analysis & Results

Statistical analysis of RCT outcomes using standard approaches.

**Analysis Types:**

- **Treatment Effects**: Average treatment effects (ATE) with OLS regression
- **Heterogeneity**: Subgroup analysis with interaction terms and forest plots
- **Balance Verification**: Compare baseline characteristics across arms
- **Attrition Analysis**: Calculate rates by treatment and test for differential dropout

**Data & Export:**

- CSV uploads or SurveyCTO API integration
- Automatic merge with randomization data
- Export results as CSV/Excel with confidence intervals and formatted tables

### 🔍 Backcheck Selection *(Coming Soon)*

Stratified sampling of cases for quality verification backchecks.

### 📊 Data Visualization *(Coming Soon)*

Interactive exploration and visualization of field data.

---

## Key Features

### Transparency & Reproducibility

- Seed-based randomization with downloadable replication code
- Complete audit trails for all operations
- Python and Stata code generation for key analyses
- Ready for pre-registration and publication

### Workflow Integration

Modules work seamlessly together with automatic data flow:

1. **Planning**: RCT Design + Power Calculations determine requirements
2. **Pre-Field**: Randomization assigns treatments to participants
3. **During Field**: Quality Checks flag data issues in real time
4. **Post-Field**: Analysis & Results for treatment effect estimation

### Configuration Flexibility

- **Interactive UI** (recommended): Visual forms with validation, no YAML editing
- **YAML Config** (advanced): For automation pipelines and batch processing

---

## Documentation

- **[UI Quick Start Guide](docs/UI_GUIDE.md)** — Step-by-step walkthrough
- **[Randomization Guide](docs/RANDOMIZATION.md)** — Detailed methodology and best practices
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — Common issues and solutions
- **[Examples](examples/)** — Sample datasets and configurations

---

## Acknowledgments

This toolkit was developed to support rigorous field operations for randomized controlled trials. The power calculations module draws on methodologies and best practices from the [Abdul Latif Jameel Poverty Action Lab (J-PAL)](https://www.povertyactionlab.org/), particularly their comprehensive [Power Calculations Guide](https://www.povertyactionlab.org/resource/power-calculations). We are grateful for J-PAL's commitment to making high-quality research methods accessible to the broader research community.
