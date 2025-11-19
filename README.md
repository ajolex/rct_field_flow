# RCT Field Flow

**A comprehensive toolkit for managing randomized controlled trial field operations** — from study design and power calculations through randomization, data collection monitoring, quality control, analysis, and reporting.

## Overview

RCT Field Flow provides an integrated web-based platform supporting the complete lifecycle of RCT field operations. The toolkit offers intuitive interfaces and automated workflows that maintain research rigor while streamlining operations from initial design through final analysis.

**📥 [Installation &amp; Setup Instructions](INSTALLATION.md)**

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

Statistical power analysis and sample size determination based on J-PAL methodologies.

**Capabilities:**

- Calculate required sample size for target effect or MDE for given sample
- Support for continuous outcomes (test scores, income) and binary outcomes (enrollment, employment)
- Individual and cluster randomization designs with ICC and design effect calculations
- Baseline covariate adjustments (R²) and imperfect compliance modeling
- Interactive power curves and cluster configuration trade-off tables
- Export Python and Stata code for transparency and pre-registration
- Built-in educational content explaining power concepts

### 🎲 Randomization

Treatment assignment with complete transparency and reproducibility.

**Methods:**

- Simple random assignment
- Stratified (block randomization within strata)
- Cluster (group-level assignment)
- Stratified-cluster (combined approach)

**Features:**

- Seed-based randomization for exact reproducibility
- Balance diagnostics comparing treatment arms
- Optional rerandomization (up to 10,000 iterations) to optimize covariate balance
- **Randomization validation**: Run randomization multiple times (e.g., 500) with different seeds to verify fairness
- Assignment probability analysis with histograms to detect systematic bias
- Preserves existing assignments for follow-up rounds
- Visual balance checks with automatic flagging
- Downloadable Python and Stata code replicating exact procedure

### 📋 Case Assignment

Distribute survey cases to enumerator teams with configurable rules.

**Capabilities:**

- Interactive team assignment rule builder
- Stratified assignment by geography, treatment arm, or other factors
- Quota management and distribution validation
- Treatment-specific form ID routing
- Direct SurveyCTO API upload (merge/append/replace modes)
- Download prepared case lists as CSV

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

### 🔍 Backcheck Selection

Sample cases for quality verification using stratified random sampling.

**Features:**

- Define sample size with high-risk quota oversampling
- Risk scoring based on quality flags
- Stratified sampling by enumerator, date, or other factors
- Visual risk distributions and enumerator summaries
- Generate backcheck rosters as CSV/Excel

### 📄 Report Generation

Automated summary reports combining monitoring and quality data.

**Components:**

- Treatment arm progress and enumerator productivity
- Quality issue summaries with flagged cases
- Timeline projections and target progress
- Key metrics dashboard

**Outputs:** HTML (always) and PDF (optional)

### 📈 Monitoring Dashboard

Real-time field operations tracking.

**Tracking:**

- Submissions per enumerator with daily counts
- Average survey duration and productivity metrics
- Progress by treatment arm vs. targets
- Timeline projections with business-day calculations
- Supervisor roll-ups and performance identification

**Integration:**

- SurveyCTO API for live data or CSV uploads
- Configurable column mapping
- Export productivity and progress tables as CSV

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
2. **Pre-Field**: Randomization assigns treatments → Case Assignment distributes work
3. **During Field**: Monitoring Dashboard tracks progress + Quality Checks flag issues
4. **Quality Assurance**: Backcheck Selection for verification sampling
5. **Post-Field**: Analysis & Results for findings + Report Generation for documentation

### Configuration Flexibility

- **Interactive UI** (recommended): Visual forms with validation, no YAML editing
- **YAML Config** (advanced): For automation pipelines and batch processing

---

## Documentation

- **[INSTALLATION.md](INSTALLATION.md)** — Setup and running instructions
- **[UI Quick Start Guide](docs/UI_GUIDE.md)** — Step-by-step walkthrough
- **[Randomization Guide](docs/RANDOMIZATION.md)** — Detailed methodology and best practices
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — Common issues and solutions
- **[Examples](examples/)** — Sample datasets and configurations

---

## Acknowledgments

This toolkit was developed to support rigorous field operations for randomized controlled trials. The power calculations module draws on methodologies and best practices from the [Abdul Latif Jameel Poverty Action Lab (J-PAL)](https://www.povertyactionlab.org/), particularly their comprehensive [Power Calculations Guide](https://www.povertyactionlab.org/resource/power-calculations). We are grateful for J-PAL's commitment to making high-quality research methods accessible to the broader research community.
