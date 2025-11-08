# Quick Start Guide: Integrated UI

This guide will help you get started with the RCT Field Flow integrated web interface.

## Launching the Application

Open your terminal and run:

```bash
streamlit run rct_field_flow/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Navigation

The sidebar on the left provides navigation between five main sections:

### 🏠 Home
- Upload your baseline data
- Initialize or load configuration
- See quick start instructions

### 🎲 Randomization
- **Upload baseline data** with participant information
- **Configure treatment arms** (names and proportions)
- **Select randomization method**: simple, stratified, or cluster
- **Add balance covariates** for rerandomization
- **Set iterations** for balance optimization
- **Run randomization** and view results
- **Download assignments** as CSV

#### Example Workflow:
1. Upload your baseline CSV (must have unique ID column)
2. Select "stratified" method
3. Choose stratification variables (e.g., province, gender)
4. Add balance covariates (e.g., age, income, household size)
5. Set iterations to 1000 for good balance
6. Click "Run Randomization"
7. Review balance table and p-values
8. Download results

### 📋 Case Assignment
- **Upload randomized data** (or use results from Randomization page)
- **Configure teams** (number and names)
- **Select assignment strategy**:
  - Round-robin: Sequential assignment
  - Balanced: Equal distribution
  - By treatment: Assign within treatment arms
- **Set custom rules** (optional) for team-specific restrictions
- **Download team assignments** ready for field distribution

#### Example Workflow:
1. Use data from previous randomization step
2. Set number of teams (e.g., 3)
3. Name each team (e.g., Team_North, Team_Central, Team_South)
4. Choose "balanced" strategy for equal distribution
5. Click "Assign Cases"
6. Download assignments for each team

### ✅ Quality Checks
- **Upload submission data** from the field
- **Configure duration thresholds** to flag surveys that are too fast/slow
- **Identify duplicates** on specified columns
- **Review enumerator summaries** with productivity metrics
- **Export flagged cases** for follow-up

#### Example Workflow:
1. Upload daily submissions CSV
2. Set duration column (e.g., "duration_minutes")
3. Set minimum (5 min) and maximum (120 min) thresholds
4. Select numeric columns to check for outliers
5. Run quality checks
6. Review flagged submissions
7. Export problem cases for supervisor review

### 📊 Monitoring Dashboard
- **Real-time progress tracking** with key metrics
- **Daily submission trends** over time
- **Treatment arm distribution** to ensure balanced enrollment
- **Enumerator productivity** with comparative metrics
- **Community-level progress** tracking
- **Auto-refresh option** for live monitoring

#### Setup Options:
1. **CSV Upload**: Upload submissions file for offline analysis
2. **SurveyCTO Integration**: Enter credentials to pull live data
   - Server: your_server.surveycto.com
   - Username: your_username
   - Password: your_password
   - Form ID: your_form_id

## Key Features

### No Configuration Files Needed
All settings are configured through web forms. No need to edit YAML files!

### Session Persistence
Your data and results are saved in the current session. You can move between pages without losing work.

### Download Capabilities
Every major output can be downloaded as CSV:
- Randomization assignments
- Case assignments
- Quality check reports
- Balance tables

### Interactive Visualizations
- Plotly charts for all visualizations
- Hover for detailed information
- Zoom, pan, and export capabilities

### Validation & Diagnostics
- Automatic validation of inputs
- Balance p-values and diagnostics for randomization
- Treatment distribution verification
- Assignment probability checks

## Tips & Best Practices

### Randomization
- **Start with 1000 iterations** for good balance
- **Use 10,000+ iterations** only if you need excellent balance (may affect inference)
- **Check the balance table** - aim for p-values > 0.10 on key covariates
- **Review p-value history** to see improvement over iterations
- **Save your seed** for reproducibility

### Case Assignment
- **Balance workload** across teams for fair distribution
- **Use by-treatment strategy** when teams specialize in specific arms
- **Test with small samples** first before assigning full dataset

### Quality Checks
- **Run daily** to catch issues early
- **Set realistic duration thresholds** based on pilot data
- **Track enumerator trends** to identify training needs
- **Follow up quickly** on flagged cases

### Monitoring
- **Enable auto-refresh** during active data collection
- **Check daily** to stay on schedule
- **Monitor enumerator productivity** for workload balancing
- **Track treatment arm balance** to ensure even enrollment

## Troubleshooting

### Application Won't Start
- Ensure all dependencies are installed: `pip install -e .`
- Check that port 8501 is not already in use
- Try specifying a different port: `streamlit run rct_field_flow/app.py --server.port 8502`

### Data Upload Issues
- Ensure CSV is properly formatted (UTF-8 encoding)
- Check that column names don't have special characters
- Verify file is not corrupted

### Randomization Errors
- Check for duplicate IDs in your baseline data
- Ensure proportions sum to 1.0
- Verify column names match your data

### Performance Issues
- Large datasets (>100k rows) may be slow
- Consider reducing iterations if taking too long
- Close unused browser tabs
- Use Chrome or Firefox for best performance

## Advanced Usage

### Custom Configuration Files
While the UI doesn't require config files, you can still:
1. Generate config through UI usage
2. Save config using "Save Current Config" button
3. Load saved config in future sessions

### Opening Monitoring Dashboard Separately
For a dedicated monitoring window:
```bash
streamlit run rct_field_flow/monitor.py
```

This runs the monitoring dashboard independently from the main app.

### Integration with CLI
The UI and CLI can be used together:
- Use UI for interactive exploration and testing
- Use CLI for automated scripts and batch processing
- Both use the same underlying engine

## Next Steps

1. **Explore the Randomization Guide**: `docs/RANDOMIZATION.md` for detailed methodology
2. **Check example data**: `examples/sample_baseline.csv` for data structure reference
3. **Review the changelog**: `CHANGELOG.md` for recent updates
4. **Visit the repository**: [github.com/ajolex/rct_field_flow](https://github.com/ajolex/rct_field_flow)

## Support

For issues or questions:
- Check the documentation in the `docs/` folder
- Review test scripts in the root directory for examples
- Open an issue on GitHub

---

**Happy Field Management!** 🔬📊
