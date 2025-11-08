# Troubleshooting Guide

## Common Issues and Solutions

### Import Error: "attempted relative import with no known parent package"

**Problem**: When running `streamlit run rct_field_flow/app.py`, you get an ImportError about relative imports.

**Solution**: The app now automatically handles both relative and absolute imports. Make sure you're using:

```bash
python -m streamlit run rct_field_flow/app.py
```

Or from the project root:

```bash
cd rct_field_flow
python -m streamlit run app.py
```

### Port Already in Use

**Problem**: Error message "Port 8501 is already in use"

**Solution**: Either:
1. Stop the existing Streamlit instance
2. Use a different port:
   ```bash
   python -m streamlit run rct_field_flow/app.py --server.port 8502
   ```

### Module Not Found Errors

**Problem**: Cannot import pandas, streamlit, or other modules

**Solution**: Install dependencies:
```bash
pip install -e .
```

Or install individually:
```bash
pip install streamlit pandas plotly numpy scipy pyyaml
```

### App Runs But Shows Blank Page

**Problem**: Browser opens but shows empty page or loading spinner

**Solutions**:
1. Check the terminal for error messages
2. Try a different browser (Chrome or Firefox recommended)
3. Clear browser cache
4. Disable browser extensions that might interfere
5. Check if there are Python errors in the terminal

### Configuration File Not Found

**Problem**: App can't find `default.yaml`

**Solution**: The app works without configuration files! Just use the UI to configure everything. If you want to use existing configs:
1. Ensure you're running from the project root directory
2. Check that `rct_field_flow/config/default.yaml` exists
3. You can also load config files through the Home page UI

### Data Upload Issues

**Problem**: CSV upload fails or data doesn't load

**Solutions**:
1. Ensure CSV is UTF-8 encoded
2. Check that file size is reasonable (<100MB recommended)
3. Verify CSV is properly formatted with headers
4. Try opening the CSV in Excel/Notepad to check for corruption

### Randomization Fails

**Problem**: "Error" message when clicking "Run Randomization"

**Common Causes & Solutions**:
1. **Duplicate IDs**: Ensure your ID column has unique values
   - Check: `df[id_column].duplicated().sum()`
   
2. **Proportions don't sum to 1.0**: Adjust treatment arm proportions
   - They should add up to exactly 1.0 (e.g., 0.5 + 0.5 = 1.0)

3. **Column not found**: Verify selected columns exist in your data
   - Check column names for typos or extra spaces

4. **Missing values in stratification**: Handle missing data first
   - Either drop rows with missing strata or create a "Missing" category

### Performance Issues

**Problem**: App is slow or unresponsive

**Solutions**:
1. **Large datasets**: 
   - Use fewer iterations (<1000 for >50k observations)
   - Reduce number of balance covariates
   - Sample your data for testing first

2. **Too many balance checks**:
   - Limit balance covariates to 5-10 key variables
   - Fewer iterations for initial testing

3. **Browser issues**:
   - Close other tabs
   - Use Chrome or Firefox
   - Restart browser

### SurveyCTO Integration Issues

**Problem**: Can't fetch data from SurveyCTO

**Solutions**:
1. **Check credentials**: Verify server, username, password
2. **Form ID**: Ensure form_id is correct
3. **Permissions**: Your account needs read access to the form
4. **Network**: Check your internet connection
5. **Server URL**: Should be `yourserver.surveycto.com` (no https://)

**Common API Errors**:

- **Error 400: "Please provide a date using the 'date' parameter"**
  - **Fixed**: The code now automatically sends `date=0` to fetch all submissions
  - **Manual fix**: Pass a date string like `'Oct 15, 2024 12:00:00 AM'` to the `since` parameter
  - **Example**: `client.get_submissions('form_id', since='Jan 1, 2025 12:00:00 AM')`

### Running on Remote Server

**Problem**: Want to access app from another computer

**Solution**: Use network settings:
```bash
python -m streamlit run rct_field_flow/app.py --server.address 0.0.0.0 --server.port 8501
```

Then access via: `http://your-server-ip:8501`

**Security Note**: Only do this on trusted networks! Consider using SSH tunneling for remote access:
```bash
ssh -L 8501:localhost:8501 user@remote-server
```

### Error Messages Reference

#### "ValueError: No group keys passed!"
- **Cause**: Empty stratification list with cluster randomization
- **Fix**: Either add strata or don't use cluster+stratified method

#### "AssertionError: Treatment arm sizes don't match"
- **Cause**: Actual distribution differs from expected
- **Fix**: This is normal for cluster randomization with varying cluster sizes
- **Note**: Cluster randomization allows up to 10% deviation

#### "KeyError: column not found"
- **Cause**: Selected column doesn't exist in data
- **Fix**: Check column names, watch for typos and case sensitivity

#### "Cannot convert non-finite values"
- **Cause**: NaN or infinite values in numeric columns
- **Fix**: Handle missing data before analysis

### Getting More Help

If you encounter issues not covered here:

1. **Check terminal output**: Detailed error messages appear in the terminal
2. **Review logs**: Look for Python tracebacks
3. **Test with sample data**: Use `examples/sample_baseline.csv`
4. **Simplify**: Start with simple randomization, then add complexity
5. **GitHub Issues**: Report bugs at github.com/ajolex/rct_field_flow

### Tips for Smooth Operation

1. **Start simple**: Test with small dataset first
2. **Save often**: Download results after each major step
3. **Use realistic seeds**: Pick random numbers, not 12345
4. **Check balance**: Review balance tables, aim for p > 0.10
5. **Document settings**: Take screenshots or save config files
6. **Test before field**: Run full pipeline with sample data

### Development Mode

For development/debugging:

```bash
# Enable debug mode
export STREAMLIT_LOG_LEVEL=debug

# Run with auto-reload
streamlit run rct_field_flow/app.py --server.runOnSave true
```

### System Requirements

**Minimum**:
- Python 3.9+
- 4GB RAM
- Modern browser (Chrome 90+, Firefox 88+)

**Recommended**:
- Python 3.11+
- 8GB RAM
- Chrome 100+ or Firefox 100+
- SSD storage

**For large studies (>100k observations)**:
- Python 3.11+
- 16GB RAM
- Fast CPU (multi-core helps with rerandomization)
