This detailed implementation guide covers the "Melt & Split" strategy. It includes the exact code you need to add to your app.py (or a helper module like utils_reshape.py) and the specific changes for your Streamlit UI.

Task 1: Add the Core Reshape Logic (Python)
Add this function to your project. It solves the "hundreds of files" problem by grouping variables by their depth (Level 1, Level 2...) and processing them in batches.

Action: Insert this code into app.py (or your data processing module).

Python

import re
import pandas as pd
import io
import zipfile

def automated_reshape_grouped(df: pd.DataFrame, id_col: str):
    """
    Automated N-Level Reshape using 'Melt & Split' strategy.
    
    Returns:
        dict: Keys are filenames (e.g., 'level_1_roster.csv'), Values are DataFrames.
    """
    # 1. Regex to capture Stub + Suffix (supports any depth: _1, _1_2, _1_2_3)
    pattern = re.compile(r'^(.+?)_(\d+(?:_\d+)*)$')
    
    # 2. Scan columns and Group by Depth
    stub_depths = {}
    stub_cols = {}

    for col in df.columns:
        match = pattern.match(col)
        if match:
            stub = match.group(1)
            suffix = match.group(2)
            depth = suffix.count('_') + 1
            
            # Update max depth for this stub
            if stub not in stub_depths or depth > stub_depths[stub]:
                stub_depths[stub] = depth
            
            if stub not in stub_cols:
                stub_cols[stub] = []
            stub_cols[stub].append(col)

    # Group stubs by their Max Depth
    groups = {}
    for stub, depth in stub_depths.items():
        if depth not in groups:
            groups[depth] = []
        groups[depth].append(stub)

    results = {}

    # 3. Process Each Group (Batch Reshape)
    for depth, stubs in groups.items():
        # Collect all columns for this depth group
        target_cols = []
        for s in stubs:
            target_cols.extend(stub_cols.get(s, []))
            
        if not target_cols:
            continue

        # A. Melt (Wide -> Long)
        # Use a temp index to preserve row uniqueness during melt
        df['__temp_id'] = range(len(df))
        long_df = pd.melt(
            df,
            id_vars=['__temp_id', id_col],
            value_vars=target_cols,
            var_name='temp_raw',
            value_name='value'
        )
        
        # B. Extract Stub & Suffix
        extracted = long_df['temp_raw'].str.extract(pattern)
        long_df['stub'] = extracted[0]
        long_df['suffix'] = extracted[1]
        
        # C. Split Suffix into Levels
        split_indices = long_df['suffix'].str.split('_', expand=True)
        level_cols = [f"level_{i+1}" for i in range(depth)]
        
        # Handle edge cases where split creates fewer columns than depth
        # (rare, but happens if a depth-2 group has some depth-1 vars)
        for i, col_name in enumerate(level_cols):
            if i < split_indices.shape[1]:
                long_df[col_name] = pd.to_numeric(split_indices[i], errors='ignore')
            else:
                long_df[col_name] = 1 # Default index if missing
        
        # D. Pivot (Long -> Wide-ish)
        # We pivot so Stubs become columns again
        pivot_index = ['__temp_id', id_col] + level_cols
        
        wide_df = long_df.pivot_table(
            index=pivot_index,
            columns='stub',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # E. Optimization: Drop Empty Rows
        # Drop rows where all variable columns are NaN
        data_cols = [c for c in wide_df.columns if c not in pivot_index]
        wide_df_clean = wide_df.dropna(subset=data_cols, how='all')
        
        # Cleanup
        wide_df_clean = wide_df_clean.drop(columns=['__temp_id'])
        
        # Sort for cleanliness
        wide_df_clean = wide_df_clean.sort_values([id_col] + level_cols)
        
        results[f"level_{depth}_data.csv"] = wide_df_clean

    return results
Task 2: Add Stata Code Generator (Automation)
This generates the exact Stata syntax to perform the same "Batch Reshape," preventing the "hundreds of files" issue in Stata as well.

Action: Insert this function into app.py.

Python

def generate_stata_batch_code(df: pd.DataFrame, id_col: str) -> str:
    """Generates Stata code that reshapes variables in batches based on depth."""
    
    # Reuse the scanning logic to identify groups
    pattern = re.compile(r'^(.+?)_(\d+(?:_\d+)*)$')
    stub_depths = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            stub = match.group(1)
            suffix = match.group(2)
            depth = suffix.count('_') + 1
            if stub not in stub_depths or depth > stub_depths[stub]:
                stub_depths[stub] = depth

    groups = {}
    for stub, depth in stub_depths.items():
        if depth not in groups:
            groups[depth] = []
        groups[depth].append(stub)

    # Generate Code
    code = f"// Automated Batch Reshape for {id_col}\n"
    code += f"// Generated by RCT Field Flow\n\n"
    code += "set more off\n\n"

    for depth, stubs in groups.items():
        # Create a list of stubs for the reshape command
        # We append '_' to match Stata's wildcard expectation (stub_)
        stubs_str = " ".join([f"{s}_" for s in stubs])
        clean_stubs_str = " ".join(stubs)
        
        code += f"""
* ---------------------------------------------------------
* LEVEL {depth} BATCH (e.g. {stubs[0]}...)
* ---------------------------------------------------------
preserve
    keep {id_col} {stubs_str}
    
    // 1. Reshape Long (String option handles 1_1_1 suffixes)
    reshape long {stubs_str}, i({id_col}) j(idx_string) string
    
    // 2. Drop Empty Rows (Optimization)
    // Check if all variables of interest are missing
    egen _miss_count = rowmiss({clean_stubs_str})
    drop if _miss_count == {len(stubs)}
    drop _miss_count
    
    // 3. Split Index into Levels
    split idx_string, parse(_) destring generate(level_)
    drop idx_string
    
    // 4. Rename variables (remove trailing underscore)
    rename ({stubs_str}) ({clean_stubs_str})
    
    // 5. Save
    order {id_col} level_*
    sort {id_col} level_*
    save "level_{depth}_data.dta", replace
restore

"""
    return code
Task 3: Update Streamlit UI (render_analysis)
Integrate the backend logic into your frontend.

Action: Replace the relevant section inside render_analysis() in app.py.

Python

def render_analysis():
    st.title("Data Reshaping & Analysis")
    
    uploaded_file = st.file_uploader("Upload Wide CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        # 1. ID Selection
        id_col = st.selectbox("Select Unique ID Column (e.g., hhid, uuid)", df.columns)
        
        # 2. Action Button
        if st.button("Auto-Detect & Reshape"):
            with st.spinner("Analyzing structure and reshaping..."):
                try:
                    # Run Python Reshape
                    reshaped_files = automated_reshape_grouped(df, id_col)
                    
                    # Run Stata Code Gen
                    stata_syntax = generate_stata_batch_code(df, id_col)
                    
                    st.success(f"Successfully reshaped into {len(reshaped_files)} relational datasets!")
                    
                    # 3. Tabs for Preview
                    tabs = st.tabs(list(reshaped_files.keys()))
                    for name, tab in zip(reshaped_files.keys(), tabs):
                        with tab:
                            st.dataframe(reshaped_files[name].head(20))
                            st.caption(f"Shape: {reshaped_files[name].shape}")

                    # 4. Download Everything (ZIP)
                    # Combine CSVs and Stata code into one ZIP
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        # Add CSVs
                        for name, data in reshaped_files.items():
                            csv_data = data.to_csv(index=False).encode('utf-8')
                            zf.writestr(name, csv_data)
                        
                        # Add Stata Do-file
                        zf.writestr("reshape_script.do", stata_syntax)
                        
                    st.download_button(
                        label="Download All (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="reshaped_data_package.zip",
                        mime="application/zip"
                    )
                    
                    # Display Stata Code for copy-paste
                    with st.expander("View Generated Stata Code"):
                        st.code(stata_syntax, language='stata')

                except Exception as e:
                    st.error(f"Error during reshape: {str(e)}")
Summary of Improvements
Batch Processing: Instead of looping 1,000 times for 1,000 variables, it should loop 3 times (for Level 1, 2, and 3), dramatically increasing speed.

Memory Logic: The dropna(subset=...) line ensures your "Level 2" file doesn't contain millions of empty rows for households that only have Level 1 data.

One-Click ZIP: The user gets everything (Clean CSVs + Stata Code) in a single package.