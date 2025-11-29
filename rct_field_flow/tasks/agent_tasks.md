Master Implementation Plan: RCT Analysis Engine Upgrade

Objective: Transform the current Analysis Page into a comprehensive RCT evaluation tool. Move beyond simple replication to a generalized framework capable of handling ATE, ITT, LATE, and TOT methodologies, using the Bruhn & Karlan paper as the foundational learning dataset and other replication resources from "C:\Users\AJolex\Documents\rct_field_flow\examples\replication" possibly for the other methods like ITT, LATE, and TOT.

Phase 1: Knowledge Ingestion & Pattern Recognition

Goal: Deconstruct the provided "Gold Standard" analysis to understand the statistical logic and workflow.

[ ] Ingest Reference Material:

Context: Read WPS6508.txt (The Paper) to understand the research hypothesis, experimental design, and statistical claims.

Code Logic: Parse C:\Users\AJolex\Documents\rct_field_flow\examples\bruh_karlan_replication\do\bruhn_karlan_schoar_replication_final.do.

Action: Map specific Stata commands (e.g., reg, ivregress, areg) to Python equivalents (Statsmodels/Linearmodels).

Focus: Identify how they handle standard errors, clustering, and fixed effects.

Data Structure: Analyze the schema of C:\Users\AJolex\Documents\rct_field_flow\examples\bruh_karlan_replication\data.

[ ] Additional resources - also very important "C:\Users\AJolex\Documents\rct_field_flow\examples\replication:

Action: Get extra insights for other rct evaluation methodologies from this comprehensive resource and include options to handle ITT, LATE, and TOT methodologies.

Phase 2: Infrastructure & Data Pipeline

Goal: Update the backend to support Stata-native files and flexible data loading.

[ ] Implement .dta Support:

Task: Upgrade the file uploader/ingestion engine to accept .dta (Stata) files alongside CSV/Excel.

Tech Stack: Utilize pandas.read_stata() ensuring preservation of categorical labels and variable metadata.

[ ] Standardize Data Cleaning:

Task: Create a preprocessing pipeline that mimics the cleaning steps found in the .do file (handling missing values, generating dummy variables for randomization strata) but generalizes them for any uploaded dataset.

Phase 3: Generalized Statistical Engine (The Core Upgrade)

Goal: Abstract the logic from the replication files into reusable Python functions.

[ ] Develop Balance Check Module:

Insight: RCTs always require a check for orthogonality (did the randomization work?).

Task: Create a function to generate "Table 1" (Balance Table) automatically, comparing means of covariates across Treatment and Control groups with t-tests/F-tests.

[ ] Implement Estimator Classes:

ITT (Intention to Treat): Build a standard OLS regression function:


$$Y = \alpha + \beta T + \epsilon$$

TOT (Treatment on Treated): Build an Instrumental Variable (IV) function where Assignment ($Z$) is the instrument for Take-up ($D$).

LATE (Local Average Treatment Effect): Implement logic for handling non-compliance (using 2SLS - Two-Stage Least Squares).

[ ] Robustness & Clustering:

Task: Ensure all regression models allow for clustered standard errors (as usually required in RCTs similar to the Karlan example).

Phase 4: Analysis Page UI/UX Overhaul

Goal: Present the complex statistics in an actionable, "best-practice" dashboard.

[ ] Dynamic Model Selection:

UI Update: Create a control panel allowing the user to select their analysis type (ITT vs. TOT vs. LATE) and define their variables (Outcome, Treatment, Instrument, Clusters).

[ ] Results Visualization:

Task: Instead of raw console output, render regression tables (coefficient, p-value, confidence intervals) in formatted Markdown or interactive tables (AgGrid).

Visualization: Add plotting features (e.g., Coefficient plots or Distribution plots of Outcome vs. Control).

[ ] Interpretation Assistant:

Task: Add a text generation layer that translates the statistical output into plain English (e.g., "The treatment increased the outcome by X units, which is statistically significant at the 5% level..."), mirroring the language style used in WPS6508.txt.