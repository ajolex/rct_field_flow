/*
================================================================================
RANDOMIZATION CODE - RCT Field Flow (Stata Implementation)
================================================================================
Generated: 2025-11-09 17:10:07
Method: stratified

RANDOMIZATION SPECIFICATION:
  - ID Column: barangay_id
  - Treatment Column: treatment
  - Method: Stratified randomization: Within-stratum random assignment (strata: province, coastal_dummy)
  - Random Seed: 20250128 (for exact replication)

TREATMENT ARMS:
  - control: 33.3% (0.3333 proportion)
  - treatment_A: 33.3% (0.3333 proportion)
  - treatment_B: 33.3% (0.3334 proportion)

* RERANDOMIZATION APPROACH:
*   - Total iterations: 10000
*   - Each iteration generates a new random assignment
*   - Balance measured using ANOVA F-tests for specified covariates
*   - Assignment with best balance (highest minimum p-value) is selected
*   - Higher p-values indicate better balance across treatment arms
*   - This approach follows Morgan & Rubin (2012) methodology
*   - Note: Rerandomization affects inference (see Bruhn & McKenzie 2009)

* BALANCE CHECKING:
*   - Covariates: barangay_area, population_2020, pop_dens_2020
*   - Method: One-way ANOVA F-test (oneway command in Stata)
*   - Null hypothesis: All treatment arms have equal means
*   - P-value interpretation: >0.05 indicates acceptable balance

IMPORTANT ASSUMPTIONS:
  1. Random seed ensures reproducibility (set seed 20250128)
  2. runiform() generates uniform [0,1] random numbers
  3. Treatment proportions are approximate targets
  4. For stratified: randomization done separately within each stratum
  5. For cluster: all units in same cluster receive same treatment
  6. Missing values in key variables will cause errors - clean data first

RANDOMIZATION METHOD DETAILS:
  - Uses Stata's runiform() which generates pseudo-random uniform variates
  - Treatment assignment based on cumulative probability thresholds
  - For N arms with proportions p1, p2, ..., pN (summing to 1.0):
      * Arm 1 assigned if random_draw < p1
      * Arm 2 assigned if p1 <= random_draw < (p1+p2)
      * Arm k assigned if sum(p1...p(k-1)) <= random_draw < sum(p1...pk)
  - This ensures expected proportions match specified proportions

REFERENCES:
  - Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate
    balance in experiments. Annals of Statistics, 40(2), 1263-1282.
  - Bruhn, M., & McKenzie, D. (2009). In pursuit of balance: Randomization in
    practice in development field experiments. American Economic Journal: 
    Applied Economics, 1(4), 200-232.

================================================================================
*/

clear all
set more off
set seed 20250128

di _n(2) "================================================================================"
di "RANDOMIZATION SCRIPT - RCT Field Flow"
di "================================================================================"
di "Date: 2025-11-09 17:10:07"
di "Method: stratified"
di "Seed: 20250128"
di "================================================================================" _n

* Load baseline data
di "Loading baseline data..."
import delimited "baseline_data.csv", clear  // Update with your actual file path
di "Observations loaded: " _N _n

* Validate data
di "Validating data..." _n
count if missing(barangay_id)
if r(N) > 0 {
    di as error "WARNING: " r(N) " observations have missing ID (barangay_id)"
}

* Check province (stratum variable)
count if missing(province)
if r(N) > 0 {
    di as error "WARNING: " r(N) " observations have missing province"
}

* Check coastal_dummy (stratum variable)
count if missing(coastal_dummy)
if r(N) > 0 {
    di as error "WARNING: " r(N) " observations have missing coastal_dummy"
}

* Stratified by: province, coastal_dummy

di _n "================================================================================"
di "TREATMENT ARMS: control (33.3%), treatment_A (33.3%), treatment_B (33.3%)"
di "================================================================================" _n


* RERANDOMIZATION WITH BALANCE OPTIMIZATION
* Initialize tracking variables
gen mostbalanced_treatment = .
local minp = 1           // Track minimum p-value across covariates in each iteration
local bestp = 0          // Track best (highest) minimum p-value across all iterations
local best_iter = 0      // Track which iteration gave best balance

di _n "================================================================================"
di "RUNNING 10,000 RANDOMIZATION ITERATIONS"
di "================================================================================"
di "Finding assignment with best balance across covariates..."
di "Balance covariates: barangay_area, population_2020, pop_dens_2020" _n

* Run 10,000 randomization iterations
forvalues i = 1/10000 {
    * Display progress every 1000 iterations
    if mod(`i', 1000) == 0 {
        di "  Iteration " %6.0f `i' " / 10,000 (best p-value so far: " %6.4f `bestp' ")"
    }
    
    * Generate random numbers and sort within strata
    gen rand = runiform()
    sort province coastal_dummy rand, stable
    
    * Assign treatments within strata
    gen treatment_temp = ""
    bysort province coastal_dummy: replace treatment_temp = "control" if _n <= 0.333300 * _N
    bysort province coastal_dummy: replace treatment_temp = "treatment_A" if _n <= 0.666600 * _N
    replace treatment_temp = "treatment_B" if treatment_temp == ""
    
    * Reset minimum p-value for this iteration
    local minp = 1
    
    * Check balance across all specified covariates
    foreach var of varlist barangay_area population_2020 pop_dens_2020 {
        qui reg `var' i.treatment_temp, robust
        qui test 1.treatment_temp 2.treatment_temp
        local pvalue = r(p)
        
        * Track the minimum p-value across all covariates in this iteration
        if `pvalue' < `minp' {
            local minp = `pvalue'
        }
    }
    
    * Update if this randomization has better (higher) minimum p-value
    if `minp' > `bestp' {
        local bestp = `minp'
        local best_iter = `i'
        replace mostbalanced_treatment = treatment_temp
    }
    
    * Clean up temporary variables
    drop rand treatment_temp
}

di _n "================================================================================"
di "RERANDOMIZATION COMPLETE"
di "================================================================================"
di "Best iteration: " `best_iter' " out of 10,000"
di "Best min p-value: " %6.4f `bestp'
di "  → Higher p-values indicate better balance"
di "  → This is the MINIMUM p-value across all tested covariates"
di "  → Selected assignment has best overall balance" _n

* Use the best balanced assignment
gen treatment = mostbalanced_treatment
drop mostbalanced_treatment

* Label treatment arms
label define treatment_lbl 0 "control" 1 "treatment_A" 2 "treatment_B"
label values treatment treatment_lbl

* Treatment distribution
tab treatment

* Treatment distribution by strata
table province coastal_dummy treatment

* Balance checks
foreach var of varlist barangay_area population_2020 pop_dens_2020 {
    di _n "Balance check for `var':"
    oneway `var' treatment, tabulate
}

di _n "================================================================================"
di "TREATMENT DISTRIBUTION SUMMARY"
di "================================================================================" _n

* Overall distribution
tab treatment, missing

* Calculate and display proportions
di _n "Treatment proportions:"
foreach arm in "control" "treatment_A" "treatment_B" {
    count if treatment == `arm'
    local n_`arm' = r(N)
    local pct_`arm' = (r(N) / _N) * 100
    di "  `arm': " r(N) " observations (" %4.1f `pct_`arm'' "%)"
}


di _n "Treatment distribution by strata:"
table province coastal_dummy treatment, row col


di _n "================================================================================"
di "BALANCE CHECK SUMMARY"
di "================================================================================" _n

* Check balance on specified covariates
foreach var of varlist barangay_area population_2020 pop_dens_2020 {
    di _n "Balance check for `var':"
    di "  Null hypothesis: Treatment arms have equal means"
    oneway `var' treatment, tabulate
    
    * Highlight imbalance
    if r(p) < 0.05 {
        di as error "  ⚠ WARNING: Significant imbalance detected (p = " %6.4f r(p) ")"
    }
    else {
        di as text "  ✓ Acceptable balance (p = " %6.4f r(p) ")"
    }
}

di _n "================================================================================"
di "SAVING RESULTS"
di "================================================================================" _n

* Save randomized assignments
export delimited using "randomized_assignments.csv", replace
di "Assignments saved to: randomized_assignments.csv"
di "Total observations: " _N
di "Treatment column: treatment"

di _n "================================================================================"
di "RANDOMIZATION COMPLETE"
di "================================================================================"
di "Review the output above to verify:"
di "  1. Treatment proportions match specified targets"
di "  2. No excessive missing data warnings"
di "  3. Balance tests show acceptable p-values (>0.05 preferred)"
di "  4. All N(province) x N(coastal_dummy) strata combinations have observations"

di "================================================================================" _n
