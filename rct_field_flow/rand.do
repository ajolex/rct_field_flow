
********************************************************************************
** 	TITLE	: ran.do
**	PURPOSE	: Stratified Randomization, and balance checks	
**	AUTHOR	: Aubrey Jolex (aubreyjolex@gmail.com)
**	DATE	: 
********************************************************************************

/********************************************************************************
    Setup Stata
*********************************************************************************/

 
 	cls
	clear 			all
	macro drop 		_all
	version 		18
	set maxvar 		32767
	set matsize     11000
	set more 		off
*********************************************************************************
	
* Set seed for reproducibility
set seed 20250128

use "C:\Users\AJolex\Desktop\panel_500brgs_AB.dta",clear

* Initialize variables
gen mostbalanced_treat = .
local minp = 1  // Track p-value across tests
local bestp = 0  // Start with 0 (most unbalanced), aim for highest p-value (most balanced)

* Run 10,000 randomizations
forvalues i = 1/5 {
    * Generate random numbers and sort within strata
    gen rand = runiform()
    sort province coastal_dummy rand, stable
    
    * Assign treatments within strata
    gen treat = .
    by province coastal_dummy: replace treat = 1 if _n<= 0.3333*_N
    by province coastal_dummy: replace treat = 2 if _n>(0.3333*_N) & _n<=(2*0.3333*_N)
    by province coastal_dummy: replace treat = 0 if treat == .
    
    * Check balance across all covariates jointly
	local bvars barangay_area population_2020 pop_dens_2020
    foreach var of local bvars  {
        qui reg `var' i.treat, robust
        qui test 1.treat 2.treat  // Test both treatments vs. control (3 as base)
        local pvalue = r(p)
        if `pvalue' < `minp' {
            local minp = `pvalue'
        }
    }
    
    * Update if this randomization is more balanced
    if `minp' > `bestp' {
        local bestp = `minp'
        replace mostbalanced_treat = treat
    }
    
		
	*Checking all random assigments to compare the results visually
	gen bestp`i' = `minp'
	gen treat`i' = treat
	
    * Clean up
    drop rand treat
	
	quietly display `i'
}

* Label treatment arms
label define treat_lbl 1 "Treatment 1" 2 "Treatment 2" 0 "Control"
label values mostbalanced_treat treat_lbl

* Display best p-value
di "Best joint p-value: `bestp'"

* Verify balance and proportions
	//balance
foreach var of local bvars {
    di "Balance check for `var':"
    reg `var' i.mostbalanced_treat, robust
    test 1.mostbalanced_treat 2.mostbalanced_treat 0.mostbalanced_treat
    di "Joint p-value = " r(p)
}

	//Proportions ~33% of the totat in each strata assigned to the treatment arm
tab mostbalanced_treat
tab province mostbalanced_treat, row
tab coastal_dummy mostbalanced_treat, row


*Export a balance table
    generate treatment_A = (mostbalanced_treat==1)
    generate treatment_B = (mostbalanced_treat==2)
	


ssc install balancetable, replace // install the user written command for exporting the balance test results

	balancetable (mean if mostbalanced_treat==0) (mean if mostbalanced_treat==1) (mean if mostbalanced_treat==2) (diff treatment_A if mostbalanced_treat!=2) (diff treatment_B if mostbalanced_treat!=1) (diff treatment_A if mostbalanced_treat!=0) `bvars' using "myfile.xls", ctitles("Mean Control" "Mean treat. A" "Mean treat. B" "Treat. A vs Control" "Treat. B vs Control" "Treat. A vs Treat. B") replace
	
	



