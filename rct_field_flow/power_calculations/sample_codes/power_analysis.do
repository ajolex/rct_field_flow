////////////////////////////////////////////////////////////
// Power analysis for Soft-Skills RCT (Barangay-level CRT)
// Author: RA/PI team
// Purpose: Compute PSPS-based descriptives, ICCs, DE, and MDEs
// Outcomes: labor_part (0/1), entrep_engage (0/1),
//           agency_prop (proportion or index), aspirations_gap_w (continuous)
// Clustering: brgy_code (barangay id); individual id: caseid
// Notes: Assumes PSPS-derived analysis dataset in the same folder.
////////////////////////////////////////////////////////////

// Clear state
clear all
set more off
version 15
seed 3061992

// -----------------------
// User parameters
// -----------------------
// Target design (pilot): 16 clusters total (K=8 per arm)
local K_per_arm 8
// Expected endline cluster size after attrition (~10% from ~20 baseline)
local m_endline 18
// Power settings
local alpha 0.05
local power 0.80

// -----------------------
// Load data
// -----------------------
// Expected file: analysis_data.dta (configurable) with variables:
//   cluster id (auto-detected) + outcomes: labor_part, entrep_engage, agency_prop, aspirations_gap_w
capture confirm file "analysis_data.dta"
if _rc {
    di as error "analysis_data.dta not found in current directory."
    di as error "Place the PSPS analysis dataset here and re-run."
    exit 601
}
// Allow passing alternative dataset via global DATA or default to local file
capture confirm global DATA
if _rc {
    global DATA "analysis_data.dta"
}
use "$DATA", clear

// Sanity checks for key variables
// Try to detect cluster/id variables if not standard
local cluster_var ""
foreach c in brgy_code barangay_id brgyid brgycode barangay_code cluster cluster_id psu {
    capture confirm variable `c'
    if !_rc & "`cluster_var'"=="" local cluster_var "`c'"
}
if "`cluster_var'"=="" {
    di as error "Cluster id variable not found. Tried: brgy_code barangay_id brgyid brgycode barangay_code cluster cluster_id psu"
    exit 111
}

local id_var ""
foreach c in caseid case_id id respondent_id person_id uid {
    capture confirm variable `c'
    if !_rc & "`id_var'"=="" local id_var "`c'"
}
if "`id_var'"=="" {
    di as error "Individual id variable not found. Tried: caseid case_id id respondent_id person_id uid"
    exit 111
}

// Resolve outcome variables with auto-detection and allow globals to override
// Globals (optional): LABOR, ENTREP, AGENCY, ASP
capture confirm global LABOR
if _rc local LABOR ""
capture confirm global ENTREP
if _rc local ENTREP ""
capture confirm global AGENCY
if _rc local AGENCY ""
capture confirm global ASP
if _rc local ASP ""

local labor_var = "${LABOR}"
if "`labor_var'"=="" {
    foreach c in labor_part labor lfp any_work30 paid_work30 work30 {
        capture confirm variable `c'
        if !_rc & "`labor_var'"=="" local labor_var "`c'"
    }
}
if "`labor_var'"=="" { di as error "Labor variable not found"; exit 111 }

local entre_var = "${ENTREP}"
if "`entre_var'"=="" {
    foreach c in entrep_engage entrepreneurship self_employed30 business30 {
        capture confirm variable `c'
        if !_rc & "`entre_var'"=="" local entre_var "`c'"
    }
}
if "`entre_var'"=="" { di as error "Entrepreneurship variable not found"; exit 111 }

local agency_var = "${AGENCY}"
if "`agency_var'"=="" {
    foreach c in agency_prop agency_index agency_sd agency {
        capture confirm variable `c'
        if !_rc & "`agency_var'"=="" local agency_var "`c'"
    }
}
if "`agency_var'"=="" { di as error "Agency variable not found"; exit 111 }

local asp_var = "${ASP}"
if "`asp_var'"=="" {
    foreach c in aspirations_gap_w aspirations_gap asp_gap_w asp_gap {
        capture confirm variable `c'
        if !_rc & "`asp_var'"=="" local asp_var "`c'"
    }
}
if "`asp_var'"=="" { di as error "Aspirations gap variable not found"; exit 111 }

foreach v in `cluster_var' `id_var' `labor_var' `entre_var' `agency_var' `asp_var' {
    capture confirm variable `v'
    if _rc {
        di as error "Variable `v' not found. Please ensure PSPS dataset is harmonised."
        exit 111
    }
}

// Keep analysis sample (drop missing outcomes for summaries)
preserve

// -----------------------
// Descriptives and ICCs
// -----------------------
tempfile iccstore
postfile ICCS str20 outcome double mean double sd double icc using `iccstore', replace

// Helper program: compute mean, sd, ICC using ANOVA (loneway)
program define _outcome_stats, rclass
    syntax varname [if] [in]
    quietly summarize `varlist' `if' `in'
    return scalar mean = r(mean)
    return scalar sd   = r(sd)
    // ICC via one-way ANOVA (works for continuous and binary approx)
    quietly loneway `varlist' `cluster_var'
    // loneway stores ICC in e(rho)
    return scalar icc  = e(rho)
end

// Continuous outcomes
foreach y in `agency_var' `asp_var' {
    qui count if !missing(`y')
    if r(N)>0 {
        quietly _outcome_stats `y'
        local mu = r(mean)
        local sd = r(sd)
        local icc = r(icc)
        post ICCS ("`y'") (`mu') (`sd') (`icc')
    }
}

// Binary outcomes
foreach y in `labor_var' `entre_var' {
    qui count if inlist(`y',0,1)
    if r(N)>0 {
        quietly _outcome_stats `y'
        local mu = r(mean)
        // SD for Bernoulli
        local sd = sqrt(`mu'*(1-`mu'))
        local icc = r(icc)
        post ICCS ("`y'") (`mu') (`sd') (`icc')
    }
}
postclose ICCS

use `iccstore', clear
tempfile stats
save `stats', replace

// -----------------------
// MDE calculations
// -----------------------
// Design effect DE = 1 + (m-1)*ICC
// For binary (two-arm, equal K, equal m):
//   MDE_p ≈ (z1 + z2) * sqrt( 2 * DE * p*(1-p) / (K*m) )
// For continuous (in SD units):
//   MDE_sd ≈ (z1 + z2) * sqrt( 2 * DE / (K*m) )

scalar z1 = invnormal(1-`alpha'/2)
scalar z2 = invnormal(`power')
scalar K  = `K_per_arm'
scalar m  = `m_endline'

tempname mem
postfile `mem' str20 outcome double mean double sd double icc double DE double MDE using power_summary, replace

use `stats', clear
quietly {
    gen DE = 1 + (m-1)*icc
    replace DE = 1 + (m-1)*icc in 1/L
}

tempvar mvar kvar zsum
gen double `mvar' = m
gen double `kvar' = K
gen double `zsum' = z1 + z2

quietly {
    foreach y in `labor_var' `entre_var' {
        // Filter to this outcome
        preserve
        keep if outcome=="`y'"
        if _N==1 {
            local p = mean[1]
            local de = DE[1]
            local mde = (`zsum'[1]) * sqrt( 2*`de' * `p'*(1-`p') / (K*m) )
            post `mem' ("`y'") (`p') (sqrt(`p'*(1-`p'))) (icc[1]) (`de') (`mde')
        }
        restore
    }
    foreach y in `agency_var' `asp_var' {
        preserve
        keep if outcome=="`y'"
        if _N==1 {
            local sd = sd[1]
            local de = DE[1]
            local mde = (`zsum'[1]) * sqrt( 2*`de' / (K*m) )
            post `mem' ("`y'") (mean[1]) (`sd') (icc[1]) (`de') (`mde')
        }
        restore
    }
}
postclose `mem'

use power_summary, clear
order outcome mean sd icc DE MDE
sort outcome
export delimited using "power_summary.csv", replace

// Display results
di as txt "\nPower/MDE summary (K=", %2.0f K, " per arm; m=", %2.0f m, "; alpha=", %4.2f `alpha', "; power=", %4.2f `power', ")"
list, abbreviate(20) noobs

restore
exit 0
