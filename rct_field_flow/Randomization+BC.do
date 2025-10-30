*-----------------------------------------------------------------------------
***STRATIFIED RANDOMISATION FOR BARANGAYS***
*Muskan Aggarwal
*Datasets: A_brgy_ocular_edited, panel_500brgs_phaseABC_replacements_20231012
*-----------------------------------------------------------------------------
clear
set more off 
version 17.0
global data "C:\Users\itr9375\OneDrive - Northwestern University\Desktop\Aspirations RCT\data"
cd "${data}\"

use panel_500brgs_phaseABC_replacements_20231012.dta, clear
rename ADM4_PCODE adm4_pcode
rename barangay barangay1
drop _merge
drop rand rand2 strata
save barangays_new.dta, replace

use A_brgy_ocular_edited.dta, clear
rename barangay barangay2
merge 1:1 adm4_pcode using barangays_new.dta
drop _merge
save barangays_final.dta, replace

use barangays_final.dta, clear

**********To drop the barangays not selected for panel data***********
count if inlist(phase_abc_20231012,2,3) & province != "AKLAN" 
*318 barangays-same as in sample frame 
drop if phase_abc_20231012 == . | phase_abc_20231012 == 1
drop if province == "AKLAN"
tab province
*318 observations

*BASE
encode (icm_base_branch), gen(branch_no)
gen base_no = branch_no
recode base_no (1/3 =1) (4/6 =2) (7/9=3)


*MIN TIME TO MUNICIPALITY
gen min_tpt_time = min(tpt_time_1,tpt_time_2,tpt_time_3,tpt_time_4, tpt_time_5,tpt_time_6)
replace min_tpt_time =. if min_tpt_time == -999
label variable min_tpt_time "Minimum time for travelling to nearest municipality (in minutes)"
gen time_to_muni = min_tpt_time
*11 missing values; Spread also has 8 missing values

*MOBILE SIGNAL
gen mobile_signal = signal_1
gen mobile_signal_0 = 0
gen mobile_signal_1 = 0
replace mobile_signal_0 = 1 if mobile_signal == 0
replace mobile_signal_1 = 1 if mobile_signal == 3


*INTERNET SIGNAL 
gen internet_signal = signal_2
gen internet_signal_0 = 0
gen internet_signal_1 = 0
replace internet_signal_0 = 1 if internet_signal == 0
replace internet_signal_1 = 1 if internet_signal == 3

// stratifies on: base_no
// balances on: population_2020, pop_dens_2020, spread, pcnt_4p, time_to_muni, mobile_signal, internet_signal
isid adm4_pcode	//assert unique ID			
sort adm4_pcode	// sort on unique ID			
version 17: set seed 71605    //seed taken from calculator.net (range:1-100000)

*Generate necessary vars:
gen m4d_treatment =.
gen besttreatment_num =.

*Initialise locals
loc p1 0						
loc p2 0 
loc p3 0
loc p4 0
loc p5 0
loc p6 0
loc p7 0
loc p8 0
loc p9 0
loc minp 0 
loc minp1 0 
loc minp2 0 
loc minp3 0
loc minp4 0
loc minp5 0
loc minp6 0
loc minp7 0
loc minp8 0
loc minp9 0


*Generate strata using stratifcation variable: province_no 
sort base_no
egen strata=group(base_no)
tab strata, missing
sort strata


quietly forvalues x = 1/10000 {		//iterate fixed # of times	

sort adm4_pcode	                    // sort on unique ID			
randtreat, generate(m4d_treatment) replace unequal(7/25 6/25 6/25 6/25) strata(base_no) misfits(wglobal) setseed(10202)  //seed taken from calculator.net (range:1-100000)

*Captures p-values from F-tests
reg population_2020 i.m4d_treatment
testparm i.m4d_treatment
loc p1=r(p) 

reg pop_dens_2020 i.m4d_treatment
testparm i.m4d_treatment
loc p2=r(p) 

reg spread i.m4d_treatment
testparm i.m4d_treatment
loc p3=r(p) 

reg pcnt_4p i.m4d_treatment
testparm i.m4d_treatment
loc p4=r(p)

reg time_to_muni i.m4d_treatment
testparm i.m4d_treatment
loc p5=r(p)

reg mobile_signal_0 i.m4d_treatment
testparm i.m4d_treatment
loc p6=r(p)

reg internet_signal_0 i.m4d_treatment
testparm i.m4d_treatment
loc p7 = r(p)
  
reg mobile_signal_1 i.m4d_treatment
testparm i.m4d_treatment
loc p8=r(p)

reg internet_signal_1 i.m4d_treatment
testparm i.m4d_treatment
loc p9 = r(p)
  
*Compare this randomization's p-values with the previous "most balanced" randomization
if min(`p1', `p2', `p3', `p4', `p5', `p6', `p7', `p8', `p9') > `minp' {	 

loc minp1 = `p1'				 

loc minp2 = `p2' 

loc minp3 = `p3'

loc minp4 = `p4'

loc minp5 = `p5'

loc minp6 = `p6'

loc minp7 = `p7'

loc minp8 = `p8'

loc minp9 = `p9'

loc minp = min(`minp1', `minp2', `minp3', `minp4', `minp5', `minp6', `minp7', `minp8', `minp9') 

*Store treatment assignments associated with observations
replace besttreatment_num=m4d_treatment
 
} 

noisily display `x'		//visual counter

} 

// this is the best treatment assignment – maximizes the minimum // F-Test p-value 

display "p1: `minp1'" 

display "p2: `minp2'" 

display "p3: `minp3'"

display "p4: `minp4'"

display "p5: `minp5'"

display "p6: `minp6'"

display "p7: `minp7'"

display "p8: `minp8'"

display "p9: `minp9'"

*The treatment assignments are stored under besttreatment_num
tab besttreatment_num
sort besttreatment_num


label define treat 0 "control" 1 "Disjoint small groups only" 2 "Conjoint small groups only" 3 "Conjoint small groups and community viewing" 
label values besttreatment_num treat
sort besttreatment_num base_no province barangay1
save randomisation_bgydata_23042024.dta, replace

******BALANCE CHECK
ssc install orth_out
clear
use randomisation_bgydata_23042024.dta
orth_out base_no population_2020 pop_dens_2020 spread pcnt_4p time_to_muni mobile_signal_0 mobile_signal_1 internet_signal_0 internet_signal_1 using "$data/balance_check_2304" , by(besttreatment_num) se compare test count replace

