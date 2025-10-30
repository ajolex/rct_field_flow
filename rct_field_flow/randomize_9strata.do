//Author: Aubrey Jolex/Mariel Felizardo
//Purpose: Selection of seed 
//Date: April 16, 2025
*******************************************************************************
clear
set more off 
version 18 
macro drop 	_all
set matsize 11000
******************************************************************************
log using "C:\Users\RMPanti\Box\00_NEW_FOLDERS\16547_psps_aspirations\08 Analysis & Data\03_Randomization\01_Barangay treatment assignments\02_data\brgyselectionseed_9strata", replace text

/* this do-file is for selecting the best seed

Datasets used and why they were used: 
1. Panel 500 - contains the variables population 2020, population density 2020, distance of barangays from city/proper, %4ps, ICM base branch
2. Barangay Ocular - contains variables transport time, mobile signal, internet signal, spread 
3. Aspirations Randomization - contains the assignments of barangays 

Variables for stratification:
1. ICM base - original randomization stratified on this 
2. ICM arm  - new variable for stratification 

Variables for balancing on:
1. Population 2020
2. Population Density 2020
3. Spread
4. Distance of barangay to city/proper in terms of travel time
5. %4Ps 
6. Mobile signal - created two dummy variables: with mobile signal and no mobile signal
7. Internet signal - created two dummy variables: with internet signal and no internet signal 

Other requirements: 
-222 must have the same proportions of the Aspirations arms*/
*******************************************************************************
//file paths 

	loc cd="C:\Users\RMPanti\Box\00_NEW_FOLDERS\16547_psps_aspirations\08 Analysis & Data"
	
	gl rand_do "`cd'\03_Randomization\01_Barangay treatment assignments\01_dofile"
	gl rand_dta "`cd'\03_Randomization\01_Barangay treatment assignments\02_data"
	
	gl panel_500 "C:\Users\RMPanti\Box\Philippines Panel\01 Panel\08 Analysis & Data\01 Inputs for Panel sampling framework\data\panel_500brgs_phaseABC_replacements_20231012.dta"
	gl ocular "C:\Users\RMPanti\Box\00_NEW_FOLDERS\16566_psps_international_care_ministries_livelihoods\16566_psps_international_care_ministries_livelihoods\04_Research Design\02_Randomization\01_data\A_brgy_ocular_edited.dta" //dataset that Muskan used 
	
********************************************************************************
//Part 1: organizing dataset to be used for the random selection 
	
**1.A. Merging datasets and keeping relevant variables

//Panel 500
	clear
	use "$panel_500"
	drop _merge
	save "$rand_dta\panel_500_2.dta", replace 

//Aspirations treatment arms 
	clear
	use "$rand_dta\Aspirations_PhaseBC_Phase_with arms.dta"
	rename adm4_pcode ADM4_PCODE
	merge 1:1 ADM4_PCODE using "$rand_dta\panel_500_2.dta"
	drop if _merge==2 
	keep ADM4_PCODE barangay city_municipality province phase_abc_20231012 brgy_prefix besttreatment_num population_2020 pop_dens_2020 dist_city_metre pcnt_4p icm_base_branch
	save "$rand_dta\Asp_midline_dtaset.dta", replace  //final dataset to be used 
	
//Barangay List Ocular 
	clear
	use "$ocular"	
	keep tpt_time_* ADM4_PCODE common_lang signal_1 m_signal signal_2 spread
	replace spread=. if spread==-999
	merge 1:1 ADM4_PCODE using "$rand_dta\Asp_midline_dtaset.dta" //8 missing barangays 
	drop if _merge==1 
	rename _merge merge_ocular 
	save "$rand_dta\Asp_midline_dtaset.dta", replace  //final dataset to be used 

//ICM Randomization-imported as excel and saving is as Stata data
	clear 
	import excel using "$rand_dta\_Other Studies\ICM_Rand_PhaseBC.xlsx", firstrow 
	save "$rand_dta\_Other Studies\ICM_Rand_PhaseBC.dta", replace 

**Here I am just merging the dataset with the ICM Randomization 
	clear
	use "$rand_dta\Asp_midline_dtaset.dta"
	merge 1:1 ADM4_PCODE using "$rand_dta\_Other Studies\ICM_Rand_PhaseBC.dta" 
	drop if _merge==2 
	rename _merge merge_ICM
	sort merge_ICM
	rename treatment_lvh ICM_treatment 
	replace ICM_treatment= "non-ICM barangay" if ICM_treatment==""
	save "$rand_dta\Asp_midline_dtaset.dta", replace //Final dataset to be used 

	

	
**1.B. Generating variables to balance on 

//time to municipality 
	**In the original, the -999 was only replaced after taking the min time to travel and putting them under one variable. Two observations under that variable were -999 when there were actually responses. This line ensures that Stata will take the travel time instead of the -999. All the other lines are taken from the original randomization 
	
	forval i=1/6 {	
		replace tpt_time_`i'=. if  tpt_time_`i'==-999			
	}
	
	gen min_tpt_time = min(tpt_time_1,tpt_time_2,tpt_time_3,tpt_time_4, tpt_time_5,tpt_time_6) 
	label variable min_tpt_time "Minimum time for travelling to nearest municipality (in minutes)"	
	gen time_to_muni = min_tpt_time


//Dummy variables for mobile signal 

	gen mobile_signal = signal_1
	gen mobile_signal_0 = 0
	gen mobile_signal_1 = 0
	replace mobile_signal_0 = 1 if mobile_signal == 0
	replace mobile_signal_1 = 1 if mobile_signal == 3


//Dummy variables for internet signal 
	gen internet_signal = signal_2
	gen internet_signal_0 = 0
	gen internet_signal_1 = 0
	replace internet_signal_0 = 1 if internet_signal == 0
	replace internet_signal_1 = 1 if internet_signal == 3


	
//Recoding base no and ICM treatment for stratification 	

*BASE
encode (icm_base_branch), gen(branch_no)
gen base_no = branch_no
recode base_no (1/3 =1) (4/6 =2) (7/9=3)

*ICM Treatment -- new variable to stratify on 
**There are other non-ICM barangays and this also generated a another strata 
encode (ICM_treatment), gen(ICM_arm) 
save "$rand_dta\Asp_midline_dtaset.dta", replace
	
********************************************************************************	
	
//Part 2: Stratification 
clear 
use "$rand_dta\Asp_midline_dtaset.dta"
isid ADM4_PCODE	//assert unique ID			
sort ADM4_PCODE //sort on unique ID			
version 18


//Erin's suggestion - 3x3 strata

tab ICM_arm
gen ICM_brgy_randomization = ICM_arm 
recode ICM_brgy_randomization (1=1) (2/4=2) (5=3)
tab ICM_brgy_randomization
label define icm_brgy 1 "ICM Control" 2 "ICM Treatment" 3 "Non-ICM barangay"
label values ICM_brgy_randomization icm_brgy




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


*Generate necessary vars:
gen m4d_treatment =.
gen ss_midline=.

loc best_seed 0
global best_diff= 9999





*Generate strata using stratification variables: ICM arm and base
sort base_no ICM_brgy_randomization, stable
egen strata=group(base_no ICM_brgy_randomization) //
tab strata, missing
sort strata, stable  

order base_no ICM_brgy_randomization, before (strata)

label define strat_values 1 "base 1 and ICM Control" 2 "base 1 and ICM treatment" 3 "base 1 and non-ICM barangay" 4 "base 2 and ICM control" 5 "base 2 and ICM treatment" 6 "base 2 and non-ICM barangay" 7 "base 3 and ICM control" 8 "base 3 and ICM treatment" 9 "base 3 and non-ICM barangay"
label values strata strat_values

tab strata besttreatment_num, row


tab besttreatment_num, matcell(freq)
matrix p_prop= freq/r(N)



quietly forvalues x = 1/10000 {		//iterate fixed # of times	
sort ADM4_PCODE                    // sort on unique ID			


loc seed= round(runiform()*1000000)

randtreat, generate (m4d_treatment) replace unequal(96/318 222/318) strata(base_no ICM_brgy_randomization) misfits(wglobal) setseed (`seed') 

tab besttreatment_num if m4d_treatment==1, matcell(m_freq)
matrix mid_prop= m_freq/r(N)

loc diff=0

forval i=1/4 {
	
	loc diff=`diff' + abs(mid_prop[`i',1] - p_prop[`i',1])
	
	
}

*Captures p-values from F-tests
reg population_2020 m4d_treatment
testparm m4d_treatment
loc p1=r(p) 

reg pop_dens_2020 m4d_treatment
testparm m4d_treatment
loc p2=r(p) 

reg spread m4d_treatment
testparm m4d_treatment
loc p3=r(p) 

reg pcnt_4p m4d_treatment
testparm m4d_treatment
loc p4=r(p)

reg time_to_muni m4d_treatment
testparm m4d_treatment
loc p5=r(p)

reg mobile_signal_0 m4d_treatment
testparm m4d_treatment
loc p6=r(p)

reg internet_signal_0 m4d_treatment
testparm m4d_treatment
loc p7 = r(p)
  
reg mobile_signal_1 m4d_treatment
testparm m4d_treatment
loc p8=r(p)

reg internet_signal_1 m4d_treatment
testparm m4d_treatment
loc p9 = r(p)
  
*Compare this randomization's p-values with the previous "most balanced" randomization
scalar min_p=min(`p1', `p2', `p3', `p4', `p5', `p6', `p7', `p8', `p9')

if min_p > `minp' & `diff'< $best_diff {

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

global best_diff= `diff'

loc best_seed = `seed'

*Store treatment assignments associated with observations
replace ss_midline=m4d_treatment
 
} 

quietly display `x'		//visual counter

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

display "best_diff: $best_diff"

display "best_seed: `best_seed'"


*Selected barangays are stored under ss_mdiline

sort ss_midline
tab strata besttreatment_num if ss_midline==1
tab besttreatment_num if ss_midline==1
tab province if ss_midline==1


label define midline_ss 0 "non-midline barangay" 1 "midline barangay" 
label values ss_midline midline_ss
sort ss_midline besttreatment_num base_no ICM_arm brgy_prefix province city_municipality barangay




cap log close 



******BALANCE CHECK
/*ssc install orth_out
clear
use "$rand_dta\midlinesample_2478.dta"
orth_out i.besttreatment_num population_2020 pop_dens_2020 spread pcnt_4p time_to_muni mobile_signal_0 mobile_signal_1 internet_signal_0 internet_signal_1 using "$rand_dta\midline_balance_check.xlsx", by(ss_midline) se compare test count replace*/


/*clear
use "$rand_dta\midlinesample.2478.dta"
tab strata if ss_midline==0 
keep ADM4_PCODE brgy_prefix province city_municipality barangay branch_no ICM_arm strata besttreatment_num ss_midline
order ss_midline, after (strata)
export excel using "$rand_dta\midline_sample.2478.xlsx",  replace firstrow (variables)



