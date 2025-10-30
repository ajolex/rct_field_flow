
**# setup Stata
*------------------------------------------------------------------------------*
	
	cls
	clear 			all
	macro drop 		_all
	version 		17
	set maxvar 		32767
	set matsize     11000
	set more 		off
	
	set seed 		20250526
	
	
	if inlist(c(username),"AJolex") {
		global BOX "C:/Users/`c(username)'/Box"
	}
	
	else if "`c(username)'" == "Other" {
		global BOX "C:/Users/Other/Box"
	}

	gl secure "D:/04 HFCs/PSPS_Wave1_Household" //encripted folder (cryptomator)
	gl public "$BOX/Philippines Panel/01 Panel/08 Analysis & Data/04 HFCs/PSPS_Wave1_Household" //Box folder
	gl icm_box "$BOX/16566_psps_international_care_ministries_livelihoods/16566_psps_international_care_ministries_livelihoods"
	
	gl assigning_cases "$icm_box/07_Questionnaires & Data/07_ICM_Follow_up/01_assigning cases"
	gl icm_cases "F:/10_Livelihood/PSPS ICM Livelihoods Study/10_ICM Follow up survey/cases"
	
	gl survey_data "$secure/4_data/2_survey"

**Extract ICM households from PSPS
use "$survey_data/Household_linked_checked.dta", clear

		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030238", "H030037", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
drop if consent_agree != 1
cap replace consent_share_pii = consent_share_pii_1 if consent_share_pii==.
drop if consent_share_pii != 1 
	
merge 1:m caseid using "$icm_box/08_Analysis & Results/01_Data Analysis/Data/icm_hh_phaseA_selected_key_sharepii.dta"

keep if _merge==3
drop _merge
*merge m:1 caseid using "C:\Users\AJolex\Box\Philippines Panel\01 Panel\08 Analysis & Data\04 HFCs\PSPS_Wave1_Household\4_data\2_survey\psps_wave1_hh_filter.dta"
*drop _merge
*keep if nonrandom_select==1

merge m:1 caseid using "$BOX/Philippines Panel/01 Panel/08 Analysis & Data/11 Sub-study Master File/01_ICMLivelihoods_Wave1/wave1_livelihoods_participants.dta"
keep if _merge==3
drop _merge
 
*drop pilot households

drop if caseid == "H019277033" | caseid == "H019277020" | caseid == "H019820016" | ///
    caseid == "H019820034" | caseid == "H030647034" | caseid == "H030647029" | ///
    caseid == "H030015023" | caseid == "H030015019" | caseid == "H030514033" | ///
    caseid == "H030514020" | caseid == "H030670029" | caseid == "H019728022" | ///
    caseid == "H019728028" | caseid == "H019505031" | caseid == "H019505023" | ///
    caseid == "H019575028" | caseid == "H019575032" | caseid == "H030535023" | ///
    caseid == "H030535032" | caseid == "H030332019" | caseid == "H030332020"


/***	
keep caseid pull_barangay pull_municipal_city pull_province hh_resp_name pull_brgy_prefix
bysort pull_brgy_prefix: gen resp_index = _n
reshape wide hh_resp_name caseid pull_barangay pull_municipal_city pull_province, i(pull_brgy_prefix) j(resp_index)
keep hh_resp_name* caseid* pull_barangay1 pull_municipal_city1 pull_province1 pull_brgy_prefix
rename (pull_barangay1 pull_municipal_city1 pull_province1) (pull_barangay pull_municipal_city pull_province)
export delimited using "${icm_cases}/icm_respondents.csv", nolabel replace

*/

keep caseid pull_barangay pull_municipal_city pull_province pull_zone pull_brgy_prefix calc_hh_mem_name1 hh_resp_name w1_liveli_treat
renam (pull_barangay pull_municipal_city pull_province pull_zone pull_brgy_prefix) (barangay municipality province zone brgy_prefix)

gen id = caseid
gen label = ""
gen users = ""
gen formids = ""
gen hh_head = calc_hh_mem_name1

order id label barangay zone

//Team
replace users = 	"team_a"	if brgy_prefix==	"H030837"
replace users = 	"team_b"	if brgy_prefix==	"H030832"


replace label = caseid + "-" + hh_resp_name
replace formids = "ICM_follow_up_launch,ICM_Business_linked_launch" if w1_liveli_treat ~= 0
replace formids = "ICM_follow_up_launch" if w1_liveli_treat == 0 | w1_liveli_treat == .

drop if users==""

export delimited using "${icm_cases}/batch25.csv", nolabel replace

preserve
gen displayc = users+" "+"-"+" " +barangay+","+" "+ municipality
collapse (first) displayc, by(brgy_prefix)
dis displayc

sort displayc
	qui count
	forvalues i = 1/`r(N)' {
		loc case = displayc[`i']
		di "`case'"
	}
restore	



