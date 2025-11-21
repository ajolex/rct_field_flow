* import_ADB_questionnaire_test-consented-available_today-labor_g2-labor_repeat_g2.do
*
* 	Imports and aggregates "ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2" (ID: ADB_questionnaire_test) data.
*
*	Inputs:  "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2.csv"
*	Outputs: "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2.dta"
*
*	Output by SurveyCTO November 20, 2025 12:43 PM.

* initialize Stata
clear all
set more off
set mem 100m

* initialize workflow-specific parameters
*	Set overwrite_old_data to 1 if you use the review and correction
*	workflow and allow un-approving of submissions. If you do this,
*	incoming data will overwrite old data, so you won't want to make
*	changes to data in your local .dta file (such changes can be
*	overwritten with each new import).
local overwrite_old_data 0

* initialize form-specific parameters
local csvfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2.csv"
local dtafile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2.dta"
local corrfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2_corrections.csv"
local note_fields1 ""
local text_fields1 "labor_repeat_index_g2 labor_repeat_name_g2 activity_prim_occ_g2 activity_prim_ind_g2"
local date_fields1 ""
local datetime_fields1 ""

disp
disp "Starting import of: `csvfile'"
disp

* import data from primary .csv file
insheet using "`csvfile'", names clear

* drop extra table-list columns
cap drop reserved_name_for_field_*
cap drop generated_table_list_lab*

* continue only if there's at least one row of data to import
if _N>0 {
	* drop note fields (since they don't contain any real data)
	forvalues i = 1/100 {
		if "`note_fields`i''" ~= "" {
			drop `note_fields`i''
		}
	}
	
	* format date and date/time fields
	forvalues i = 1/100 {
		if "`datetime_fields`i''" ~= "" {
			foreach dtvarlist in `datetime_fields`i'' {
				cap unab dtvarlist : `dtvarlist'
				if _rc==0 {
					foreach dtvar in `dtvarlist' {
						tempvar tempdtvar
						rename `dtvar' `tempdtvar'
						gen double `dtvar'=.
						cap replace `dtvar'=clock(`tempdtvar',"MDYhms",2025)
						* automatically try without seconds, just in case
						cap replace `dtvar'=clock(`tempdtvar',"MDYhm",2025) if `dtvar'==. & `tempdtvar'~=""
						format %tc `dtvar'
						drop `tempdtvar'
					}
				}
			}
		}
		if "`date_fields`i''" ~= "" {
			foreach dtvarlist in `date_fields`i'' {
				cap unab dtvarlist : `dtvarlist'
				if _rc==0 {
					foreach dtvar in `dtvarlist' {
						tempvar tempdtvar
						rename `dtvar' `tempdtvar'
						gen double `dtvar'=.
						cap replace `dtvar'=date(`tempdtvar',"MDY",2025)
						format %td `dtvar'
						drop `tempdtvar'
					}
				}
			}
		}
	}

	* ensure that text fields are always imported as strings (with "" for missing values)
	* (note that we treat "calculate" fields as text; you can destring later if you wish)
	tempvar ismissingvar
	quietly: gen `ismissingvar'=.
	forvalues i = 1/100 {
		if "`text_fields`i''" ~= "" {
			foreach svarlist in `text_fields`i'' {
				cap unab svarlist : `svarlist'
				if _rc==0 {
					foreach stringvar in `svarlist' {
						quietly: replace `ismissingvar'=.
						quietly: cap replace `ismissingvar'=1 if `stringvar'==.
						cap tostring `stringvar', format(%100.0g) replace
						cap replace `stringvar'="" if `ismissingvar'==1
					}
				}
			}
		}
	}
	quietly: drop `ismissingvar'



	* label variables
	label variable key "Unique submission ID"
	cap label variable submissiondate "Date/time submitted"
	cap label variable formdef_version "Form version used on device"
	cap label variable review_status "Review status"
	cap label variable review_comments "Comments made during review"
	cap label variable review_corrections "Corrections made during review"


	label variable lab_note_g2 "Now I would like to speak to \${labor_repeat_name_g2}. Are they available to be "
	note lab_note_g2: "Now I would like to speak to \${labor_repeat_name_g2}. Are they available to be interviewed?"
	label define lab_note_g2 1 "Yes" 0 "No"
	label values lab_note_g2 lab_note_g2

	label variable consent_lab_g2 "Now I would like to ask you about some of your labor and domestic duties. Do you"
	note consent_lab_g2: "Now I would like to ask you about some of your labor and domestic duties. Do you give your consent to be part of this survey?"
	label define consent_lab_g2 1 "Yes" 0 "No"
	label values consent_lab_g2 consent_lab_g2

	label variable activity_yn_g2 "In the past week, did you, (\${labor_repeat_name_g2}), engage in any economic ac"
	note activity_yn_g2: "In the past week, did you, (\${labor_repeat_name_g2}), engage in any economic activity for pay or for profit in any establishment, office, farm, private home or without pay on a family farm or enterprise for at least one hour?"
	label define activity_yn_g2 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values activity_yn_g2 activity_yn_g2

	label variable activity_temp_absent_g2 "Did you have a paid job or business last week but were temporarily absent or did"
	note activity_temp_absent_g2: "Did you have a paid job or business last week but were temporarily absent or did not work at least one hour?"
	label define activity_temp_absent_g2 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values activity_temp_absent_g2 activity_temp_absent_g2

	label variable activity_n_emplymt_g2 "What is your nature of employment?"
	note activity_n_emplymt_g2: "What is your nature of employment?"
	label define activity_n_emplymt_g2 1 "Permanent" 2 "Short-term, seasonal, contract-based, casual" 3 "Worked for different employers or customers on day to day or week to week basis" -999 "Don't know" -888 "Refuse to answer"
	label values activity_n_emplymt_g2 activity_n_emplymt_g2

	label variable activity_wrk_class_g2 "What is your class of worker?"
	note activity_wrk_class_g2: "What is your class of worker?"
	label define activity_wrk_class_g2 1 "Worked for private household" 2 "Worked for private establishment" 3 "Worked for gov't/gov't-controlled corporation" 4 "Self-employed without any paid employee" 5 "Employer in own family-operated farm or business" 6 "Worked with pay in own family-operated farm or business" 7 "Worked without pay in own family-operated farm or business" -999 "Don't know" -888 "Refuse to answer"
	label values activity_wrk_class_g2 activity_wrk_class_g2

	label variable activity_prim_occ_g2 "What was your primary occupation during the past week?"
	note activity_prim_occ_g2: "What was your primary occupation during the past week?"

	label variable activity_prim_ind_g2 "In what kind of industry did you work during the past week?"
	note activity_prim_ind_g2: "In what kind of industry did you work during the past week?"

	label variable activity_prim_hrs_g2 "What is the total number of hours you worked during the past week?"
	note activity_prim_hrs_g2: "What is the total number of hours you worked during the past week?"

	label variable job_serach_yn_g2 "In the past week, did you look for a job?"
	note job_serach_yn_g2: "In the past week, did you look for a job?"
	label define job_serach_yn_g2 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values job_serach_yn_g2 job_serach_yn_g2

	label variable school_attend_yn_g2 "In the past week, did you attend school? (in-person or online learning)"
	note school_attend_yn_g2: "In the past week, did you attend school? (in-person or online learning)"
	label define school_attend_yn_g2 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values school_attend_yn_g2 school_attend_yn_g2

	label variable home_act_yn_g2 "In the past week, did you take care of the household? (For example, swept the fl"
	note home_act_yn_g2: "In the past week, did you take care of the household? (For example, swept the floor, cooked meals, took care of children, washed clothes, fixed the roof of the house, painted the wall, or other household activities)"
	label define home_act_yn_g2 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values home_act_yn_g2 home_act_yn_g2

	label variable other_act_yn_g2 "In the past week, did you do other activities excluding domestic tasks and careg"
	note other_act_yn_g2: "In the past week, did you do other activities excluding domestic tasks and caregiving? (For example social gathering, sports, patrol, community service, recitation activities, worship, or other activities)"
	label define other_act_yn_g2 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values other_act_yn_g2 other_act_yn_g2

	label variable act_most_time_g2 "Among these three activities: attending school, taking care of the household, or"
	note act_most_time_g2: "Among these three activities: attending school, taking care of the household, or other activities, which one did you spend most of your time on in the past week? Please choose 'None' if you did not spend time on any of these."
	label define act_most_time_g2 1 "Attended school" 2 "Took care of the household" 3 "Other activities" 4 "None" -999 "Don't know" -888 "Refuse to answer"
	label values act_most_time_g2 act_most_time_g2

	label variable home_act_hrs_g2 "In the last 7 days, how many hours in total did you perform homemaking or caregi"
	note home_act_hrs_g2: "In the last 7 days, how many hours in total did you perform homemaking or caregiving duties for your own home?"






	* append old, previously-imported data (if any)
	cap confirm file "`dtafile'"
	if _rc == 0 {
		* mark all new data before merging with old data
		gen new_data_row=1
		
		* pull in old data
		append using "`dtafile'"
		
		* drop duplicates in favor of old, previously-imported data if overwrite_old_data is 0
		* (alternatively drop in favor of new data if overwrite_old_data is 1)
		sort key
		by key: gen num_for_key = _N
		drop if num_for_key > 1 & ((`overwrite_old_data' == 0 & new_data_row == 1) | (`overwrite_old_data' == 1 & new_data_row ~= 1))
		drop num_for_key

		* drop new-data flag
		drop new_data_row
	}
	
	* save data to Stata format
	save "`dtafile'", replace

	* show codebook and notes
	codebook
	notes list
}

disp
disp "Finished import of: `csvfile'"
disp

* OPTIONAL: LOCALLY-APPLIED STATA CORRECTIONS
*
* Rather than using SurveyCTO's review and correction workflow, the code below can apply a list of corrections
* listed in a local .csv file. Feel free to use, ignore, or delete this code.
*
*   Corrections file path and filename:  C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-labor_g2-labor_repeat_g2_corrections.csv
*
*   Corrections file columns (in order): key, fieldname, value, notes

capture confirm file "`corrfile'"
if _rc==0 {
	disp
	disp "Starting application of corrections in: `corrfile'"
	disp

	* save primary data in memory
	preserve

	* load corrections
	insheet using "`corrfile'", names clear
	
	if _N>0 {
		* number all rows (with +1 offset so that it matches row numbers in Excel)
		gen rownum=_n+1
		
		* drop notes field (for information only)
		drop notes
		
		* make sure that all values are in string format to start
		gen origvalue=value
		tostring value, format(%100.0g) replace
		cap replace value="" if origvalue==.
		drop origvalue
		replace value=trim(value)
		
		* correct field names to match Stata field names (lowercase, drop -'s and .'s)
		replace fieldname=lower(subinstr(subinstr(fieldname,"-","",.),".","",.))
		
		* format date and date/time fields (taking account of possible wildcards for repeat groups)
		forvalues i = 1/100 {
			if "`datetime_fields`i''" ~= "" {
				foreach dtvar in `datetime_fields`i'' {
					* skip fields that aren't yet in the data
					cap unab dtvarignore : `dtvar'
					if _rc==0 {
						gen origvalue=value
						replace value=string(clock(value,"MDYhms",2025),"%25.0g") if strmatch(fieldname,"`dtvar'")
						* allow for cases where seconds haven't been specified
						replace value=string(clock(origvalue,"MDYhm",2025),"%25.0g") if strmatch(fieldname,"`dtvar'") & value=="." & origvalue~="."
						drop origvalue
					}
				}
			}
			if "`date_fields`i''" ~= "" {
				foreach dtvar in `date_fields`i'' {
					* skip fields that aren't yet in the data
					cap unab dtvarignore : `dtvar'
					if _rc==0 {
						replace value=string(clock(value,"MDY",2025),"%25.0g") if strmatch(fieldname,"`dtvar'")
					}
				}
			}
		}

		* write out a temp file with the commands necessary to apply each correction
		tempfile tempdo
		file open dofile using "`tempdo'", write replace
		local N = _N
		forvalues i = 1/`N' {
			local fieldnameval=fieldname[`i']
			local valueval=value[`i']
			local keyval=key[`i']
			local rownumval=rownum[`i']
			file write dofile `"cap replace `fieldnameval'="`valueval'" if key=="`keyval'""' _n
			file write dofile `"if _rc ~= 0 {"' _n
			if "`valueval'" == "" {
				file write dofile _tab `"cap replace `fieldnameval'=. if key=="`keyval'""' _n
			}
			else {
				file write dofile _tab `"cap replace `fieldnameval'=`valueval' if key=="`keyval'""' _n
			}
			file write dofile _tab `"if _rc ~= 0 {"' _n
			file write dofile _tab _tab `"disp"' _n
			file write dofile _tab _tab `"disp "CAN'T APPLY CORRECTION IN ROW #`rownumval'""' _n
			file write dofile _tab _tab `"disp"' _n
			file write dofile _tab `"}"' _n
			file write dofile `"}"' _n
		}
		file close dofile
	
		* restore primary data
		restore
		
		* execute the .do file to actually apply all corrections
		do "`tempdo'"

		* re-save data
		save "`dtafile'", replace
	}
	else {
		* restore primary data		
		restore
	}

	disp
	disp "Finished applying corrections in: `corrfile'"
	disp
}
