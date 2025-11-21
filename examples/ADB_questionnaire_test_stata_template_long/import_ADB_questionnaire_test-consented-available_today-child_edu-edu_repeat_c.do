* import_ADB_questionnaire_test-consented-available_today-child_edu-edu_repeat_c.do
*
* 	Imports and aggregates "ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c" (ID: ADB_questionnaire_test) data.
*
*	Inputs:  "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c.csv"
*	Outputs: "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c.dta"
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
local csvfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c.csv"
local dtafile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c.dta"
local corrfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c_corrections.csv"
local note_fields1 ""
local text_fields1 "edu_repeat_index_c edu_repeat_name_c edu_repeat_age_c b_year_c lrn_c prov_start_edu_c prov_start_edu_oth_c munic_start_edu_c munic_start_edu_oth_c brngy_start_edu_c brngy_start_edu_oth_c"
local text_fields2 "edu_attended_oth_c edu_completed_oth_c primary_school_c_count interruption_primary_repeat_c_co prim_int_count_c junior_highschool_c_count interruption_junior_hs_repeat_c_ jhs_int_count_c"
local text_fields3 "senior_hs_c_count interruption_senior_hs_repeat_c_ shs_int_count_c hs_c_count interruption_hs_repeat_c_count hs_int_count_c instr_types instr_types_oth count_instr_types instr_lbl"
local text_fields4 "covid_instr_repeat_count"
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


	label variable lrn_c "What is \${edu_repeat_name_c}'s learner reference number (LRN)?"
	note lrn_c: "What is \${edu_repeat_name_c}'s learner reference number (LRN)?"

	label variable edu_curriculum_c "During \${edu_repeat_name_c}'s time of study, were they under the K-12 curriculu"
	note edu_curriculum_c: "During \${edu_repeat_name_c}'s time of study, were they under the K-12 curriculum?"
	label define edu_curriculum_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values edu_curriculum_c edu_curriculum_c

	label variable age_start_edu_c "How old was \${edu_repeat_name_c} when they first attended school?"
	note age_start_edu_c: "How old was \${edu_repeat_name_c} when they first attended school?"

	label variable prov_start_edu_c "Which province was \${edu_repeat_name_c} living when they started school?"
	note prov_start_edu_c: "Which province was \${edu_repeat_name_c} living when they started school?"

	label variable prov_start_edu_oth_c "Please specify the province:"
	note prov_start_edu_oth_c: "Please specify the province:"

	label variable munic_start_edu_c "Which municipality was \${edu_repeat_name_c} living when they first started scho"
	note munic_start_edu_c: "Which municipality was \${edu_repeat_name_c} living when they first started school?"

	label variable munic_start_edu_oth_c "Please specify the municipality:"
	note munic_start_edu_oth_c: "Please specify the municipality:"

	label variable brngy_start_edu_c "Which barangay was \${edu_repeat_name_c} living when they started school?"
	note brngy_start_edu_c: "Which barangay was \${edu_repeat_name_c} living when they started school?"

	label variable brngy_start_edu_oth_c "Please specify the barangay:"
	note brngy_start_edu_oth_c: "Please specify the barangay:"

	label variable preschool_first_c "Was the first educational institution \${edu_repeat_name_c} attended preschool?"
	note preschool_first_c: "Was the first educational institution \${edu_repeat_name_c} attended preschool?"
	label define preschool_first_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values preschool_first_c preschool_first_c

	label variable currently_edu_c "Is \${edu_repeat_name_c} currently enrolled at any educational institution? (inc"
	note currently_edu_c: "Is \${edu_repeat_name_c} currently enrolled at any educational institution? (including part-time or online education)"
	label define currently_edu_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values currently_edu_c currently_edu_c

	label variable edu_attended_c "What is the highest level of education \${edu_repeat_name_c} has attended? Atten"
	note edu_attended_c: "What is the highest level of education \${edu_repeat_name_c} has attended? Attended refers to any enrollment, even if that level was not completed. If currently enrolled, select present level."
	label define edu_attended_c 0 "None" 1 "Pre-Kinder/Daycare" 2 "Kinder" 3 "1st Grade" 4 "2nd Grade" 5 "3rd Grade" 6 "4th Grade" 7 "5th Grade" 8 "6th Grade/elementary graduate" 9 "7th Grade / 1st year junior high school" 10 "8th Grade / 2nd year junior high school" 11 "9th Grade / 3rd year junior high school" 12 "10th Grade / HS Graduate / 4th year junior high school" 13 "11th Grade" 14 "12th Grade / SHS Graduate" 15 "1st Year Vocational training or associates degree" 16 "2nd Year Vocational training or associates degree" 17 "Vocational training or associates degree graduate" 18 "1st year of college" 19 "2nd year of college" 20 "3rd year of college" 21 "4th year of college or higher" 22 "College graduate" 23 "Education beyond college" -666 "Other" -999 "Don't know" -888 "Refuse to answer"
	label values edu_attended_c edu_attended_c

	label variable edu_attended_oth_c "Please specify"
	note edu_attended_oth_c: "Please specify"

	label variable edu_completed_c "What is the highest level of education \${edu_repeat_name_c} has completed? Comp"
	note edu_completed_c: "What is the highest level of education \${edu_repeat_name_c} has completed? Completed refers to the level of education attended and officially completed"
	label define edu_completed_c 0 "None" 1 "Pre-Kinder/Daycare" 2 "Kinder" 3 "1st Grade" 4 "2nd Grade" 5 "3rd Grade" 6 "4th Grade" 7 "5th Grade" 8 "6th Grade/elementary graduate" 9 "7th Grade / 1st year junior high school" 10 "8th Grade / 2nd year junior high school" 11 "9th Grade / 3rd year junior high school" 12 "10th Grade / HS Graduate / 4th year junior high school" 13 "11th Grade" 14 "12th Grade / SHS Graduate" 15 "1st Year Vocational training or associates degree" 16 "2nd Year Vocational training or associates degree" 17 "Vocational training or associates degree graduate" 18 "1st year of college" 19 "2nd year of college" 20 "3rd year of college" 21 "4th year of college or higher" 22 "College graduate" 23 "Education beyond college" -666 "Other" -999 "Don't know" -888 "Refuse to answer"
	label values edu_completed_c edu_completed_c

	label variable edu_completed_oth_c "Please specify"
	note edu_completed_oth_c: "Please specify"

	label variable edu_completed_age_c "How old was \${edu_repeat_name_c} when they completed their highest level of edu"
	note edu_completed_age_c: "How old was \${edu_repeat_name_c} when they completed their highest level of education?"

	label variable prim_numyrs_c "For how many years was \${edu_repeat_name_c} in elementary school?"
	note prim_numyrs_c: "For how many years was \${edu_repeat_name_c} in elementary school?"

	label variable prim_count_c "How many elementary schools did \${edu_repeat_name_c} ever attend?"
	note prim_count_c: "How many elementary schools did \${edu_repeat_name_c} ever attend?"

	label variable prim_interrupt_yn_c "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their "
	note prim_interrupt_yn_c: "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their elementary education?"
	label define prim_interrupt_yn_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values prim_interrupt_yn_c prim_interrupt_yn_c

	label variable jhs_numyrs_c "For how many years was \${edu_repeat_name_c} in junior high school?"
	note jhs_numyrs_c: "For how many years was \${edu_repeat_name_c} in junior high school?"

	label variable jhs_count_c "How many junior high schools did \${edu_repeat_name_c} ever attend?"
	note jhs_count_c: "How many junior high schools did \${edu_repeat_name_c} ever attend?"

	label variable jhs_interrupt_yn_c "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their "
	note jhs_interrupt_yn_c: "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their junior high school education?"
	label define jhs_interrupt_yn_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values jhs_interrupt_yn_c jhs_interrupt_yn_c

	label variable shs_numyrs_c "For how many years was \${edu_repeat_name_c} in senior high school?"
	note shs_numyrs_c: "For how many years was \${edu_repeat_name_c} in senior high school?"

	label variable shs_count_c "How many senior high schools did \${edu_repeat_name_c} ever attend?"
	note shs_count_c: "How many senior high schools did \${edu_repeat_name_c} ever attend?"

	label variable shs_interrupt_yn_c "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their "
	note shs_interrupt_yn_c: "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their senior high school education?"
	label define shs_interrupt_yn_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values shs_interrupt_yn_c shs_interrupt_yn_c

	label variable hs_numyrs_c "For how many years was \${edu_repeat_name_c} in high school?"
	note hs_numyrs_c: "For how many years was \${edu_repeat_name_c} in high school?"

	label variable hs_count_c "How many high schools did \${edu_repeat_name_c} ever attend?"
	note hs_count_c: "How many high schools did \${edu_repeat_name_c} ever attend?"

	label variable hs_interrupt_yn_c "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their "
	note hs_interrupt_yn_c: "Did \${edu_repeat_name_c} ever stop studying for more than 4 weeks during their high school education?"
	label define hs_interrupt_yn_c 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values hs_interrupt_yn_c hs_interrupt_yn_c

	label variable instr_types "What types of instruction did \${edu_repeat_name_c} regularly use while their sc"
	note instr_types: "What types of instruction did \${edu_repeat_name_c} regularly use while their school was closed during COVID-19? (Select all that apply)"

	label variable instr_types_oth "Please specify the other instruction type:"
	note instr_types_oth: "Please specify the other instruction type:"






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
*   Corrections file path and filename:  C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-child_edu-edu_repeat_c_corrections.csv
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
