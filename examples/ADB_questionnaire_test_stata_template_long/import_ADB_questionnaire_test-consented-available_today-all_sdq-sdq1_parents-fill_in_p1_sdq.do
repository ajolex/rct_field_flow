* import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq.do
*
* 	Imports and aggregates "ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq" (ID: ADB_questionnaire_test) data.
*
*	Inputs:  "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq.csv"
*	Outputs: "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq.dta"
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
local csvfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq.csv"
local dtafile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq.dta"
local corrfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq_corrections.csv"
local note_fields1 ""
local text_fields1 "temp_sdq_p1"
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


	label variable select_sdqp1_child "Please select the child for the relevant questionnaire."
	note select_sdqp1_child: "Please select the child for the relevant questionnaire."
	label define select_sdqp1_child 1 "\${calc_hh_mem_name1}" 2 "\${calc_hh_mem_name2}" 3 "\${calc_hh_mem_name3}" 4 "\${calc_hh_mem_name4}" 5 "\${calc_hh_mem_name5}" 6 "\${calc_hh_mem_name6}" 7 "\${calc_hh_mem_name7}" 8 "\${calc_hh_mem_name8}" 9 "\${calc_hh_mem_name9}" 10 "\${calc_hh_mem_name10}" 11 "\${calc_hh_mem_name11}" 12 "\${calc_hh_mem_name12}" 13 "\${calc_hh_mem_name13}" 14 "\${calc_hh_mem_name14}" 15 "\${calc_hh_mem_name15}" 16 "\${calc_hh_mem_name16}" 17 "\${calc_hh_mem_name17}" 18 "\${calc_hh_mem_name18}" 19 "\${calc_hh_mem_name19}" 20 "\${calc_hh_mem_name20}" 21 "\${calc_hh_mem_name21}"
	label values select_sdqp1_child select_sdqp1_child

	label variable sdq_p1_1 "Question 1"
	note sdq_p1_1: "Question 1"
	label define sdq_p1_1 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_1 sdq_p1_1

	label variable sdq_p1_2 "Question 2"
	note sdq_p1_2: "Question 2"
	label define sdq_p1_2 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_2 sdq_p1_2

	label variable sdq_p1_3 "Question 3"
	note sdq_p1_3: "Question 3"
	label define sdq_p1_3 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_3 sdq_p1_3

	label variable sdq_p1_4 "Question 4"
	note sdq_p1_4: "Question 4"
	label define sdq_p1_4 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_4 sdq_p1_4

	label variable sdq_p1_5 "Question 5"
	note sdq_p1_5: "Question 5"
	label define sdq_p1_5 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_5 sdq_p1_5

	label variable sdq_p1_6 "Question 6"
	note sdq_p1_6: "Question 6"
	label define sdq_p1_6 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_6 sdq_p1_6

	label variable sdq_p1_7 "Question 7"
	note sdq_p1_7: "Question 7"
	label define sdq_p1_7 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_7 sdq_p1_7

	label variable sdq_p1_8 "Question 8"
	note sdq_p1_8: "Question 8"
	label define sdq_p1_8 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_8 sdq_p1_8

	label variable sdq_p1_9 "Question 9"
	note sdq_p1_9: "Question 9"
	label define sdq_p1_9 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_9 sdq_p1_9

	label variable sdq_p1_10 "Question 10"
	note sdq_p1_10: "Question 10"
	label define sdq_p1_10 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_10 sdq_p1_10

	label variable sdq_p1_11 "Question 11"
	note sdq_p1_11: "Question 11"
	label define sdq_p1_11 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_11 sdq_p1_11

	label variable sdq_p1_12 "Question 12"
	note sdq_p1_12: "Question 12"
	label define sdq_p1_12 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_12 sdq_p1_12

	label variable sdq_p1_13 "Question 13"
	note sdq_p1_13: "Question 13"
	label define sdq_p1_13 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_13 sdq_p1_13

	label variable sdq_p1_14 "Question 14"
	note sdq_p1_14: "Question 14"
	label define sdq_p1_14 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_14 sdq_p1_14

	label variable sdq_p1_15 "Question 15"
	note sdq_p1_15: "Question 15"
	label define sdq_p1_15 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_15 sdq_p1_15

	label variable sdq_p1_16 "Question 16"
	note sdq_p1_16: "Question 16"
	label define sdq_p1_16 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_16 sdq_p1_16

	label variable sdq_p1_17 "Question 17"
	note sdq_p1_17: "Question 17"
	label define sdq_p1_17 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_17 sdq_p1_17

	label variable sdq_p1_18 "Question 18"
	note sdq_p1_18: "Question 18"
	label define sdq_p1_18 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_18 sdq_p1_18

	label variable sdq_p1_19 "Question 19"
	note sdq_p1_19: "Question 19"
	label define sdq_p1_19 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_19 sdq_p1_19

	label variable sdq_p1_20 "Question 20"
	note sdq_p1_20: "Question 20"
	label define sdq_p1_20 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_20 sdq_p1_20

	label variable sdq_p1_21 "Question 21"
	note sdq_p1_21: "Question 21"
	label define sdq_p1_21 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_21 sdq_p1_21

	label variable sdq_p1_22 "Question 22"
	note sdq_p1_22: "Question 22"
	label define sdq_p1_22 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_22 sdq_p1_22

	label variable sdq_p1_23 "Question 23"
	note sdq_p1_23: "Question 23"
	label define sdq_p1_23 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_23 sdq_p1_23

	label variable sdq_p1_24 "Question 24"
	note sdq_p1_24: "Question 24"
	label define sdq_p1_24 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_24 sdq_p1_24

	label variable sdq_p1_25 "Question 25"
	note sdq_p1_25: "Question 25"
	label define sdq_p1_25 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_p1_25 sdq_p1_25

	label variable sdq_p1_another_child "Is there another questionnaire to encode?"
	note sdq_p1_another_child: "Is there another questionnaire to encode?"
	label define sdq_p1_another_child 1 "Yes" 0 "No"
	label values sdq_p1_another_child sdq_p1_another_child






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
*   Corrections file path and filename:  C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq_corrections.csv
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
