* import_ADB_questionnaire_test-hs_group_g1-hs_g1.do
*
* 	Imports and aggregates "ADB_questionnaire-hs_group_g1-hs_g1" (ID: ADB_questionnaire_test) data.
*
*	Inputs:  "C:/Users/AJolex/Downloads/ADB_questionnaire-hs_group_g1-hs_g1.csv"
*	Outputs: "C:/Users/AJolex/Downloads/ADB_questionnaire-hs_group_g1-hs_g1.dta"
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
local csvfile "C:/Users/AJolex/Downloads/ADB_questionnaire-hs_group_g1-hs_g1.csv"
local dtafile "C:/Users/AJolex/Downloads/ADB_questionnaire-hs_group_g1-hs_g1.dta"
local corrfile "C:/Users/AJolex/Downloads/ADB_questionnaire-hs_group_g1-hs_g1_corrections.csv"
local note_fields1 ""
local text_fields1 "hs_repeat_index_g1 index_namhs_g1 index_namhs_hili_g1 hs_prov_g1 hs_prov_oth_g1 hs_munic_g1 hs_munic_oth_g1 hs_school_name_g1 hs_school_name_oth_g1 hs_id_g1 hs_lbl_pull_g1 hs_lbl_name_g1 hs_start_g1"
local text_fields2 "hs_start_year_g1 hs_stop_g1 hs_live_prov_g1 hs_live_prov_oth_g1 hs_live_munic_g1 hs_live_munic_oth_g1 hs_live_brgny_g1 hs_live_brngy_oth_g1 live_else_hs_g1_count livehs_count_g1"
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


	label variable hs_reg6_g1 "Is \${index_namhs_g1} high school located in Region 6 or Negros Island Region?"
	note hs_reg6_g1: "Is \${index_namhs_g1} high school located in Region 6 or Negros Island Region?"
	label define hs_reg6_g1 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values hs_reg6_g1 hs_reg6_g1

	label variable hs_prov_g1 "In which Province is \${index_namhs_g1} high school located?"
	note hs_prov_g1: "In which Province is \${index_namhs_g1} high school located?"

	label variable hs_prov_oth_g1 "Please specify"
	note hs_prov_oth_g1: "Please specify"

	label variable hs_munic_g1 "In which Municipality is \${index_namhs_g1} high school located?"
	note hs_munic_g1: "In which Municipality is \${index_namhs_g1} high school located?"

	label variable hs_munic_oth_g1 "Please specify the municipality of the school:"
	note hs_munic_oth_g1: "Please specify the municipality of the school:"

	label variable hs_school_name_g1 "What is the name of \${index_namhs_g1} high school \${edu_repeat_name_g1} attend"
	note hs_school_name_g1: "What is the name of \${index_namhs_g1} high school \${edu_repeat_name_g1} attended?"

	label variable hs_school_name_oth_g1 "Please specify the school name:"
	note hs_school_name_oth_g1: "Please specify the school name:"

	label variable hs_start_g1 "When did \${edu_repeat_name_g1} start going to \${index_namhs_g1} high school? ("
	note hs_start_g1: "When did \${edu_repeat_name_g1} start going to \${index_namhs_g1} high school? (MM-YYYY)"

	label variable hs_stop_g1 "When did \${edu_repeat_name_g1} stop going to \${index_namhs_g1} high school? (M"
	note hs_stop_g1: "When did \${edu_repeat_name_g1} stop going to \${index_namhs_g1} high school? (MM-YYYY)"

	label variable hs_same_barangay_g1 "During \${edu_repeat_name_g1}'s studies at \${hs_lbl_name_g1}, did they live in "
	note hs_same_barangay_g1: "During \${edu_repeat_name_g1}'s studies at \${hs_lbl_name_g1}, did they live in the same barangay as the school?"
	label define hs_same_barangay_g1 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values hs_same_barangay_g1 hs_same_barangay_g1

	label variable hs_live_prov_g1 "Which province did \${edu_repeat_name_g1} live in during their studies at \${hs_"
	note hs_live_prov_g1: "Which province did \${edu_repeat_name_g1} live in during their studies at \${hs_lbl_name_g1}?"

	label variable hs_live_prov_oth_g1 "Please specify the province:"
	note hs_live_prov_oth_g1: "Please specify the province:"

	label variable hs_live_munic_g1 "Which municipality did \${edu_repeat_name_g1} live in during their studies at \$"
	note hs_live_munic_g1: "Which municipality did \${edu_repeat_name_g1} live in during their studies at \${hs_lbl_name_g1}?"

	label variable hs_live_munic_oth_g1 "Please specify the municipality:"
	note hs_live_munic_oth_g1: "Please specify the municipality:"

	label variable hs_live_brgny_g1 "Which barangay did \${edu_repeat_name_g1} live in during their studies at \${hs_"
	note hs_live_brgny_g1: "Which barangay did \${edu_repeat_name_g1} live in during their studies at \${hs_lbl_name_g1}?"

	label variable hs_live_brngy_oth_g1 "Please specify the barangay:"
	note hs_live_brngy_oth_g1: "Please specify the barangay:"

	label variable hs_live_else_yn_g1 "Did \${edu_repeat_name_g1} live anywhere else during their studies at \${hs_lbl_"
	note hs_live_else_yn_g1: "Did \${edu_repeat_name_g1} live anywhere else during their studies at \${hs_lbl_name_g1}?"
	label define hs_live_else_yn_g1 1 "Yes" 0 "No" -999 "Don't Know" -888 "Refuse to answer"
	label values hs_live_else_yn_g1 hs_live_else_yn_g1






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
*   Corrections file path and filename:  C:/Users/AJolex/Downloads/ADB_questionnaire-hs_group_g1-hs_g1_corrections.csv
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
