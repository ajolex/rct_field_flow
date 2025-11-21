* import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq_child-sdq_child_report.do
*
* 	Imports and aggregates "ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report" (ID: ADB_questionnaire_test) data.
*
*	Inputs:  "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report.csv"
*	Outputs: "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report.dta"
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
local csvfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report.csv"
local dtafile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report.dta"
local corrfile "C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report_corrections.csv"
local note_fields1 ""
local text_fields1 "illit_child_index child_sdq_repeat_name"
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


	label variable sdq_c_consent "Introduce yourself to the child. Then ask the child to introduce themself before"
	note sdq_c_consent: "Introduce yourself to the child. Then ask the child to introduce themself before reading the consent form below. Make sure to address the child by their name. Be open and friendly. I am from Innovations for Poverty Action (IPA), a research organization. We are collaborating with Asian Development Bank (ADB) to conduct a survey on education, migration, labor, and children's health. We would like to invite you to participate. Participation in this survey is completely voluntary. If you agree to participate, you will answer a few questions. We do not anticipate any risks to you from participating in the survey. If at any point you feel uncomfortable, you are free to stop answering the questions, with no penalty. Please be assured that all your answers will be kept confidential. No names or personal information will be released, and only the research team will have access to any identifying data. Feel free to ask me any questions. Please answer any questions the child may have. ...If I have answered all of your questions, do you give your consent to be part of this survey?"
	label define sdq_c_consent 1 "Yes" 0 "No"
	label values sdq_c_consent sdq_c_consent

	label variable sdq_child1 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child1: "Consider how things have been for you over the last six months. How well does the statement describe you? I try to be nice to other people. I care about their feelings"
	label define sdq_child1 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child1 sdq_child1

	label variable sdq_child2 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child2: "Consider how things have been for you over the last six months. How well does the statement describe you? I am restless, I cannot stay still for long"
	label define sdq_child2 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child2 sdq_child2

	label variable sdq_child3 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child3: "Consider how things have been for you over the last six months. How well does the statement describe you? I get a lot of headaches, stomach-aches or sickness"
	label define sdq_child3 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child3 sdq_child3

	label variable sdq_child4 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child4: "Consider how things have been for you over the last six months. How well does the statement describe you? I try to be nice to other people. I care about their feelings"
	label define sdq_child4 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child4 sdq_child4

	label variable sdq_child5 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child5: "Consider how things have been for you over the last six months. How well does the statement describe you? I get very angry and often lose my temper"
	label define sdq_child5 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child5 sdq_child5

	label variable sdq_child6 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child6: "Consider how things have been for you over the last six months. How well does the statement describe you? I would rather be alone than with people of my age"
	label define sdq_child6 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child6 sdq_child6

	label variable sdq_child7 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child7: "Consider how things have been for you over the last six months. How well does the statement describe you? I usually do as I am told"
	label define sdq_child7 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child7 sdq_child7

	label variable sdq_child8 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child8: "Consider how things have been for you over the last six months. How well does the statement describe you? I worry a lot"
	label define sdq_child8 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child8 sdq_child8

	label variable sdq_child9 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child9: "Consider how things have been for you over the last six months. How well does the statement describe you? I am helpful if someone is hurt, upset or feeling ill"
	label define sdq_child9 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child9 sdq_child9

	label variable sdq_child10 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child10: "Consider how things have been for you over the last six months. How well does the statement describe you? I am constantly fidgeting or squirming"
	label define sdq_child10 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child10 sdq_child10

	label variable sdq_child11 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child11: "Consider how things have been for you over the last six months. How well does the statement describe you? I have one good friend or more"
	label define sdq_child11 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child11 sdq_child11

	label variable sdq_child12 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child12: "Consider how things have been for you over the last six months. How well does the statement describe you? I fight a lot. I can make other people do what I want"
	label define sdq_child12 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child12 sdq_child12

	label variable sdq_child13 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child13: "Consider how things have been for you over the last six months. How well does the statement describe you? I am often unhappy, depressed or tearful"
	label define sdq_child13 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child13 sdq_child13

	label variable sdq_child14 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child14: "Consider how things have been for you over the last six months. How well does the statement describe you? Other people my age generally like me"
	label define sdq_child14 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child14 sdq_child14

	label variable sdq_child15 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child15: "Consider how things have been for you over the last six months. How well does the statement describe you? I am easily distracted, I find it difficult to concentrate"
	label define sdq_child15 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child15 sdq_child15

	label variable sdq_child16 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child16: "Consider how things have been for you over the last six months. How well does the statement describe you? I am nervous in new situations. I easily lose confidence"
	label define sdq_child16 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child16 sdq_child16

	label variable sdq_child17 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child17: "Consider how things have been for you over the last six months. How well does the statement describe you? I am kind to younger children"
	label define sdq_child17 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child17 sdq_child17

	label variable sdq_child18 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child18: "Consider how things have been for you over the last six months. How well does the statement describe you? I am often accused of lying or cheating"
	label define sdq_child18 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child18 sdq_child18

	label variable sdq_child19 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child19: "Consider how things have been for you over the last six months. How well does the statement describe you? Other children or young people pick on me or bully me"
	label define sdq_child19 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child19 sdq_child19

	label variable sdq_child20 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child20: "Consider how things have been for you over the last six months. How well does the statement describe you? I often offer to help others (parents, teachers, children)"
	label define sdq_child20 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child20 sdq_child20

	label variable sdq_child21 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child21: "Consider how things have been for you over the last six months. How well does the statement describe you? I think before I do things"
	label define sdq_child21 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child21 sdq_child21

	label variable sdq_child22 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child22: "Consider how things have been for you over the last six months. How well does the statement describe you? I take things that are not mine from home, school or elsewhere"
	label define sdq_child22 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child22 sdq_child22

	label variable sdq_child23 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child23: "Consider how things have been for you over the last six months. How well does the statement describe you? I get along better with adults than with people my own age"
	label define sdq_child23 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child23 sdq_child23

	label variable sdq_child24 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child24: "Consider how things have been for you over the last six months. How well does the statement describe you? I have many fears, I am easily scared"
	label define sdq_child24 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child24 sdq_child24

	label variable sdq_child25 "Consider how things have been for you over the last six months. How well does th"
	note sdq_child25: "Consider how things have been for you over the last six months. How well does the statement describe you? I finish the work I'm doing. My attention is good"
	label define sdq_child25 0 "Not true" 1 "Somewhat true" 2 "Certainly true" -999 "Don't know" -888 "Refuse to answer"
	label values sdq_child25 sdq_child25






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
*   Corrections file path and filename:  C:/Users/AJolex/Downloads/ADB_questionnaire-consented-available_today-all_sdq-sdq_child-sdq_child_report_corrections.csv
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
