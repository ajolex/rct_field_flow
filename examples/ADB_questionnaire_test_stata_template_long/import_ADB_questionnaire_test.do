* import_ADB_questionnaire_test.do
*
* 	Imports and aggregates "ADB_questionnaire" (ID: ADB_questionnaire_test) data.
*
*	Inputs:  "C:/Users/AJolex/Downloads/ADB_questionnaire.csv"
*	Outputs: "C:/Users/AJolex/Downloads/ADB_questionnaire.dta"
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
local csvfile "C:/Users/AJolex/Downloads/ADB_questionnaire.csv"
local dtafile "C:/Users/AJolex/Downloads/ADB_questionnaire.dta"
local corrfile "C:/Users/AJolex/Downloads/ADB_questionnaire_corrections.csv"
local note_fields1 ""
local text_fields1 "deviceid devicephonenum hhid device_info duration comments calc_current_year fo enum_id enum_name pull_province pull_municipal_city pull_barangay pull_adm4_pcode consent_no start_roster firstn_resp"
local text_fields2 "lastn_resp birthdate_resp full_name_resp roster_repeat_count temp_mem_count join_other_mem join_other_age calc_hh_mem_name1 calc_hh_mem_name2 calc_hh_mem_name3 calc_hh_mem_name4 calc_hh_mem_name5"
local text_fields3 "calc_hh_mem_name6 calc_hh_mem_name7 calc_hh_mem_name8 calc_hh_mem_name9 calc_hh_mem_name10 calc_hh_mem_name11 calc_hh_mem_name12 calc_hh_mem_name13 calc_hh_mem_name14 calc_hh_mem_name15 line_num1"
local text_fields4 "line_num2 line_num3 line_num4 line_num5 line_num6 line_num7 line_num8 line_num9 line_num10 line_num11 line_num12 line_num13 line_num14 line_num15 join_all_ages join_all_names join_other_memid"
local text_fields5 "count_children_4to10 count_parents_of4to10 join_children_4to10 join_adults_18to40 join_adults_parents1 join_parents_of4to10 join_adults_parents count_parents_of5to18 join_adults_parents5to18"
local text_fields6 "join_parentsof5to18 count_children_5to15 join_children_5to15 count_children_5to18 join_children_5to18 join_child_age_5to18 join_adults_parents11to17 join_parents_of11to17 count_children_11to18"
local text_fields7 "join_children_11to18 count_children_5to8 join_children_5to8 join_children_5to10 count_children_5to10 count_children_9to12 join_children_9to12 count_children_13to18 join_children_13to18"
local text_fields8 "join_resnopar_5to40 join_age_5to40 join_allparents join_ages_parents join_all_edulab count_resnopar_5to40 count_allparents count_all_edulab join_ages_edulab join_health join_ages_health count_health"
local text_fields9 "join_kid_names join_kid_ages count_kids is_resp_kid resp_kid_name resp_kid_age all_kid_names all_kid_ages count_all_kids join_parents_names join_parents_ages count_parents is_resp_parent"
local text_fields10 "resp_parent_name resp_parent_age all_parent_names all_parent_ages count_all_parents joinhealthnames joinhealthages countinghealth join_residents_19to40 join_resident_ages_19to40 count_residents_19to40"
local text_fields11 "is_resp_resident_19to40 resp_resident_name_19to40 resp_resident_age_19to40 all_resident_names_19to40 all_resident_ages_19to40 count_all_residents_19to40 any_child_5to18 join_parentfolk"
local text_fields12 "join_parentfolk_ages count_parentfolk is_resp_parentfolk resp_parentfolk_name resp_parentfolk_age all_parentfolk_names all_parentfolk_ages count_all_parentfolk join_edu_names join_edu_ages count_edu"
local text_fields13 "count_adults_41over join_lab_names count_lab join_kids_9to18 count_kids_9to18 choice_filter_expr nr_roster_repeat_count calc_hh_mem_name16 calc_hh_mem_name17 calc_hh_mem_name18 calc_hh_mem_name19"
local text_fields14 "calc_hh_mem_name20 calc_hh_mem_name21 nr_temp_mem_count join_nr_other_mem join_nr_other_age all_possible_members count_all_members child_parent_link_count allselected_parents parent_dedup_count"
local text_fields15 "uniqua_parents uniqua_parents_dedup uniqua_parents_lbl uniqua_parents_lbl_dedup parent_randnums parent_randnums_dedup res_parents_lbl res_parents_lbl_dedup nonparent_list_count nonparent_names"
local text_fields16 "nonparent_randnums is_nonparent_resp nonparent_resp_name all_nonparent_19to40 final_edu_names count_edu_test all_rando_names_space all_rando_names all_rando_numbers count_allrandnums"
local text_fields17 "elig_member_repeat_count min_rand max_rand min_name max_name join_2randmems_name count_join_rand end_roster dur_roster rands_avail_when rands_avail_thurs start_edu_g1 edu_repeat_g1_count"
local text_fields18 "temp_prim_count_g1 temp_ju_count_g1 temp_se_count_g1 temp_col_count_g1 temp_post_count_g1 end_education_g1 dur_education_g1 start_education_c edu_repeat_c_count end_education_c dur_education_c"
local text_fields19 "start_education_g2 edu_repeat_g2_count end_education_g2 dur_education_g2 start_health child_health_repeat_count end_health dur_health start_labor_g1 labor_repeat_g1_count end_labor_g1 dur_labor_g1"
local text_fields20 "start_labor_g2 labor_repeat_g2_count end_labor_g2 dur_labor_g2 start_sdq1_parents sdq_parent_report_count fill_in_p1_sdq_count sdq_p1_count end_sdq_parents dur_sdq_parents start_sdq_parents2"
local text_fields21 "sdq2_parent_report_count fill_in_p2_sdq_count sdq_p2_count end_sdq2_parents dur_sdq2_parents start_sdq_child child_lit_repeat_count literate_indexes illiterate_indexes count_lit_children"
local text_fields22 "count_illit_children sdq_child_report_count sdqc_paper_start sdqc_paper_duration dur_sdq_child fill_in_c_sdq_count sdq_c_count end_sdq_child dur_sdq_child_mod instanceid"
local date_fields1 ""
local datetime_fields1 "submissiondate starttime endtime"

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


	* consolidate unique ID into "key" variable
	replace key=instanceid if key==""
	drop instanceid


	* label variables
	label variable key "Unique submission ID"
	cap label variable submissiondate "Date/time submitted"
	cap label variable formdef_version "Form version used on device"
	cap label variable review_status "Review status"
	cap label variable review_comments "Comments made during review"
	cap label variable review_corrections "Corrections made during review"


	label variable temp_color_understand "All the text that is blue will contain instructions that should NOT be read to t"
	note temp_color_understand: "All the text that is blue will contain instructions that should NOT be read to the respondent. Is this clearly understood?"
	label define temp_color_understand 1 "Yes" 0 "No"
	label values temp_color_understand temp_color_understand

	label variable fo "Name / ID"
	note fo: "Name / ID"

	label variable consent_agree "If I have answered all of your questions, do you give your consent for you and y"
	note consent_agree: "If I have answered all of your questions, do you give your consent for you and your household to be part of this survey?"
	label define consent_agree 1 "Yes" 0 "No"
	label values consent_agree consent_agree

	label variable consent_no "Be polite when asking the reason. May we ask the reason?"
	note consent_no: "Be polite when asking the reason. May we ask the reason?"

	label variable randsel_yesno "Is this one of the randomly selected households where resident members also answ"
	note randsel_yesno: "Is this one of the randomly selected households where resident members also answer for themselves (Protocol B)? For most households today (Day 3), select 'no'."
	label define randsel_yesno 1 "Yes" 0 "No"
	label values randsel_yesno randsel_yesno

	label variable members_ck "I would like to ask you the names of ALL resident household members. Are there c"
	note members_ck: "I would like to ask you the names of ALL resident household members. Are there children aged 5-18 and their parents, and other adults between 19-40?"
	label define members_ck 1 "Yes" 0 "No"
	label values members_ck members_ck

	label variable firstn_resp "What is your first name?"
	note firstn_resp: "What is your first name?"

	label variable lastn_resp "What is your last name?"
	note lastn_resp: "What is your last name?"

	label variable sex_resp "What is your sex assigned at birth?"
	note sex_resp: "What is your sex assigned at birth?"
	label define sex_resp 1 "Female" 2 "Male"
	label values sex_resp sex_resp

	label variable relation_resp "What is your relationship to the head of this household? The head of the househo"
	note relation_resp: "What is your relationship to the head of this household? The head of the household is the primary decision maker for the household."
	label define relation_resp 1 "Head" 2 "Spouse/partner" 3 "Son/daughter" 4 "Son-in-law/daughter-in-law" 5 "Stepson/stepdaughter" 6 "Grandchild" 7 "Brother/sister" 8 "Brother-in-law/sister-in-law" 9 "Father/mother" 10 "Father-in-law/mother-in-law" 11 "Grandparent" 12 "Great-grandchild" 13 "Other family" 14 "Household help" 15 "Lodger" 16 "Friend" 17 "Nephew/Niece" 18 "Stepfather/Stepmother"
	label values relation_resp relation_resp

	label variable age_resp "How old are you?"
	note age_resp: "How old are you?"

	label variable birthdate_resp "What is your birthdate? MM-DD-YYYY"
	note birthdate_resp: "What is your birthdate? MM-DD-YYYY"

	label variable marital_resp "What is your marital status?"
	note marital_resp: "What is your marital status?"
	label define marital_resp 1 "Married or in a committed union and living with partner" 2 "Married or in a committed union, not living with partner" 3 "Divorced" 4 "Separated" 5 "Widow or Widower" 6 "Single, never married" -666 "Other" -999 "Don't know" -888 "Refuse to answer"
	label values marital_resp marital_resp

	label variable hh_head "Can you please confirm the head of the household"
	note hh_head: "Can you please confirm the head of the household"
	label define hh_head 1 "\${calc_hh_mem_name1}" 2 "\${calc_hh_mem_name2}" 3 "\${calc_hh_mem_name3}" 4 "\${calc_hh_mem_name4}" 5 "\${calc_hh_mem_name5}" 6 "\${calc_hh_mem_name6}" 7 "\${calc_hh_mem_name7}" 8 "\${calc_hh_mem_name8}" 9 "\${calc_hh_mem_name9}" 10 "\${calc_hh_mem_name10}" 11 "\${calc_hh_mem_name11}" 12 "\${calc_hh_mem_name12}" 13 "\${calc_hh_mem_name13}" 14 "\${calc_hh_mem_name14}" 15 "\${calc_hh_mem_name15}" 16 "\${calc_hh_mem_name16}" 17 "\${calc_hh_mem_name17}" 18 "\${calc_hh_mem_name18}" 19 "\${calc_hh_mem_name19}" 20 "\${calc_hh_mem_name20}" 21 "\${calc_hh_mem_name21}"
	label values hh_head hh_head

	label variable nr_yn "Now I would like to ask you about non-resident members of this household, includ"
	note nr_yn: "Now I would like to ask you about non-resident members of this household, including: - Anyone who would normally live here but is away for at least 3 months or spends the majority of their time in another place for work - Anyone who would normally live here but is away for at least 3 months or spends the majority of their time in another place for school at a post-secondary, college or university level - Anyone who has moved elsewhere for work or school and generally stays there but visits during holidays or weekends Are there any non-resident members?"
	label define nr_yn 1 "Yes" 0 "No"
	label values nr_yn nr_yn

	label variable rands_avail "Are all selected member(s) available to be interviewed today?"
	note rands_avail: "Are all selected member(s) available to be interviewed today?"
	label define rands_avail 1 "Yes, both of them" 0 "None" 2 "Only one of them"
	label values rands_avail rands_avail

	label variable avail_tmrw "Are these members available anytime tomorrow?"
	note avail_tmrw: "Are these members available anytime tomorrow?"
	label define avail_tmrw 1 "Yes, both of them" 0 "None" 2 "Only one of them"
	label values avail_tmrw avail_tmrw

	label variable rands_avail_when "What time tomorrow are they available?"
	note rands_avail_when: "What time tomorrow are they available?"

	label variable avail_thurs "Are these members available anytime this Thursday?"
	note avail_thurs: "Are these members available anytime this Thursday?"
	label define avail_thurs 1 "Yes, both of them" 0 "None" 2 "Only one of them"
	label values avail_thurs avail_thurs

	label variable rands_avail_thurs "What time on Thursday are they available?"
	note rands_avail_thurs: "What time on Thursday are they available?"

	label variable sdq1_literacy "Are you comfortable with reading and answering these questions by yourself using"
	note sdq1_literacy: "Are you comfortable with reading and answering these questions by yourself using a pen and paper? If this is not possible, we also have an option for me to read them out loud and you answer verbally."
	label define sdq1_literacy 1 "Yes" 0 "No"
	label values sdq1_literacy sdq1_literacy

	label variable paper_question_yesnop1 "Are there paper questionnaires to encode?"
	note paper_question_yesnop1: "Are there paper questionnaires to encode?"
	label define paper_question_yesnop1 1 "Yes" 0 "No"
	label values paper_question_yesnop1 paper_question_yesnop1

	label variable sdq2_respondent "Please select the parent/guardian respondent for this module."
	note sdq2_respondent: "Please select the parent/guardian respondent for this module."
	label define sdq2_respondent 1 "\${calc_hh_mem_name1}" 2 "\${calc_hh_mem_name2}" 3 "\${calc_hh_mem_name3}" 4 "\${calc_hh_mem_name4}" 5 "\${calc_hh_mem_name5}" 6 "\${calc_hh_mem_name6}" 7 "\${calc_hh_mem_name7}" 8 "\${calc_hh_mem_name8}" 9 "\${calc_hh_mem_name9}" 10 "\${calc_hh_mem_name10}" 11 "\${calc_hh_mem_name11}" 12 "\${calc_hh_mem_name12}" 13 "\${calc_hh_mem_name13}" 14 "\${calc_hh_mem_name14}" 15 "\${calc_hh_mem_name15}" 16 "\${calc_hh_mem_name16}" 17 "\${calc_hh_mem_name17}" 18 "\${calc_hh_mem_name18}" 19 "\${calc_hh_mem_name19}" 20 "\${calc_hh_mem_name20}" 21 "\${calc_hh_mem_name21}"
	label values sdq2_respondent sdq2_respondent

	label variable sdq2_literacy "Are you comfortable with reading and answering these questions by yourself using"
	note sdq2_literacy: "Are you comfortable with reading and answering these questions by yourself using a pen and paper? If this is not possible, we also have an option for me to read them out loud and you answer verbally."
	label define sdq2_literacy 1 "Yes" 0 "No"
	label values sdq2_literacy sdq2_literacy

	label variable paper_question_yesnop2 "Are there paper questionnaires to encode?"
	note paper_question_yesnop2: "Are there paper questionnaires to encode?"
	label define paper_question_yesnop2 1 "Yes" 0 "No"
	label values paper_question_yesnop2 paper_question_yesnop2

	label variable paper_question_yesno "Are there paper questionnaires to encode?"
	note paper_question_yesno: "Are there paper questionnaires to encode?"
	label define paper_question_yesno 1 "Yes" 0 "No"
	label values paper_question_yesno paper_question_yesno






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
*   Corrections file path and filename:  C:/Users/AJolex/Downloads/ADB_questionnaire_corrections.csv
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


* launch .do files to process repeat groups

do "import_ADB_questionnaire_test-college_group_g1-college_g1.do"
do "import_ADB_questionnaire_test-college_group_g1-interruption_college_repeat_g1.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq1_parents-fill_in_p1_sdq.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq1_parents-sdq_parent_report.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq2_parents-fill_in_p2_sdq.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq2_parents-sdq2_parent_report.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq_child-child_lit_repeat.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq_child-fill_in_c_sdq.do"
do "import_ADB_questionnaire_test-consented-available_today-all_sdq-sdq_child-sdq_child_report.do"
do "import_ADB_questionnaire_test-consented-available_today-child_edu-edu_repeat_c.do"
do "import_ADB_questionnaire_test-consented-available_today-child_health-child_health_repeat.do"
do "import_ADB_questionnaire_test-consented-available_today-education_g1-edu_repeat_g1.do"
do "import_ADB_questionnaire_test-consented-available_today-education_g2-edu_repeat_g2.do"
do "import_ADB_questionnaire_test-consented-available_today-labor_g1-labor_repeat_g1.do"
do "import_ADB_questionnaire_test-consented-available_today-labor_g2-labor_repeat_g2.do"
do "import_ADB_questionnaire_test-consented-short_roster-child_parent_link.do"
do "import_ADB_questionnaire_test-consented-short_roster-elig_member_repeat.do"
do "import_ADB_questionnaire_test-consented-short_roster-non_resident_roster-nr_roster_repeat.do"
do "import_ADB_questionnaire_test-consented-short_roster-nonparent_list.do"
do "import_ADB_questionnaire_test-consented-short_roster-parent_dedup.do"
do "import_ADB_questionnaire_test-consented-short_roster-resident_roster-other_members_grp-roster_repeat.do"
do "import_ADB_questionnaire_test-covid_instr-covid_instr_repeat.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-college_group_g2-college_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-college_group_g2-interruption_college_repeat_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-hs_group_g2-hs_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-hs_group_g2-interruption_hs_repeat_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-junior_hs_group_g2-interruption_junior_hs_repeat_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-junior_hs_group_g2-junior_highschool_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-move_history_g2-repeat_move_history_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-post_colleges_group_g2-interruption_post-college_repeat_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-post_colleges_group_g2-post_college_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-primary_school_history_g2-interruption_primary_repeat_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-primary_school_history_g2-primary_school_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-senior_hs_group_g2-interruption_senior_hs_repeat_g2.do"
do "import_ADB_questionnaire_test-edu_g2_available-consented_edu_g2-senior_hs_group_g2-senior_hs_g2.do"
do "import_ADB_questionnaire_test-hs_group_c-hs_c.do"
do "import_ADB_questionnaire_test-hs_group_c-interruption_hs_repeat_c.do"
do "import_ADB_questionnaire_test-hs_group_g1-hs_g1.do"
do "import_ADB_questionnaire_test-hs_group_g1-interruption_hs_repeat_g1.do"
do "import_ADB_questionnaire_test-junior_hs_group_c-interruption_junior_hs_repeat_c.do"
do "import_ADB_questionnaire_test-junior_hs_group_c-junior_highschool_c.do"
do "import_ADB_questionnaire_test-junior_hs_group_g1-interruption_junior_hs_repeat_g1.do"
do "import_ADB_questionnaire_test-junior_hs_group_g1-junior_highschool_g1.do"
do "import_ADB_questionnaire_test-live_else_col_g1.do"
do "import_ADB_questionnaire_test-live_else_col_g2.do"
do "import_ADB_questionnaire_test-live_else_hs_c.do"
do "import_ADB_questionnaire_test-live_else_hs_g1.do"
do "import_ADB_questionnaire_test-live_else_hs_g2.do"
do "import_ADB_questionnaire_test-live_else_jhs_c.do"
do "import_ADB_questionnaire_test-live_else_jhs_g1.do"
do "import_ADB_questionnaire_test-live_else_jhs_g2.do"
do "import_ADB_questionnaire_test-live_else_post_g1.do"
do "import_ADB_questionnaire_test-live_else_post_g2.do"
do "import_ADB_questionnaire_test-live_else_prim_c.do"
do "import_ADB_questionnaire_test-live_else_prim_g1.do"
do "import_ADB_questionnaire_test-live_else_prim_g2.do"
do "import_ADB_questionnaire_test-live_else_shs_c.do"
do "import_ADB_questionnaire_test-live_else_shs_g1.do"
do "import_ADB_questionnaire_test-live_else_shs_g2.do"
do "import_ADB_questionnaire_test-move_group_g1-repeat_move_history_g1.do"
do "import_ADB_questionnaire_test-post_colleges_group_g1-interruption_post-college_repeat_g1.do"
do "import_ADB_questionnaire_test-post_colleges_group_g1-post_college_g1.do"
do "import_ADB_questionnaire_test-primary_school_history_c-interruption_primary_repeat_c.do"
do "import_ADB_questionnaire_test-primary_school_history_c-primary_school_c.do"
do "import_ADB_questionnaire_test-primary_school_history_g1-interruption_primary_repeat_g1.do"
do "import_ADB_questionnaire_test-primary_school_history_g1-primary_school_g1.do"
do "import_ADB_questionnaire_test-senior_hs_group_c-interruption_senior_hs_repeat_c.do"
do "import_ADB_questionnaire_test-senior_hs_group_c-senior_hs_c.do"
do "import_ADB_questionnaire_test-senior_hs_group_g1-interruption_senior_hs_repeat_g1.do"
do "import_ADB_questionnaire_test-senior_hs_group_g1-senior_hs_g1.do"
