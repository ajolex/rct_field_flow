********************************************************************************
** 	TITLE	: 4b_additional_hfcs.do
**
**	PURPOSE	: 
**				
**	AUTHOR	: 
**
**	DATE	: 
********************************************************************************
	

*------------------------------------------------------------------------------*
* FOOD CONSUMPTION
*------------------------------------------------------------------------------*	
{ // food consumption
* reshaping to long by food category
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users count_res_members

	forvalues i = 1/9 {
		use "${checkedsurvey}", clear
		
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
		
		keep `generalkeep' fd_cons_val_`i'_? index_fd_cat_`i' fd_cat_`i' ///
						fd_cons_2a_`i'_? fd_cons_2aunit_`i'_? fd_cons_2aunitoth_`i'_? fd_cons_2aunit_lbl_`i'_? fd_cons_2b_`i'_?  ///
						fd_cons_3a_`i'_? fd_cons_3aunit_`i'_? fd_cons_3aunitoth_`i'_? fd_cons_3aunit_lbl_`i'_? fd_cons_3b_`i'_? ///
						fd_cons_4a_`i'_? fd_cons_4aunit_`i'_? fd_cons_4aunitoth_`i'_? fd_cons_4aunit_lbl_`i'_? fd_cons_4b_`i'_?
						
		rename *_`i'_* *_*
		rename (index_fd_cat_`i' fd_cat_`i') (index_fd_cat fd_cat)
		
		reshape long 	fd_cons_2a_ fd_cons_2aunit_ fd_cons_2aunitoth_ fd_cons_2aunit_lbl_ fd_cons_2b_  ///
						fd_cons_3a_ fd_cons_3aunit_ fd_cons_3aunitoth_ fd_cons_3aunit_lbl_ fd_cons_3b_ ///
						fd_cons_4a_ fd_cons_4aunit_ fd_cons_4aunitoth_ fd_cons_4aunit_lbl_ fd_cons_4b_ ///
						fd_cons_val_, i(caseid) j(fd_cons_index)
						
		rename *_ *
						
		tempfile food_`i'
		save `food_`i'', replace
	}

* appending & dropping blanks
	clear
	forvalues i = 1/9 {
		append using `food_`i''
	}
	
	drop if mi(fd_cons_val)

* Destring variables
	destring fd_cons_?a fd_cons_?aunit fd_cons_?b, replace
	
	forvalues i = 2/4 {
		recode fd_cons_`i'a fd_cons_`i'aunit fd_cons_`i'b (-888 = .r) (-999 = .d)  
	} 
	
	
* label
	label define fd_cons_units 					///
				1	"Kilograms (Kg)" 			///
				2	"Grams (g)"					///
				3	"Gallons"					///
				4	"5-gallon blue container"	///
				5	"Liters (L)"				///
				6	"Millileters (mL)"			///
				7	"Cans"						///
				8	"Cans (500 mL)"				///	
				9	"Cans (330 mL)"				///
				10	"Bottle"					///
				11	"Bottle (500 ml)"			///
				12	"Bottle (330 ml)"			///
				13	"Lipid / Lapad"				///
				14	"Long-neck"					///
				15	"Packs"						///
				16	"Large packs"				///
				17	"Medium packs"				///
				18	"Small packs"				///
				19	"Whole (chicken)"			///
				20	"Pieces or units"			///
				21	"Gantang"					///
				22	"Tumpok"					///
				23	"Small cup"					///
				24	"Salmon"					///
				25	"Letse"						///
				26	"Bundle"					///
				27	"Pesos"						///
				-666	"Other"					///
				.d	"Don't Know"				///
				.r	"Refuse to answer"			//
	
label val fd_cons_?aunit fd_cons_units

* Reshape variables again (interested in price per unit, don't care about how food was aquired)
	drop if caseid == "H030727034" & fd_cat == "Vegetables" & fd_cons_val == "meat_2" // dropping strange case

	gen fd_cons_val_2 = fd_cons_val
	gen fd_cons_val_3 = fd_cons_val
	gen fd_cons_val_4 = fd_cons_val
	
	gen rowid = caseid + " " + fd_cons_val
	
	drop fd_cons_val
	
	rename 	(fd_cons_?a fd_cons_?aunit fd_cons_?b fd_cons_?aunitoth fd_cons_?aunit_lbl) ///
			(fd_cons_a_? fd_cons_aunit_? fd_cons_b_? fd_cons_aunitoth_? fd_cons_aunit_lbl_?)

	reshape long fd_cons_val_ fd_cons_a_ fd_cons_aunit_ fd_cons_b_ fd_cons_aunitoth_ fd_cons_aunit_lbl_, i(rowid) j(aquired)

	
	rename *_ *
	label define aquired 2 "purchased" 3 "produced" 4 "gifts"
	label val aquired aquired
	
	drop if fd_cons_b == 0
	
	
* Checking for the outliers
	gen food_item_unit = fd_cons_val + "_" + string(fd_cons_aunit)
	
	gen unit_price = fd_cons_b
		replace unit_price = fd_cons_b/fd_cons_a if !mi(fd_cons_a)
	
	gen quant_outlier = 0
	gen price_outlier = 0
	
	gen quant_iqr = .
	gen price_iqr = .
	
	gen quant_av = . 
	gen price_av = .
	
* Find outliers
	levelsof(food_item_unit), loc(items)
	foreach itm of local items {
		
		* quant version
		qui count if !mi(fd_cons_a) & food_item_unit == "`itm'" 
		
		if `r(N)' > 0 {
			qui summ fd_cons_a if food_item_unit == "`itm'", d
			
			replace quant_iqr = r(p75)-r(p25) if food_item_unit == "`itm'"
			replace quant_av = r(mean) if food_item_unit == "`itm'"
			
			replace quant_outlier = 1 if ((fd_cons_a > `r(p75)' + (1.5*quant_iqr)) | (fd_cons_a < `r(p25)' - (1.5*quant_iqr))) & food_item_unit == "`itm'" & !mi(fd_cons_a)
		}
		
			qui summ unit_price if food_item_unit == "`itm'", d
		if `r(N)' > 1 {
				
			replace price_iqr = r(p75)-r(p25) if food_item_unit == "`itm'"
			replace price_av = r(mean) if food_item_unit == "`itm'"
			replace price_outlier = 1 if ((unit_price > `r(p75)' + (1.5*price_iqr)) | (unit_price < `r(p25)' - (1.5*price_iqr))) & food_item_unit == "`itm'" & !mi(unit_price)
		}
	}
	
preserve
	import delimited "${tempfolder}/preload_fd_cat", clear
	keep cons fd_cons_name
	rename cons fd_cons_val
	
	tempfile foodcodes
	save `foodcodes'
restore

	merge m:1 fd_cons_val using `foodcodes', keep(match) nogen
	
*=================================
*	OUTPUT
*=================================
	cap mkdir "${checkfolder}/${folder_date}/hfc_consumption"
	
	* Food "Other" Unit Options
preserve
	keep if !mi(fd_cons_aunitoth)
	
	loc keepvars subdate caseid fo fc_name pull_municipal_city pull_barangay fd_cons_val fd_cons_name fd_cons_aunitoth
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_consumption/1_fd_other", replace	
restore
	
	* Outliers Price
preserve
	keep if price_outlier == 1
	decode fd_cons_aunit, gen(fd_cons_aunit_str)
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay fd_cons_val ///
	fd_cons_name unit_price fd_cons_aunit_str price_av fd_cons_b fd_cons_a count_res_members startdate subdate
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_consumption/2_fd_price_outliers", replace
restore
	
	* Outliers Quantity	
preserve

	keep if quant_outlier == 1
	decode fd_cons_aunit, gen(fd_cons_aunit_str)
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay fd_cons_val ///
	fd_cons_name fd_cons_a fd_cons_aunit_str quant_av count_res_members startdate subdate
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_consumption/2_fd_quant_outliers", replace
restore	
}

*------------------------------------------------------------------------------*
* PREPARED FOOD CONSUMPTION
*------------------------------------------------------------------------------*
{ // prepped food consumption

loc generalkeep	pull_brgy_prefix caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users count_res_members startdate subdate


	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
	
	
	
	keep `generalkeep'  fd_cons_prep_val_? index_fd_cons_prep_? ///
						fd_cons_5a_? fd_cons_5b_? fd_cons_6a_? fd_cons_6b_?
					
	reshape long 	fd_cons_prep_val_ index_fd_cons_prep_ ///
					fd_cons_5a_ fd_cons_5b_ fd_cons_6a_ fd_cons_6b_, i(caseid) j(fd_cons_index)
						
	rename *_ *
	
	* Quick test
		gen false_launch = 1 if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
				
	drop if mi(fd_cons_prep_val)

* Destring variables
	destring fd_cons_5a fd_cons_6a fd_cons_6b, replace
	recode fd_cons_5a fd_cons_6a fd_cons_6b (-888 = .r) (-999 = .d)  
	
* Get the names of the variables
preserve
	import delimited "${tempfolder}/preload_fd_cat", clear
	keep cons fd_cons_name
	rename cons fd_cons_prep_val
	
	tempfile foodcodes
	save `foodcodes'
restore

	merge m:1 fd_cons_prep_val using `foodcodes', keep(match) nogen
	
* Checking for the outliers
	egen prep_total = rowtotal(fd_cons_6a fd_cons_6b), m

	gen av_price = prep_total/fd_cons_5a if !mi(fd_cons_5a)
	
	gen price_outlier = 0
	gen price_iqr = .
	gen price_av = .
	
* Find outliers
	levelsof(fd_cons_prep_val), loc(items)
	foreach itm of local items {
				
		qui summ av_price if fd_cons_prep_val == "`itm'", d
		if `r(N)' > 1 {
			replace price_iqr = r(p75)-r(p25) if fd_cons_prep_val == "`itm'"
			replace price_av = r(mean) if fd_cons_prep_val == "`itm'"
			replace price_outlier = 1 if ((av_price > `r(p75)' + (1.5*price_iqr)) | (av_price < `r(p25)' - (1.5*price_iqr))) & fd_cons_prep_val == "`itm'" & !mi(av_price)
		}
	}	

* 

	* Food prep table	
preserve
	keep if price_outlier == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	fd_cons_prep_val fd_cons_name prep_total av_price fd_cons_5a price_av count_res_members
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_consumption/4_fd_prep_outliers", replace
	
restore	
}

*------------------------------------------------------------------------------*
* NON-FOOD CONSUMPTION
*------------------------------------------------------------------------------*
{ // non-food consumption
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users count_res_members

* reshaping to long by non-food category	
	forvalues i = 1/7 {
		use "${checkedsurvey}", clear
		
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
		keep `generalkeep' nfd_cons_val_`i'_? nfd_cons_2_`i'_? nfd_cons_3_`i'_? nfd_cons_4_`i'_? 
						
		rename *_`i'_* *_*
		
		reshape long 	nfd_cons_val_ nfd_cons_2_ nfd_cons_3_ nfd_cons_4_, i(caseid) j(nfd_cons_index)
						
		rename *_ *
						
		tempfile nonfood_`i'
		save `nonfood_`i'', replace
	}

* appending & dropping blanks
	clear
	forvalues i = 1/7 {
		append using `nonfood_`i''
	}
	
	drop if mi(nfd_cons_val)

* Destring variables & recode
	destring nfd_cons_2 nfd_cons_3 nfd_cons_4, replace
	recode nfd_cons_2 nfd_cons_3 nfd_cons_4 (-888 = .r) (-999 = .d)  
	
* Get the names of the variables
preserve
	import delimited "${tempfolder}/preload_nfd_cat", clear
	keep nfd_cons nfd_cons_name
	rename nfd_cons nfd_cons_val
	
	tempfile nonfoodcodes
	save `nonfoodcodes'
restore

	merge m:1 nfd_cons_val using `nonfoodcodes', keep(match) nogen
	
* Checking for the outliers
	egen nfd_total = rowtotal(nfd_cons_2 nfd_cons_3 nfd_cons_4), m
	drop if nfd_total == 0 // shouldn't be zero

	gen amt_outlier = 0
	gen amt_iqr = .
	gen amt_av = .
	
* Find outliers
	levelsof(nfd_cons_val), loc(items)
	foreach itm of local items {
				
		qui summ nfd_total if nfd_cons_val == "`itm'", d
		if `r(N)' > 1 {
			replace amt_iqr = r(p75)-r(p25) if nfd_cons_val == "`itm'"
			replace amt_av = r(mean) if nfd_cons_val == "`itm'"
			replace amt_outlier = 1 if ((nfd_total > `r(p75)' + (1.5*amt_iqr)) | (nfd_total < `r(p25)' - (1.5*amt_iqr))) & nfd_cons_val == "`itm'" & !mi(nfd_total)
		}
	}	
	
	* NON food consumption outliers
	
preserve
	keep if amt_outlier == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	nfd_cons_val nfd_cons_name nfd_total amt_av nfd_cons_2 nfd_cons_3 nfd_cons_4 count_res_members
	
	rename (nfd_cons_2 nfd_cons_3 nfd_cons_4) (purchase produced gifted)
	
	save "${checkfolder}/${folder_date}/hfc_consumption/5_nonfood_outliers", replace
restore	
}

*------------------------------------------------------------------------------*
* CROPS
*------------------------------------------------------------------------------*
{ // crops
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users count_res_members

	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
	
	keep `generalkeep'  index_crops_? index_crops_?? calc_crop_name_? calc_crop_name_?? ///
		 agri_crops_szn_? agri_crops_szn_?? agri_crops_snn_no_? agri_crops_snn_no_?? ///
		 agri_crops_qty_? agri_crops_qty_?? agri_crops_sale_ltm_? agri_crops_sale_ltm_?? ///
		 agri_crops_rev_? agri_crops_rev_?? agri_crops_no_sale_? agri_crops_no_sale_??

	reshape long index_crops_ calc_crop_name_ agri_crops_szn_ agri_crops_snn_no_ agri_crops_qty_ agri_crops_sale_ltm_ agri_crops_rev_ agri_crops_no_sale_ , i(caseid) j(index2)
	
	rename *_ *
	drop if mi(index_crops)
	
	* 
	
		
	* Change crop names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' = calc_crop_name // item to merge with
	}
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "crops"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore

	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen crop_name = "", a(calc_crop_name)
	foreach lang of local languages {
		replace crop_name = `lang'_eng if mi(crop_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace crop_name = calc_crop_name if mi(crop_name) // adding the cases from "other"
		drop calc_crop_name
	
* Create a test for the outliers
	* Quantity harvested
	* Revenue (given quantity?)
	* Number of harvest seasons?
	
	gen rev_kg = agri_crops_rev/agri_crops_qty
	
	* Find outliers
	levelsof(crop_name), loc(items)
	
	loc outliervars agri_crops_qty rev_kg agri_crops_snn_no
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different crops
		foreach itm of local items {
	
			qui summ `var' if crop_name == "`itm'", d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if crop_name == "`itm'"
				replace `var'_av = r(mean) if crop_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & crop_name == "`itm'" & !mi(`var')
			} // close outlier identification
		} // close crop loop
	} // close outlier variable loop
	
	
** Create Tables
	
	* quantity
	cap mkdir "${checkfolder}/${folder_date}/hfc_crops"
	
preserve
	keep if agri_crops_qty_outlier == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate crop_name ///
	agri_crops_qty  agri_crops_qty_av
	
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_crops/1_quantity", replace
restore	
	
	* Revenue	
preserve

	decode agri_crops_sale_ltm, gen(agri_crops_sale_ltm2)
	drop agri_crops_sale_ltm
	rename agri_crops_sale_ltm2 agri_crops_sale_ltm
	
	keep if rev_kg_outlier == 1
	
	loc keepvars 	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	crop_name rev_kg rev_kg_av agri_crops_qty agri_crops_sale_ltm 
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_crops/2_revenue", replace	
restore		
	
	* Season
	putexcel set "${checkfolder}/${folder_date}/hfc_crops.xlsx", modify sh(3_season, replace)
	
preserve
	keep if agri_crops_snn_no_outlier == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	crop_name agri_crops_snn_no agri_crops_snn_no_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_crops/3_season", replace	
	
restore		
	
	
}
	
	
*------------------------------------------------------------------------------*
* AGRICULTURE PLOTS & INPUTS
*------------------------------------------------------------------------------*
{
loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users count_res_members

	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
	
	* count number of plots
	qui summ agri_plot_n
	
	gen tag_diffunit = 0
	
	* standardize plot size into sqm
	forvalues i = 1/`r(max)' {
		
		loc j = `i' - 1
		
		gen plot_area_sqm_`i' = ., a(agri_plot_area_unit_oth_`i')
		
		
		* create conversion
		replace plot_area_sqm_`i' = agri_plot_area_`i' * 4046.86 		if agri_plot_area_unit_`i' == 1 // acre to sqm
		replace plot_area_sqm_`i' = agri_plot_area_`i' * 10000 			if agri_plot_area_unit_`i' == 2 // hectare to sqm
		replace plot_area_sqm_`i' = agri_plot_area_`i' 					if agri_plot_area_unit_`i' == 3 // square meters no conversion
		replace plot_area_sqm_`i' = agri_plot_area_`i' * 0.836 			if agri_plot_area_unit_`i' == 4 // square yards to sqm
		
		* tag cases
		if `i' != 1 	replace tag_diffunit = 1 if (agri_plot_area_unit_`i' != agri_plot_area_unit_`j') & !mi(agri_plot_area_unit_`i')
	}

	forvalues i = 1/15 {
		di "--- count `i' --- "
		tab agri_plot_area_unit_oth_`i'
	}
	
	gen str_allplots = ""
	
	forvalues i = 1/15 {
		
		replace str_allplots = str_allplots + string(agri_plot_area_`i') + " acres, " 			if agri_plot_area_unit_`i' == 1
		replace str_allplots = str_allplots + string(agri_plot_area_`i') + " hectare, " 		if agri_plot_area_unit_`i' == 2
		replace str_allplots = str_allplots + string(agri_plot_area_`i') + " square meters, " 	if agri_plot_area_unit_`i' == 3
		replace str_allplots = str_allplots + string(agri_plot_area_`i') + " square yards, " 	if agri_plot_area_unit_`i' == 4
		replace str_allplots = str_allplots + string(agri_plot_area_`i') + agri_plot_area_unit_oth_`i'	if agri_plot_area_unit_`i' == -666
	}
	
	
	egen plot_area_sqm = rowtotal(plot_area_sqm_*), m
	
	
		gen plot_area_sqm_outlier = 0
		gen plot_area_sqm_iqr = .
		gen plot_area_sqm_av = .
		
		
			qui summ plot_area_sqm, d
			if `r(N)' > 1 {
					
				replace plot_area_sqm_iqr = r(p75)-r(p25)
				replace plot_area_sqm_av = r(mean)
				replace plot_area_sqm_outlier = 1 if ((plot_area_sqm > `r(p75)' + (1.5*plot_area_sqm_iqr)) | (plot_area_sqm < `r(p25)' - (1.5*plot_area_sqm_iqr))) & !mi(plot_area_sqm)
			} // close outlier identification

			
** Table for Plot Size Outliers

		cap mkdir "${checkfolder}/${folder_date}/hfc_plots"
preserve
	keep if plot_area_sqm_outlier == 1
	
	loc keepvars	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate agri_plot_n ///
	plot_area_sqm plot_area_sqm_av str_allplots
	
	
	keep `keepvars'
	order `keepvars' 
	
	save "${checkfolder}/${folder_date}/hfc_plots/1_plotsize", replace
restore	
	
** Inputs

loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users 

	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 

	keep `generalkeep' index_cost_items_? temp_cost_items_name_? agri_cost_?
	
	reshape long index_cost_items_ temp_cost_items_name_ agri_cost_, i(caseid) j(index_num)
	
	rename *_ *
	drop if mi(temp_cost_items_name)
	
	
* Translations
	* Change crop names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' = temp_cost_items_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "agri_cost_ltm2"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore

	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen item_name = "", a(temp_cost_items_name)
	foreach lang of local languages {
		replace item_name = `lang'_eng if mi(item_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace item_name = temp_cost_items_name if mi(item_name) // adding the cases from "other"
		drop temp_cost_items_name	
	
	
** Identify outliers

		gen agri_cost_outlier = 0
		gen agri_cost_iqr = .
		gen agri_cost_av = .
		
		levelsof(item_name), loc(items)
		* different inputs
		foreach itm of local items {
	
			qui summ agri_cost if item_name == "`itm'", d
			if `r(N)' > 1 {
					
				replace agri_cost_iqr = r(p75)-r(p25) if item_name == "`itm'"
				replace agri_cost_av = r(mean) if item_name == "`itm'"
				replace agri_cost_outlier = 1 if ((agri_cost > `r(p75)' + (1.5*agri_cost_iqr)) | (agri_cost < `r(p25)' - (1.5*agri_cost_iqr))) & item_name == "`itm'" & !mi(agri_cost)
			} // close outlier identification
		} // close item loop
	
	
** Table

preserve
	keep if agri_cost_outlier == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate item_name ///
	item_name agri_cost agri_cost_av
	
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_plots/2_inputs", replace
restore	
}


*------------------------------------------------------------------------------*
* LIVESTOCK
*------------------------------------------------------------------------------*
{
* Bought - number bought
* Price per animal
* Currently own
* Currently raise
* Price of animal (sold all of them)
* Gifts
* stolen
* slaughter
* disease
	cap mkdir "${checkfolder}/${folder_date}/hfc_livestock"
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users

	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
	
	keep `generalkeep'  calc_lv_name_? lv_buy_no_? lv_buy_price_? lv_total_own_? lv_total_raise_? lv_t_value_? lv_gifts_? lv_stolen_? lv_slaughter_? lv_disease_? lv_disease_num_?

	reshape long calc_lv_name_ lv_buy_no_ lv_buy_price_ lv_total_own_ lv_total_raise_ lv_t_value_ lv_gifts_ lv_stolen_ lv_slaughter_ lv_disease_ lv_disease_num_, i(caseid) j(index2)
	
	rename *_ *
	drop if mi(calc_lv_name)

* --- 	TRANSLATIONS --- *
	* Change names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' = calc_lv_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "livestock"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore



	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen lv_name = "", a(calc_lv_name)
	foreach lang of local languages {
		replace lv_name = `lang'_eng if mi(lv_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace lv_name = calc_lv_name if mi(lv_name) // adding the cases from "other"
		drop calc_lv_name	


* Gen variables
	gen buy_price = lv_buy_price/lv_buy_no
	gen own_price = lv_t_value/lv_total_own
	
** Outliers **
	
* Find outliers
	levelsof(lv_name), loc(items)
	
	loc outliervars lv_buy_no buy_price lv_total_own lv_total_raise own_price lv_gifts lv_stolen lv_slaughter lv_disease_num
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different livestock
		foreach itm of local items {
	
			qui summ `var' if lv_name == "`itm'" & `var' != 0, d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if lv_name == "`itm'"
				replace `var'_av = r(mean) if lv_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & lv_name == "`itm'" & !mi(`var')
			} // close outlier identification
		} // close livestock loop
	} // close outlier variable loop

** Tables

* Animals bought
	

preserve
	keep if lv_buy_no_outlier == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate lv_name ///
	lv_buy_no lv_buy_no_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_livestock/1_num_buy", replace
restore	


* Price per animal bought
preserve
	keep if buy_price_outlier == 1 | buy_price < 100
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name buy_price buy_price_av lv_buy_no lv_buy_price
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_livestock/2_buy_price", replace
restore	



* Bought - number bought (lv_buy_no)
* Price per animal (buy_price)
* Currently own (lv_total_own)
* Currently raise (lv_total_raise)
* Price of animal - sold all of them (own_price) 
* Gifts lv_gifts 
* stolen lv_stolen 
* slaughter lv_slaughter 
* disease lv_disease_num

preserve
	keep if lv_total_own_outlier == 1 & lv_total_own != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name lv_total_own lv_total_own_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_livestock/3_lv_own", replace
restore	

preserve
	keep if lv_total_raise_outlier == 1 & lv_total_raise != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate lv_name lv_total_raise lv_total_raise_av 

	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_livestock/4_lv_raise", replace
restore	

* Price per animal if sold
preserve
	keep if own_price_outlier == 1 | own_price < 100
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name own_price own_price_av lv_total_own lv_t_value
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_livestock/5_own_price", replace
restore	

preserve
	keep if lv_gifts_outlier == 1 & lv_gifts != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name lv_gifts lv_gifts_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_livestock/6_lv_gifts", replace
restore	

preserve
	keep if lv_stolen_outlier == 1 & lv_stolen != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name lv_stolen lv_stolen_av
	
	keep `keepvars'
	order `keepvars'	
	
	
	save "${checkfolder}/${folder_date}/hfc_livestock/7_lv_stolen", replace
restore	


preserve
	keep if lv_slaughter_outlier == 1 & lv_slaughter != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name lv_slaughter lv_slaughter_av
	
	keep `keepvars'
	order `keepvars'	
	
	save "${checkfolder}/${folder_date}/hfc_livestock/8_lv_slaughter", replace
restore	

preserve
	keep if lv_disease_num_outlier == 1 & lv_disease_num != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	lv_name lv_disease_num lv_disease_num_av
	
	keep `keepvars'
	order `keepvars'	
	
	save "${checkfolder}/${folder_date}/hfc_livestock/9_lv_disease_num", replace
restore	
}

*------------------------------------------------------------------------------*
* POULTRY
*------------------------------------------------------------------------------*
{
	
* Bought - number bought
* Price per animal
* Currently own
* Currently raise
* Price of animal (sold all of them)
* Gifts
* stolen
* slaughter
* disease
	cap mkdir "${checkfolder}/${folder_date}/hfc_poultry"
	
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users

	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
	
	keep `generalkeep'  calc_pltry_name_? pltry_buy_no_? pltry_buy_price_? pltry_total_own_? pltry_total_raise_? pltry_t_value_? pltry_gifts_? pltry_stolen_? pltry_slaughter_? pltry_disease_? pltry_disease_num_?

	reshape long calc_pltry_name_ pltry_buy_no_ pltry_buy_price_ pltry_total_own_ pltry_total_raise_ pltry_t_value_ pltry_gifts_ pltry_stolen_ pltry_slaughter_ pltry_disease_ pltry_disease_num_, i(caseid) j(index2)
	
	rename *_ *
	drop if mi(calc_pltry_name)

* --- 	TRANSLATIONS --- *
	* Change names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' = calc_pltry_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "poultry_list"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore



	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen pltry_name = "", a(calc_pltry_name)
	foreach lang of local languages {
		replace pltry_name = `lang'_eng if mi(pltry_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace pltry_name = calc_pltry_name if mi(pltry_name) // adding the cases from "other"
		drop calc_pltry_name	


* Gen variables
	gen buy_price = pltry_buy_price/pltry_buy_no
	gen own_price = pltry_t_value/pltry_total_own
	
** Outliers **
	
* Find outliers
	levelsof(pltry_name), loc(items)
	
	loc outliervars pltry_buy_no buy_price pltry_total_own pltry_total_raise own_price pltry_gifts pltry_stolen pltry_slaughter pltry_disease_num
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different Poultry
		foreach itm of local items {
	
			qui summ `var' if pltry_name == "`itm'" & `var' != 0, d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if pltry_name == "`itm'"
				replace `var'_av = r(mean) if pltry_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & pltry_name == "`itm'" & !mi(`var')
			} // close outlier identification
		} // close Poultry loop
	} // close outlier variable loop

** Tables

* Animals bought
preserve
	keep if pltry_buy_no_outlier == 1
	
	
	loc keepvars	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
					pltry_name pltry_buy_no pltry_buy_no_av
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/1_num_buy", replace
restore	


* Price per animal bought
preserve
	keep if buy_price_outlier == 1 | buy_price < 100
	
	loc keepvars	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
					pltry_name buy_price buy_price_av pltry_buy_no pltry_buy_price
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/2_buy_price", replace
restore	



* Bought - number bought (pltry_buy_no)
* Price per animal (buy_price)
* Currently own (pltry_total_own)
* Currently raise (pltry_total_raise)
* Price of animal - sold all of them (own_price) 
* Gifts pltry_gifts 
* stolen pltry_stolen 
* slaughter pltry_slaughter 
* disease pltry_disease_num

preserve
	keep if pltry_total_own_outlier == 1 & pltry_total_own != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	pltry_total_own pltry_total_own_av
	
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/3_pltry_own", replace
restore	

preserve
	keep if pltry_total_raise_outlier == 1 & pltry_total_raise != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	pltry_total_raise pltry_total_raise_av
	
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/4_pltry_raise", replace
restore	

* Price per animal if sold
preserve
	keep if own_price_outlier == 1 | own_price < 100
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	own_price own_price_av pltry_total_own pltry_t_value
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/5_own_price", replace
restore	

preserve
	keep if pltry_gifts_outlier == 1 & pltry_gifts != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	pltry_gifts pltry_gifts_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/6_pltry_gifts", replace
restore	

preserve
	keep if pltry_stolen_outlier == 1 & pltry_stolen != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	pltry_stolen pltry_stolen_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/7_pltry_stolen", replace
restore	


preserve
	keep if pltry_slaughter_outlier == 1 & pltry_slaughter != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	pltry_slaughter pltry_slaughter_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/8_pltry_slaughter", replace
restore	

preserve
	keep if pltry_disease_num_outlier == 1 & pltry_disease_num != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate pltry_name ///
	pltry_disease_num pltry_disease_num_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_poultry/9_pltry_disease_num", replace
restore	
}


*------------------------------------------------------------------------------*
* ASSETS
*------------------------------------------------------------------------------*
{
loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users count_res_members
	
	
	
*=================================
*	Vehicles
*=================================

{
	cap mkdir "${checkfolder}/${folder_date}/hfc_asset_vehicle"
	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 


	keep `generalkeep' index_asset_vehicle_? calc_asset_vehicle_name_? calc_asset_vehicle_val_? asset_vehicle_2_? ///
	asset_vehicle_3a_? asset_vehicle_3b_? asset_vehicle_4a_? asset_vehicle_4b_? asset_vehicle_5_?

	reshape long index_asset_vehicle_ calc_asset_vehicle_name_ calc_asset_vehicle_val_ asset_vehicle_2_ ///
	asset_vehicle_3a_ asset_vehicle_3b_ asset_vehicle_4a_ asset_vehicle_4b_ asset_vehicle_5_, i(caseid) j(index2)
	
	rename *_ *
	
	drop if mi(index_asset_vehicle)
	
* TRANSLATIONS
* --- 	TRANSLATIONS --- *
	* Change names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' =  calc_asset_vehicle_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "vehicle_list"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore

	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen vehicle_name = "", a(calc_asset_vehicle_name)
	foreach lang of local languages {
		replace vehicle_name = `lang'_eng if mi(vehicle_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace vehicle_name = calc_asset_vehicle_name if mi(vehicle_name) // adding the cases from "other"
		drop calc_asset_vehicle_name
* --- END TRANSLATIONS --- *

* Gen variables
	gen new_price_unit = asset_vehicle_4a/asset_vehicle_3a
	gen sec_price_unit = asset_vehicle_4b/asset_vehicle_3b
	
** Outliers **
	
* Find outliers
	levelsof(vehicle_name), loc(items)
	
	loc outliervars asset_vehicle_2 asset_vehicle_3a asset_vehicle_3b new_price_unit sec_price_unit asset_vehicle_5
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different vehicles
		foreach itm of local items {
	
			qui summ `var' if vehicle_name == "`itm'" & `var' != 0, d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if vehicle_name == "`itm'"
				replace `var'_av = r(mean) if vehicle_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & vehicle_name == "`itm'" & !mi(`var')
				
				if inlist("`var'","asset_vehicle_3a","asset_vehicle_3b", "asset_vehicle_5") {
					replace `var'_outlier = 0 if `var' == 0 // changing back cases to not flag as outliers
				}
						
				
			} // close outlier identification
		} // close vehicle loop
	} // close outlier variable loop

	replace new_price_unit_outlier = 1 if new_price_unit <= 1000 & !mi(new_price_unit) // want to flag things that look very low
	replace sec_price_unit_outlier = 1 if sec_price_unit <= 1000 & !mi(sec_price_unit) // want to flag things that look very low
	
** TABLES

* Number total owned
preserve
	keep if asset_vehicle_2_outlier == 1 & asset_vehicle_2 != 0
	
	loc keepvars 	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	vehicle_name asset_vehicle_2 asset_vehicle_2_av count_res_members
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_vehicle/1_own", replace
restore		

* Buy new
preserve
	keep if asset_vehicle_3a_outlier == 1 & asset_vehicle_3a != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	vehicle_name asset_vehicle_3a asset_vehicle_3a_av
	
	keep `keepvars'
	order `keepvars'	
	
	save "${checkfolder}/${folder_date}/hfc_asset_vehicle/2_buy_new", replace
restore

* Buy second hand 
preserve
	keep if asset_vehicle_3b_outlier == 1 & asset_vehicle_3b != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	vehicle_name asset_vehicle_3b asset_vehicle_3b_av
	
	keep `keepvars'
	order `keepvars'	
	
	save "${checkfolder}/${folder_date}/hfc_asset_vehicle/3_buy_sec", replace
restore

* Unit price new
preserve
	keep if new_price_unit_outlier == 1 & new_price_unit != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	vehicle_name new_price_unit new_price_unit_av asset_vehicle_3a asset_vehicle_4a
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_vehicle/4_unitprice_new", replace
restore

* Unit price old
preserve
	keep if sec_price_unit_outlier == 1 & sec_price_unit != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	vehicle_name sec_price_unit sec_price_unit_av asset_vehicle_3b asset_vehicle_4b
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_vehicle/5_unitprice_sec", replace
restore

* Repairs
preserve
	keep if asset_vehicle_5_outlier == 1 & asset_vehicle_5 != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	vehicle_name asset_vehicle_5 asset_vehicle_5_av asset_vehicle_2
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_vehicle/6_repairs", replace
restore
}


*=================================
*	Equipment
*=================================
{
	use "${checkedsurvey}", clear
	cap mkdir "${checkfolder}/${folder_date}/hfc_asset_equipment"
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 


	keep `generalkeep' index_asset_equip_? calc_asset_equip_name_? calc_asset_equip_val_? asset_equip_2_? ///
	asset_equip_3a_? asset_equip_3b_? asset_equip_4a_? asset_equip_4b_? 

	reshape long index_asset_equip_ calc_asset_equip_name_ calc_asset_equip_val_ asset_equip_2_ ///
	asset_equip_3a_ asset_equip_3b_ asset_equip_4a_ asset_equip_4b_ , i(caseid) j(index2)
	
	rename *_ *
	
	drop if mi(index_asset_equip)
	
* TRANSLATIONS
* --- 	TRANSLATIONS --- *
	* Change names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' =  calc_asset_equip_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "equip_list"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore

	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen equip_name = "", a(calc_asset_equip_name)
	foreach lang of local languages {
		replace equip_name = `lang'_eng if mi(equip_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace equip_name = calc_asset_equip_name if mi(equip_name) // adding the cases from "other"
		drop calc_asset_equip_name
* --- END TRANSLATIONS --- *

* Gen variables
	gen new_price_unit = asset_equip_4a/asset_equip_3a
	gen sec_price_unit = asset_equip_4b/asset_equip_3b
	
** Outliers **
	
* Find outliers
	levelsof(equip_name), loc(items)
	
	loc outliervars asset_equip_2 asset_equip_3a asset_equip_3b new_price_unit sec_price_unit 
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different Equipments
		foreach itm of local items {
	
			qui summ `var' if equip_name == "`itm'" & `var' != 0, d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if equip_name == "`itm'"
				replace `var'_av = r(mean) if equip_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & equip_name == "`itm'" & !mi(`var')
				
				if inlist("`var'","asset_equip_3a","asset_equip_3b") {
					replace `var'_outlier = 0 if `var' == 0 // changing back cases to not flag as outliers
				}
						
				
			} // close outlier identification
		} // close Equipment loop
	} // close outlier variable loop

	replace new_price_unit_outlier = 1 if new_price_unit <= 200 & !mi(new_price_unit) // want to flag things that look very low
	replace sec_price_unit_outlier = 1 if sec_price_unit <= 200 & !mi(sec_price_unit) // want to flag things that look very low
	
** TABLES

* Number total owned
preserve
	keep if asset_equip_2_outlier == 1 & asset_equip_2 != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	equip_name asset_equip_2 asset_equip_2_av count_res_members
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_equipment/1_own", replace
restore		

* Buy new
preserve
	keep if asset_equip_3a_outlier == 1 & asset_equip_3a != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	equip_name asset_equip_3a asset_equip_3a_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_equipment/2_buy_new", replace
restore

* Buy second hand 
preserve
	keep if asset_equip_3b_outlier == 1 & asset_equip_3b != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	equip_name asset_equip_3b asset_equip_3b_av
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_equipment/3_buy_sec", replace
restore

* Unit price new
preserve
	keep if new_price_unit_outlier == 1 & new_price_unit != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	equip_name new_price_unit new_price_unit_av asset_equip_3a asset_equip_4a
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_equipment/4_unitprice_new", replace
restore

* Unit price old
preserve
	keep if sec_price_unit_outlier == 1 & sec_price_unit != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	equip_name sec_price_unit sec_price_unit_av asset_equip_3b asset_equip_4b
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_equipment/5_unitprice_sec", replace
restore


}

*=================================
*	APPLIANCE
*=================================
{
	cap mkdir "${checkfolder}/${folder_date}/hfc_asset_appliance"
	
	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 


	keep `generalkeep' index_asset_appliance_? calc_asset_appliance_name_? calc_asset_appliance_val_? asset_appliance_2_? ///
	asset_appliance_3a_? asset_appliance_3b_? asset_appliance_4a_? asset_appliance_4b_? 

	reshape long index_asset_appliance_ calc_asset_appliance_name_ calc_asset_appliance_val_ asset_appliance_2_ ///
	asset_appliance_3a_ asset_appliance_3b_ asset_appliance_4a_ asset_appliance_4b_ , i(caseid) j(index2)
	
	rename *_ *
	
	drop if mi(index_asset_appliance)
	
* TRANSLATIONS
* --- 	TRANSLATIONS --- *
	* Change names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' =  calc_asset_appliance_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "appliance_list"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore

	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen equip_name = "", a(calc_asset_appliance_name)
	foreach lang of local languages {
		replace equip_name = `lang'_eng if mi(equip_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	
		replace equip_name = calc_asset_appliance_name if mi(equip_name) // adding the cases from "other"
		drop calc_asset_appliance_name
* --- END TRANSLATIONS --- *

* Gen variables
	gen new_price_unit = asset_appliance_4a/asset_appliance_3a
	gen sec_price_unit = asset_appliance_4b/asset_appliance_3b
	
** Outliers **
	
* Find outliers
	levelsof(equip_name), loc(items)
	
	loc outliervars asset_appliance_2 asset_appliance_3a asset_appliance_3b new_price_unit sec_price_unit 
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different appliances
		foreach itm of local items {
	
			qui summ `var' if equip_name == "`itm'" & `var' != 0, d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if equip_name == "`itm'"
				replace `var'_av = r(mean) if equip_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & equip_name == "`itm'" & !mi(`var')
				
				if inlist("`var'","asset_appliance_3a","asset_appliance_3b") {
					replace `var'_outlier = 0 if `var' == 0 // changing back cases to not flag as outliers
				}
						
				
			} // close outlier identification
		} // close appliance loop
	} // close outlier variable loop

	replace new_price_unit_outlier = 1 if new_price_unit <= 200 & !mi(new_price_unit) // want to flag things that look very low
	replace sec_price_unit_outlier = 1 if sec_price_unit <= 200 & !mi(sec_price_unit) // want to flag things that look very low
	
** TABLES

* Number total owned
preserve
	keep if asset_appliance_2_outlier == 1 & asset_appliance_2 != 0
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate ///
	equip_name asset_appliance_2 asset_appliance_2_av count_res_members
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_asset_appliance/1_own", replace 
restore		

* Buy new
preserve
	keep if asset_appliance_3a_outlier == 1 & asset_appliance_3a != 0
		
	loc keepvars 	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate equip_name asset_appliance_3a asset_appliance_3a_av
	keep `keepvars'
	order `keepvars'
		
	save "${checkfolder}/${folder_date}/hfc_asset_appliance/2_buy_new", replace 

restore

* Buy second hand 
preserve
	
	keep if asset_appliance_3b_outlier == 1 & asset_appliance_3b != 0
			
	loc keepvars 	caseid fo fc_name pull_municipal_city pull_barangay startdate subdate equip_name asset_appliance_3b asset_appliance_3b_av
	keep `keepvars'
	order `keepvars'
			
	save "${checkfolder}/${folder_date}/hfc_asset_appliance/3_buy_sec", replace
restore

* Unit price new
preserve
	keep if new_price_unit_outlier == 1 & new_price_unit != 0
			
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate equip_name new_price_unit new_price_unit_av asset_appliance_3a asset_appliance_4a
	keep `keepvars'
	order `keepvars'
			
	save "${checkfolder}/${folder_date}/hfc_asset_appliance/4_unitprice_new", replace
restore

* Unit price old
preserve
	keep if sec_price_unit_outlier == 1 & sec_price_unit != 0
			
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate equip_name sec_price_unit sec_price_unit_av asset_appliance_3b asset_appliance_4b
	keep `keepvars'
	order `keepvars'
			
	save "${checkfolder}/${folder_date}/hfc_asset_appliance/5_unitprice_sec", replace
restore


}

}

*------------------------------------------------------------------------------*
* LABOR
*------------------------------------------------------------------------------*
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users

*=================================
*	AG WORK JOBS
*=================================
{
	cap mkdir "${checkfolder}/${folder_date}/hfc_labor_agwork"
	
	use "${checkedsurvey}", clear
	
	destring count_hh_mem_older_six, replace
	qui summ count_hh_mem_older_six 
	loc maxmem = `r(max)'
	
	forvalues i = 1/`maxmem' {
		use "${checkedsurvey}", clear
		
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
		* variables to keep (reshaping by member)
		keep `generalkeep' index_earn_unearn_`i' earn_unearn_ori_index_`i' earn_inc_ag_`i' ///
		agwork_name_`i'_? agwork_prod_`i'_? agwork_dy_`i'_? agwork_hr_`i'_? ///
		agwork_pay_sch_`i'_? agwork_pay_sch_oth_`i'_? agwork_pay_amt_`i'_? agwork_pay_amt_1_`i'_? ///
		agwork_pay_amt_add_`i'_?
		
		rename *_`i'_* *_*
		rename (index_earn_unearn_`i' earn_unearn_ori_index_`i' earn_inc_ag_`i') (index_earn_unearn earn_unearn_ori_index earn_inc_ag)
		
		
		reshape long ///
		agwork_name_ agwork_prod_ agwork_dy_ agwork_hr_ ///
		agwork_pay_sch_ agwork_pay_sch_oth_ agwork_pay_amt_ agwork_pay_amt_1_ ///
		agwork_pay_amt_add_, i(caseid) j(repeatindex)
						
		rename *_ *
		
		gen repeatmemindex = `i'
		
		tempfile agwork_`i'
		save `agwork_`i'', replace
	
	}

** Get info for all members
	clear
	forvalues i = 1/`maxmem' {
		append using `agwork_`i''
	}
	
	keep if earn_inc_ag == 1 // only keep those who have ag work
	keep if !mi(agwork_name)

** Some labeling
	gen paysched = ""
		replace paysched = "Daily" if agwork_pay_sch == 1
		replace paysched = "Weekly" if agwork_pay_sch == 2
		replace paysched = "Every other week" if agwork_pay_sch == 3
		replace paysched = "Monthly" if agwork_pay_sch == 4
		replace paysched = "Every other month" if agwork_pay_sch == 5
		replace paysched = "Quarterly, or 4 times per year" if agwork_pay_sch == 6
		replace paysched = "Irregular, such as piece work or comission based work" if agwork_pay_sch == 7
		replace paysched = "Once a year" if agwork_pay_sch == 8
		replace paysched = "One time only" if agwork_pay_sch == 9
		replace paysched = "Twice per month" if agwork_pay_sch == 10
		replace paysched = "Two times per year" if agwork_pay_sch == 11
		replace paysched = agwork_pay_sch_oth if agwork_pay_sch == -666
		
* Standardizing
	gen inc_monthly = .
		replace inc_monthly = agwork_pay_amt * agwork_dy * 4 if agwork_pay_sch == 1 & !mi(agwork_pay_amt) 		// daily
		replace inc_monthly = agwork_pay_amt_1 * agwork_dy * 4 if agwork_pay_sch == 1 & !mi(agwork_pay_amt_1)	// daily
		replace inc_monthly = agwork_pay_amt * 4 if agwork_pay_sch == 2 & !mi(agwork_pay_amt) 		// weekly
		replace inc_monthly = agwork_pay_amt_1 * 4 if agwork_pay_sch == 2 & !mi(agwork_pay_amt_1)	// weekly
		replace inc_monthly = agwork_pay_amt * 2 if agwork_pay_sch == 3 & !mi(agwork_pay_amt) 		// every other week
		replace inc_monthly = agwork_pay_amt_1 * 2 if agwork_pay_sch == 3 & !mi(agwork_pay_amt_1)	// every other week
		replace inc_monthly = agwork_pay_amt if agwork_pay_sch == 4 & !mi(agwork_pay_amt) 		// monthly
		replace inc_monthly = agwork_pay_amt_1 if agwork_pay_sch == 4 & !mi(agwork_pay_amt_1)	// monthly
		replace inc_monthly = agwork_pay_amt / 2 if agwork_pay_sch == 5 & !mi(agwork_pay_amt) 		// every other month
		replace inc_monthly = agwork_pay_amt_1 / 2 if agwork_pay_sch == 5 & !mi(agwork_pay_amt_1)	// every other month
		replace inc_monthly = agwork_pay_amt * 4 / 12 if agwork_pay_sch == 6 & !mi(agwork_pay_amt) 	// quarterly
		replace inc_monthly = agwork_pay_amt_1 * 4 / 12 if agwork_pay_sch == 6 & !mi(agwork_pay_amt_1)	// quarterly
		replace inc_monthly = agwork_pay_amt / 12 if agwork_pay_sch == 8 & !mi(agwork_pay_amt) 	// yearly
		replace inc_monthly = agwork_pay_amt_1 / 12 if agwork_pay_sch == 8 & !mi(agwork_pay_amt_1)	// yearly
		replace inc_monthly = agwork_pay_amt * 2 if agwork_pay_sch == 10 & !mi(agwork_pay_amt) 	// twice per month
		replace inc_monthly = agwork_pay_amt_1 * 2 if agwork_pay_sch == 10 & !mi(agwork_pay_amt_1)	// twice per month
		replace inc_monthly = agwork_pay_amt / 6 if agwork_pay_sch == 11 & !mi(agwork_pay_amt) 	// twice per year
		replace inc_monthly = agwork_pay_amt_1 / 6 if agwork_pay_sch == 11 & !mi(agwork_pay_amt_1)	// twice per year
			
	gen inc_irreg = .
		replace inc_irreg = agwork_pay_amt if inlist(agwork_pay_sch,7,9,-666) & !mi(agwork_pay_amt)
		replace inc_irreg = agwork_pay_amt_1 if inlist(agwork_pay_sch,7,9,-666) & !mi(agwork_pay_amt_1)
		
		
	
* finding correct variable
	loc varopts agwork_name agwork_pay_amt agwork_pay_amt_1 agwork_pay_amt_add
		
	foreach v of local varopts {
		gen long_`v' = "`v'" + "_" + string(repeatmemindex) + "_" + string(repeatindex)
		
	}
	

* finding outliers
// Deciding to flag based on thresholds as opposed to outliers
	gen flag = 1 if (inc_monthly >= 50000 | inc_monthly <= 1200) & !mi(inc_monthly)
	replace flag = 1 if (inc_irreg >= 45000 | inc_irreg <= 100 ) & !mi(inc_irreg)

	levelsof paysched, loc(payoptions)
	
	*loc outliervars agwork_pay_amt agwork_pay_amt_1 agwork_pay_amt_add
	loc outliervars agwork_pay_amt_add
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different crops
		foreach opt of local payoptions {
	
			qui summ `var' if paysched == "`opt'", d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if paysched == "`opt'"
				replace `var'_av = r(mean) if paysched == "`opt'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & paysched == "`opt'" & !mi(`var')
			} // close outlier identification
		} // close amt loop
	} // close outlier variable loop

	*replace agwork_pay_amt_outlier = 1 if inrange(agwork_pay_amt,0,10)
	*replace agwork_pay_amt_1_outlier = 1 if inrange(agwork_pay_amt_1,0,10)
	replace agwork_pay_amt_add_outlier = 1 if inrange(agwork_pay_amt_add,1,10)


** TABLES	
	
* AG-WORK PAY AMT
preserve
	*keep if agwork_pay_amt_outlier == 1
	keep if flag == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate long_agwork_pay_amt agwork_name paysched agwork_pay_amt agwork_pay_amt_1 agwork_dy agwork_hr inc_monthly
	
	keep `keepvars'
	order `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_labor_agwork/1_payamt", replace
restore		
	
* AG-WORK PAY AMT COMMISSION
preserve
	keep if agwork_pay_amt_add_outlier == 1
		
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate long_agwork_pay_amt agwork_name paysched agwork_pay_amt_add agwork_pay_amt_add_av agwork_dy agwork_hr
	keep `keepvars'
	order `keepvars'
			
	save "${checkfolder}/${folder_date}/hfc_labor_agwork/2_commission", replace
restore	
	
}
	

*=================================
*	NON-AG WORK JOBS
*=================================
{
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users
	use "${checkedsurvey}", clear
	
	destring count_hh_mem_older_six, replace
	qui summ count_hh_mem_older_six 
	loc maxmem = `r(max)'
	
	forvalues i = 1/`maxmem' {
		use "${checkedsurvey}", clear
		
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
		* variables to keep (reshaping by member)
		keep `generalkeep' index_earn_unearn_`i' earn_unearn_ori_index_`i' earn_inc_nonag_`i' ///
		nonagwork_name_`i'_? nonagwork_dy_`i'_? nonagwork_hr_`i'_? ///
		nonagwork_pay_sch_`i'_? nonagwork_pay_sch_oth_`i'_? nonagwork_pay_amt_`i'_? nonagwork_pay_amt_1_`i'_? ///
		nonagwork_pay_add_`i'_?
		
		rename *_`i'_* *_*
		rename (index_earn_unearn_`i' earn_unearn_ori_index_`i' earn_inc_nonag_`i') (index_earn_unearn earn_unearn_ori_index earn_inc_nonag)
		
		
		reshape long ///
		nonagwork_name_ nonagwork_dy_ nonagwork_hr_ ///
		nonagwork_pay_sch_ nonagwork_pay_sch_oth_ nonagwork_pay_amt_ nonagwork_pay_amt_1_ ///
		nonagwork_pay_add_, i(caseid) j(repeatindex)
						
		rename *_ *
		
		gen repeatmemindex = `i'
		
		tempfile nonagwork_`i'
		save `nonagwork_`i'', replace
	
	}

** Get info for all members
	clear
	forvalues i = 1/`maxmem' {
		append using `nonagwork_`i''
	}
	
	keep if earn_inc_nonag == 1 // only keep those who have nonag work
	keep if !mi(nonagwork_name)

** Some labeling
	gen paysched = ""
		replace paysched = "Daily" if nonagwork_pay_sch == 1
		replace paysched = "Weekly" if nonagwork_pay_sch == 2
		replace paysched = "Every other week" if nonagwork_pay_sch == 3
		replace paysched = "Monthly" if nonagwork_pay_sch == 4
		replace paysched = "Every other month" if nonagwork_pay_sch == 5
		replace paysched = "Quarterly, or 4 times per year" if nonagwork_pay_sch == 6
		replace paysched = "Irregular, such as piece work or comission based work" if nonagwork_pay_sch == 7
		replace paysched = "Once a year" if nonagwork_pay_sch == 8
		replace paysched = "One time only" if nonagwork_pay_sch == 9
		replace paysched = "Twice per month" if nonagwork_pay_sch == 10
		replace paysched = "Two times per year" if nonagwork_pay_sch == 11
		replace paysched = nonagwork_pay_sch_oth if nonagwork_pay_sch == -666
		
* Standardizing
	gen inc_monthly = .
		replace inc_monthly = nonagwork_pay_amt * nonagwork_dy * 4 if nonagwork_pay_sch == 1 & !mi(nonagwork_pay_amt) 		// daily
		replace inc_monthly = nonagwork_pay_amt_1 * nonagwork_dy * 4 if nonagwork_pay_sch == 1 & !mi(nonagwork_pay_amt_1)	// daily
		replace inc_monthly = nonagwork_pay_amt * 4 if nonagwork_pay_sch == 2 & !mi(nonagwork_pay_amt) 		// weekly
		replace inc_monthly = nonagwork_pay_amt_1 * 4 if nonagwork_pay_sch == 2 & !mi(nonagwork_pay_amt_1)	// weekly
		replace inc_monthly = nonagwork_pay_amt * 2 if nonagwork_pay_sch == 3 & !mi(nonagwork_pay_amt) 		// every other week
		replace inc_monthly = nonagwork_pay_amt_1 * 2 if nonagwork_pay_sch == 3 & !mi(nonagwork_pay_amt_1)	// every other week
		replace inc_monthly = nonagwork_pay_amt if nonagwork_pay_sch == 4 & !mi(nonagwork_pay_amt) 		// monthly
		replace inc_monthly = nonagwork_pay_amt_1 if nonagwork_pay_sch == 4 & !mi(nonagwork_pay_amt_1)	// monthly
		replace inc_monthly = nonagwork_pay_amt / 2 if nonagwork_pay_sch == 5 & !mi(nonagwork_pay_amt) 		// every other month
		replace inc_monthly = nonagwork_pay_amt_1 / 2 if nonagwork_pay_sch == 5 & !mi(nonagwork_pay_amt_1)	// every other month
		replace inc_monthly = nonagwork_pay_amt * 4 / 12 if nonagwork_pay_sch == 6 & !mi(nonagwork_pay_amt) 	// quarterly
		replace inc_monthly = nonagwork_pay_amt_1 * 4 / 12 if nonagwork_pay_sch == 6 & !mi(nonagwork_pay_amt_1)	// quarterly
		replace inc_monthly = nonagwork_pay_amt / 12 if nonagwork_pay_sch == 8 & !mi(nonagwork_pay_amt) 	// yearly
		replace inc_monthly = nonagwork_pay_amt_1 / 12 if nonagwork_pay_sch == 8 & !mi(nonagwork_pay_amt_1)	// yearly
		replace inc_monthly = nonagwork_pay_amt * 2 if nonagwork_pay_sch == 10 & !mi(nonagwork_pay_amt) 	// twice per month
		replace inc_monthly = nonagwork_pay_amt_1 * 2 if nonagwork_pay_sch == 10 & !mi(nonagwork_pay_amt_1)	// twice per month
		replace inc_monthly = nonagwork_pay_amt / 6 if nonagwork_pay_sch == 11 & !mi(nonagwork_pay_amt) 	// twice per year
		replace inc_monthly = nonagwork_pay_amt_1 / 6 if nonagwork_pay_sch == 11 & !mi(nonagwork_pay_amt_1)	// twice per year
			
	gen inc_irreg = .
		replace inc_irreg = nonagwork_pay_amt if inlist(nonagwork_pay_sch,7,9,-666) & !mi(nonagwork_pay_amt)
		replace inc_irreg = nonagwork_pay_amt_1 if inlist(nonagwork_pay_sch,7,9,-666) & !mi(nonagwork_pay_amt_1)
		
* finding correct variable
	loc varopts nonagwork_name nonagwork_pay_sch nonagwork_pay_sch_oth ///
		nonagwork_pay_amt nonagwork_pay_amt_1 nonagwork_pay_add
		
	foreach v of local varopts {
		gen long_`v' = "`v'" + "_" + string(repeatmemindex) + "_" + string(repeatindex)	
	}
	
* finding outliers
// Deciding to flag based on thresholds as opposed to outliers
	gen flag = 1 if (inc_monthly >= 50000 | inc_monthly <= 1200) & !mi(inc_monthly)
	replace flag = 1 if (inc_irreg >= 45000 | inc_irreg <= 100 ) & !mi(inc_irreg)


	levelsof paysched, loc(payoptions)
	
	*loc outliervars nonagwork_pay_amt nonagwork_pay_amt_1 nonagwork_pay_add
	loc outliervars nonagwork_pay_add
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different crops
		foreach opt of local payoptions {
	
			qui summ `var' if paysched == "`opt'", d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if paysched == "`opt'"
				replace `var'_av = r(mean) if paysched == "`opt'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & paysched == "`opt'" & !mi(`var')
			} // close outlier identification
		} // close amt loop
	} // close outlier variable loop

	*replace nonagwork_pay_amt_outlier = 1 if inrange(nonagwork_pay_amt,0,10)
	*replace nonagwork_pay_amt_1_outlier = 1 if inrange(nonagwork_pay_amt_1,0,10)
	replace nonagwork_pay_add_outlier = 1 if inrange(nonagwork_pay_add,1,10)



** TABLES

* AG-WORK PAY AMT
	cap mkdir "${checkfolder}/${folder_date}/hfc_labor_nonagwork"

preserve
	*keep if nonagwork_pay_amt_outlier == 1
	keep if flag == 1
	
	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate long_nonagwork_pay_amt nonagwork_name paysched nonagwork_pay_amt nonagwork_pay_amt_1 nonagwork_dy nonagwork_hr inc_monthly
	keep `keepvars'
	order `keepvars'
		
	save "${checkfolder}/${folder_date}/hfc_labor_nonagwork/1_payamt", replace
restore		
	
* AG-WORK PAY AMT COMMISSION
preserve
	keep if nonagwork_pay_add_outlier == 1

	loc keepvars caseid fo fc_name pull_municipal_city pull_barangay startdate subdate long_nonagwork_pay_amt nonagwork_name paysched nonagwork_pay_add nonagwork_pay_add_av nonagwork_dy nonagwork_hr
	keep `keepvars'
	order `keepvars'

	save "${checkfolder}/${folder_date}/hfc_labor_nonagwork/2_commission", replace

restore	
	
}


*------------------------------------------------------------------------------*
* EDUCATION EXPENSES
*------------------------------------------------------------------------------*
{
	cap mkdir "${checkfolder}/${folder_date}/hfc_educ_expense"
	
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users
	use "${checkedsurvey}", clear
	
* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 	
	
** Need to create values e.g., grade level and age
	destring calc_edu_current, replace
	qui summ calc_edu_current
	loc expmax = `r(max)'
	
	qui summ count_res_members
	loc resmax = `r(max)'
	
	qui summ count_nonres_members
	loc nonresmax = `r(max)'
	
	destring educ_all_members_age_* nr_age_calc_* nr_edu_completed_*, replace
	
	forvalues i = 1/`expmax' {
		qui gen edu_exp_age_`i' = ., b(edu_exp_1_`i')
		qui gen edu_exp_grd_`i' = ., a(edu_exp_age_`i')
		
		* fill in the resident member details
		forvalues j = 1/`resmax' {
			qui replace edu_exp_age_`i' = educ_all_members_age_`j' if temp_edu_current_name_`i' == educ_all_members_name_`j' & mi(edu_exp_age_`i') & !mi(educ_all_members_age_`j')
			qui replace edu_exp_grd_`i' = edu_grade_`j' if temp_edu_current_name_`i' == educ_all_members_name_`j' & mi(edu_exp_grd_`i') & !mi(edu_grade_`j')
		}
		
		* fill in the non-resident details
		forvalues k = 1/`nonresmax' {
			qui replace edu_exp_age_`i' = nr_age_calc_`k'  if temp_edu_current_name_`i' == temp_nr_comp_name_`k'  & mi(edu_exp_age_`i') & !mi(nr_age_calc_`k')
			qui replace edu_exp_grd_`i' = nr_edu_completed_`k' if temp_edu_current_name_`i' == temp_nr_comp_name_`k' & mi(edu_exp_grd_`i') & !mi(nr_edu_completed_`k')
		}
	}
	
	
** reshape module
	keep `generalkeep' edu_exp_age_* edu_exp_grd_* edu_exp_1_* edu_exp_2_* edu_exp_3_* edu_exp_4_* edu_exp_5_* edu_exp_6_* edu_exp_7_* edu_exp_8_* edu_exp_oth_* edu_scholarship_* edu_assist_* index_edu_current_name_*
	
	reshape long 	edu_exp_age_ edu_exp_grd_ edu_exp_1_ edu_exp_2_ edu_exp_3_ /// 
					edu_exp_4_ edu_exp_5_ edu_exp_6_ edu_exp_7_ edu_exp_8_ /// 
					edu_exp_oth_ edu_scholarship_ edu_assist_ index_edu_current_name_, i(caseid) j(index2)
					
	rename *_ *
	drop if mi(index_edu_current_name)
	
* grade level string
	label val edu_exp_grd edu_grade_1 // label the numerical grades
	decode edu_exp_grd, gen(edu_exp_grd_str) // make this a string variable
	
* create categories to compare general levels to one another for outliers
	gen group_edu_level = .
		replace group_edu_level = 1 if inrange(edu_exp_grd,1,2) // daycare and kinder
		replace group_edu_level = 2 if inrange(edu_exp_grd,3,11) // 1st-8th grade
		replace group_edu_level = 3 if inrange(edu_exp_grd,12,15) // 9th-12th grade
		replace group_edu_level = 4 if inrange(edu_exp_grd,16,22) // vocational, college, more
		replace group_edu_level = 5 if edu_exp_grd == -666 | mi(edu_exp_grd) // other, most likely ALS	
	
	label define group_edu_level /// 
			1 "daycare and kinder" ///
			2 "1st-8th grade" ///
			3 "9th-12th grade" ///
			4 "vocational, college, more" ///
			5 "other, most likely ALS"
			
	label val group_edu_level group_edu_level
	
**  outliers
	local expensevars 	edu_exp_1 edu_exp_2 edu_exp_3 edu_exp_4 edu_exp_5 /// 
						edu_exp_6 edu_exp_7 edu_exp_8 edu_exp_oth /// 
						edu_scholarship edu_assist
	

	foreach var of local expensevars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		forvalues i = 1/5 { // for the different grade level categories
			qui summ `var' if group_edu_level == `i', d // will allow for entries of "0"
			if `r(N)' > 1 {
				replace `var'_iqr = `r(p75)'-`r(p25)' if group_edu_level == `i'
				replace `var'_av = `r(mean)' if group_edu_level == `i'
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5* `var'_iqr)) | (`var' < `r(p25)' - (1.5* `var'_iqr))) & !mi(`var'_iqr) & group_edu_level == `i' & !mi(`var')
			}
			
		}
	}
	

** Educational expenses
	foreach var of local expensevars {
		preserve
			keep `generalkeep' edu_exp_age edu_exp_grd_str `var' `var'_outlier `var'_av index2
			keep if `var'_outlier == 1 | inrange(`var',1,10)
			
			drop `var'_outlier
			
			gen variable = "`var'" + "_" + string(index2)
			
			gen expense = ""
			replace expense = "School fees incl PTA" 	if "`var'" == "edu_exp_1"
			replace expense = "Boarding & lodging" 		if "`var'" == "edu_exp_2"
			replace expense = "School uniform" 			if "`var'" == "edu_exp_3"
			replace expense = "Books & supplies" 		if "`var'" == "edu_exp_4"
			replace expense = "Books & learning materials not required" 								if "`var'" == "edu_exp_5"
			replace expense = "Non-academic extracurricular activities (sports, youth clubs, etc.)" 	if "`var'" == "edu_exp_6"
			replace expense = "Tutoring, coaching, academic support" 									if "`var'" == "edu_exp_7"
			replace expense = "Allowance & transport to/from school" 									if "`var'" == "edu_exp_8"
			replace expense = "Other educational expenses" 												if "`var'" == "edu_exp_oth"
			replace expense = "Scholarship assistance" 													if "`var'" == "edu_scholarship"
			replace expense = "Assistance to attend school from relatives or friends" 					if "`var'" == "edu_assist"
			
			rename `var' 		expense_amt
			rename `var'_av 	expense_av
			
			tempfile `var'_temp
			save ``var'_temp'
		restore
	}
	
	
	clear 
	foreach var of local expensevars {
		append using ``var'_temp'
	}
	
	label var expense_amt "Expense amount the FO entered"
	label var expense_av "Average expense across all entries"
	
	save "${checkfolder}/${folder_date}/hfc_educ_expense/educ_expenses", replace
}
	
*------------------------------------------------------------------------------*
* Health Items
*------------------------------------------------------------------------------*
{
	cap mkdir "${checkfolder}/${folder_date}/hfc_health"
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users

	use "${checkedsurvey}", clear
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
	
// 	WAIT TIMES
preserve
	keep `generalkeep' health_access_3
	keep if health_access_3 >= 240 & !mi(health_access_3)
	
	
	rename health_access_3 health_access
	gen variable = "health_access_3"
	
	gen wait_for = "BHW"
	
	tempfile bhw_wait
	save `bhw_wait'
restore 
preserve
	keep `generalkeep' health_access_6
	keep if health_access_6 >= 240 & !mi(health_access_6)
	gen wait_for = "doctor or nurse"
	
	rename health_access_6 health_access
	gen variable = "health_access_6"
	
	tempfile doc_wait
	save `doc_wait'
restore 
preserve
	keep `generalkeep' health_access_11
	keep if health_access_11 >= 240 & !mi(health_access_11)
	gen wait_for = "dentist"
	
	rename health_access_11 health_access
	gen variable = "health_access_11"
	
	tempfile dentist_wait
	save `dentist_wait'
restore 
	
	clear 
	use `bhw_wait'
	append using `doc_wait'
	append using `dentist_wait'
	
	
	save "${checkfolder}/${folder_date}/hfc_health/wait_times", replace

// 	HEALTH EXPENSES
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users
	use "${checkedsurvey}", clear

		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
			
	forvalues i = 1/12 {

		gen health_exp`i'_outlier = 0
		gen health_exp`i'_iqr = .
		gen health_exp`i'_av = .
		
		qui summ health_exp`i' if health_exp`i'!= 0, d
			if `r(N)' > 1 {
					
				replace health_exp`i'_iqr = r(p75)-r(p25)
				replace health_exp`i'_av = r(mean)
				replace health_exp`i'_outlier = 1 if ((health_exp`i' > `r(p75)' + (1.5*health_exp`i'_iqr)) | (health_exp`i' < `r(p25)' - (1.5*health_exp`i'_iqr))) & !mi(health_exp`i')
			}
	}
	
	forvalues i = 1/12{

		preserve
			keep `generalkeep'  health_exp`i' health_exp`i'_outlier health_exp`i'_av count_res_members
			keep if health_exp`i'_outlier == 1 | inrange(health_exp`i',1,10)
			drop health_exp`i'_outlier
			
			gen variable = "health_exp`i'"
			rename health_exp`i' 		expense_amt
			rename health_exp`i'_av		expense_mean
			
			gen expense = ""
			replace expense = "medication over the counter" if `i' == 1
			replace expense = "dietary supplements" if `i' == 2
			replace expense = "medical equipment (e.g., glasses, crutches, BP app)" if `i' == 3
			replace expense = "other medical products (e.g., bandages, syringes, knee support, pregnancy tests)" if `i' == 4
			replace expense = "medical insurance" if `i' == 5
			replace expense = "outpatient: ambulance" if `i' == 6
			replace expense = "outpatient: hospital/clinic fees" if `i' == 7
			replace expense = "outpatient: medical imaging and other tests (x-rays, urine tests, blood tests)" if `i' == 8
			replace expense = "outpatient: traditional and/or herbal doctor fees" if `i' == 9
			replace expense = "inpatient: ambulance" if `i' == 10
			replace expense = "inpatient: hospital/clinic fees" if `i' == 11
			replace expense = "inpatient: medical imaging and other tests (x-rays, urine tests, blood tests)" if `i' == 12
				
			tempfile health_exp`i'_tempfile
			save `health_exp`i'_tempfile'
		restore
		
	}

	clear
	use `health_exp1_tempfile'
	
	forvalues i = 2/12 {
		append using `health_exp`i'_tempfile'
	}

	label var expense_amt "Expense amount the FO entered"
	label var expense_mean "Average expense across all entries"
	
	save "${checkfolder}/${folder_date}/hfc_health/health_expenses", replace
}
	
	
	
	
*------------------------------------------------------------------------------*
* Credit module
*------------------------------------------------------------------------------*
{
	cap mkdir "${checkfolder}/${folder_date}/hfc_credit"
	loc generalkeep	startdate subdate  caseid fo fo_id fo_name sfo_id sfo_name fc_id fc_name pull_barangay pull_municipal_city pull_province users

	use "${checkedsurvey}", clear
	
	
		* dropping false launch cases
		drop if ///
			inlist(pull_brgy_prefix, "H004457", "H004067", "H004582", "H006382", "H006352") | ///
			inlist(pull_brgy_prefix, "H006303", "H030037", "H030238", "H030754", "H030256") | ///
			inlist(pull_brgy_prefix, "H030682", "H030628", "H030773", "H030436", "H030427") | ///
			inlist(pull_brgy_prefix, "H030508", "H030140", "H019526", "H019017", "H019126") 
			
		* keep relevant
		keep `generalkeep' index_credit_? temp_credit_name_? credit_amt_? credit_fee_? /// 
			 credit_reason_? credit_reason_oth_? credit_pay_sch_? credit_pay_sch_oth_? ///
			 credit_pay_amt_? credit_outstnd_?
			 
		
		reshape long index_credit_ temp_credit_name_ credit_amt_ credit_fee_ /// 
			 credit_reason_ credit_reason_oth_ credit_pay_sch_ credit_pay_sch_oth_ ///
			 credit_pay_amt_ credit_outstnd_ , i(caseid) j(index2)
	
		rename *_ *
	
		drop if mi(index_credit)
	
* TRANSLATIONS
* --- 	TRANSLATIONS --- *
	* Change names to English
	loc languages Hiligaynon Kinaraya Bisaya Akeanon Tagalog
	
	foreach lang of local languages {
		gen label`lang' =  temp_credit_name // item to merge with
	}
	
	
	* Getting translations from choices sheet
preserve
	import excel "${tempfolder}/choices", clear first
	keep if list_name == "credit"
	drop K-AG
	
	foreach lang of local languages {
		gen `lang'_eng = label
	}
	
	tempfile translations
	save `translations'
restore

	* Testing the merge on each language
	foreach lang of local languages {
		merge m:1 label`lang' using `translations', gen(m_`lang') keepus(`lang'_eng)
		keep if inlist(m_`lang',1,3)
	}

	* Create single English translation
	gen credit_name = "", a(temp_credit_name)
	foreach lang of local languages {
		replace credit_name = `lang'_eng if mi(credit_name) & m_`lang' == 3
		drop `lang'_eng m_`lang' label`lang' // drop the temp vars
	}
	 
		replace credit_name = temp_credit_name if mi(credit_name)
		drop temp_credit_name
		
		replace credit_name = "Bank or other financial institution, eg: a microfinance institution, home credit" if credit_name == "Bangko ukon iban pa nga pinansyal nga institusyon halimbawa microfinance nga institusyon."
	
* creating reason for credit
	split credit_reason
	loc num = `r(k_new)'
	
	forvalues i = 1/`num' {
		replace credit_reason`i' = "Purchase of house or land for dwelling" if credit_reason`i' == "1"
		replace credit_reason`i' = "Household consumption needs"  if credit_reason`i' == "2"
		replace credit_reason`i' = "Housing rent" if credit_reason`i' == "3"
		replace credit_reason`i' = "Maintenance or repairs of dwellings" if credit_reason`i' == "4"
		replace credit_reason`i' = "Payback existing debt" if credit_reason`i' == "5"
		replace credit_reason`i' = "Ceremonies (wedding, funderal)" if credit_reason`i' == "6"
		replace credit_reason`i' = "Purchase of assets for own use" if credit_reason`i' == "7"
		replace credit_reason`i' = "Education" if credit_reason`i' == "8"
		replace credit_reason`i' = "Healthcare" if credit_reason`i' == "9"
		replace credit_reason`i' = "Shocks" if credit_reason`i' == "10"
		replace credit_reason`i' = credit_reason_oth if credit_reason`i' == "-666"	
	}
	
	gen reason_list = credit_reason1 if !mi(credit_reason1), a(credit_reason)
	forvalues i = 2/`num' {
		replace reason_list = reason_list + ", " + credit_reason`i' if !mi(credit_reason`i')
	}	
		
	drop credit_reason1-credit_reason`num'
	
	gen credit_fee_perc = round(credit_fee/credit_amt * 100)

** Outliers **
	
* Find outliers
	levelsof(credit_name), loc(items)
	
	loc outliervars  credit_amt credit_fee_perc
	
	* vars for outliers
	foreach var of local outliervars {
		
		gen `var'_outlier = 0
		gen `var'_iqr = .
		gen `var'_av = .
		
		
		* different appliances
		foreach itm of local items {
	
			qui summ `var' if credit_name == "`itm'" & `var' != 0, d
			if `r(N)' > 1 {
					
				replace `var'_iqr = r(p75)-r(p25) if credit_name == "`itm'"
				replace `var'_av = r(mean) if credit_name == "`itm'"
				replace `var'_outlier = 1 if ((`var' > `r(p75)' + (1.5*`var'_iqr)) | (`var' < `r(p25)' - (1.5*`var'_iqr))) & credit_name == "`itm'" & !mi(`var')
			} // close outlier identification
		} // close appliance loop
	} // close outlier variable loop

	replace credit_amt_outlier = 1 if credit_amt <= 200 & !mi(credit_amt) // want to flag things that look very low

*** Saving the dataset checks
	
** Outlier in credit amts
	
preserve
	keep if credit_amt_outlier == 1
	
	loc keepvars startdate subdate caseid fo fc_name pull_barangay pull_municipal_city pull_province credit_name credit_amt reason_list credit_amt_av
	order `keepvars'
	keep `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_credit/1_credit_amt", replace
restore

** Fees (as a percentage)
preserve
	keep if credit_fee_perc_outlier == 1
	
	loc keepvars startdate subdate caseid fo fc_name pull_barangay pull_municipal_city pull_province credit_name credit_amt credit_fee credit_fee_perc reason_list
	order `keepvars'
	keep `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_credit/2_credit_fees", replace
restore

** Fees same percentage
preserve
	keep if credit_fee == credit_amt & (credit_amt != 0)
	
	loc keepvars startdate subdate caseid fo fc_name pull_barangay pull_municipal_city pull_province credit_name credit_amt credit_fee credit_fee_perc reason_list
	order `keepvars'
	keep `keepvars'
	
	save "${checkfolder}/${folder_date}/hfc_credit/3_same_credit_fees", replace
restore
}


