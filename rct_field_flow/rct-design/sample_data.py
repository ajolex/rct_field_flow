"""
Sample data module for RCT Design Wizard
Provides realistic example data for testing and demonstration
"""

def get_education_sample_data() -> dict:
    """
    Generate a comprehensive sample concept note for an education intervention
    Based on a realistic Malawi literacy program
    
    Returns:
        Dictionary with all 15 sections populated with example data
    """
    return {
        "meta": {
            "title": "Improving Early Grade Literacy in Rural Malawi",
            "pis": ["Dr. Sarah Johnson", "Prof. James Mzamane"],
            "organization": "RCT Learning Lab",
            "country": "Malawi",
            "funder": "Global Education Fund",
            "project_stage": "Concept note",
            "study_start_date": "2024-06-01",
            "study_end_date": "2026-12-31"
        },
        
        "problem_policy_relevance": {
            "problem_statement": "In rural Malawi, only 15% of Grade 3 students can read a simple story with comprehension, far below the 80% target set by the Ministry of Education. Teachers lack access to evidence-based teaching materials and training in phonics-based instruction, leading to high dropout rates particularly among girls.",
            "evidence_gap": "While global evidence shows phonics-based instruction is effective in Sub-Saharan Africa, there is limited evidence on the cost-effectiveness of deploying trained reading coaches alongside teacher professional development in resource-constrained settings like Malawi.",
            "beneficiaries": "Approximately 3,200 students (age 6-9) across 16 primary schools in Lilongwe and Mchinji districts, and 48 primary school teachers.",
            "policy_priority": "The Ministry of Education has prioritized improving literacy outcomes in early grades to meet SDG 4 targets. This study will inform district-level literacy interventions and teacher deployment strategies."
        },
        
        "theory_of_change": {
            "activities": [
                "Train 48 teachers in phonics-based reading instruction (5-day intensive workshop + monthly coaching)",
                "Deploy 8 reading coaches to provide classroom-based teacher support (1 coach per 6 schools)",
                "Provide 16 schools with reading materials in Chichewa aligned to phonics curriculum",
                "Establish monthly learning circles where teachers share best practices"
            ],
            "outputs": [
                "48 teachers complete phonics training and demonstrate competency in lesson delivery",
                "8 reading coaches trained and deployed",
                "3,200 Grade 1-3 students exposed to phonics-based reading instruction",
                "School and district leaders engaged in implementation oversight"
            ],
            "short_run_outcomes": [
                "Teachers implement phonics instruction in 80%+ of lessons observed",
                "Students participate in daily reading practice activities (30 minutes minimum)",
                "Teachers report increased confidence in teaching reading"
            ],
            "long_run_outcomes": [
                "Grade 3 reading proficiency improves from 15% to at least 45% by endline",
                "Gender gap in reading proficiency narrows (currently girls are 8pp behind boys)",
                "Student retention improves, particularly for girls"
            ],
            "assumptions": [
                "Teachers will be motivated to implement phonics instruction after training",
                "Reading coaches will have sufficient autonomy and resources to support teachers",
                "District education officials will facilitate school access and data collection",
                "School absenteeism will not exceed 20% during intervention period"
            ],
            "risks": [
                "High teacher turnover may reduce intervention exposure",
                "School closures due to political or health crises",
                "Coaches may focus on compliant schools, creating implementation heterogeneity",
                "Language barriers if coaches speak different local dialects"
            ]
        },
        
        "intervention_implementation": {
            "description": "The Malawi Early Literacy Program (MELP) combines teacher professional development through intensive phonics training with sustained classroom coaching. Reading coaches visit schools bi-weekly to model lessons, observe teachers, and provide feedback using a structured coaching protocol.",
            "components": [
                "Component 1: Initial Teacher Training - 5-day residential phonics workshop covering letter-sound correspondence, blending, decoding strategies, and classroom management",
                "Component 2: Reading Coach Support - Bi-weekly in-classroom coaching visits with structured observation and feedback using the Early Literacy Coaching Protocol",
                "Component 3: Learning Materials - Grade-level appropriate readers and flashcards in Chichewa using phonics-aligned content",
                "Component 4: School Leadership Engagement - Quarterly meetings with head teachers and district officials to monitor implementation"
            ],
            "delivery_channels": [
                "Face-to-face training through district education offices",
                "In-classroom coaching during school hours",
                "WhatsApp community for coaches and teachers to share resources (weekly)"
            ],
            "frequency_intensity": "Teachers receive 5 days of initial training, then bi-weekly classroom coaching visits (2 hours each) for 24 months. Reading coaches maintain caseloads of 4-6 schools each.",
            "eligibility_criteria": "Schools: 6+ students per grade in Grades 1-3; Rural locations; <1 hour travel time from district center. Teachers: Primary school teacher in participating school; Teaching Grades 1-3; Committed to 2-year participation.",
            "implementers": [
                "Malawi Ministry of Education (official partnership and oversight)",
                "NGO: Save the Future (program coordination and training delivery)",
                "RCT Learning Lab (research design and evaluation)"
            ],
            "operational_constraints": [
                "Limited transport budget for reading coach movements (some schools accessible only by foot during rainy season)",
                "Power outages affect data collection using tablets in some locations",
                "Teacher salary delays may affect motivation for participation in learning circles"
            ]
        },
        
        "study_population_sampling": {
            "unit_of_randomization": "School (16 schools total: 8 treatment, 8 comparison)",
            "expected_total_n": 3200,
            "breakdown_by_arm": {
                "treatment": 1600,
                "comparison": 1600
            },
            "sampling_frame_source": "Ministry of Education's School Census 2023 for rural primary schools in Lilongwe and Mchinji districts",
            "inclusion_exclusion": "Inclusion: Primary schools with 6+ students per grade in Grades 1-3; Rural location; <1 hour travel time. Exclusion: Schools with pre-existing literacy programs; Schools serving refugee populations; Special education schools.",
            "coverage_limitations": "Results generalizable to rural Malawi in similar literacy contexts but may not apply to urban settings or countries with different education infrastructure. Teachers self-selected for study participation may be more motivated than average."
        },
        
        "outcomes_measurement": {
            "primary_outcomes": [
                "Grade 3 reading proficiency (dichotomous: can/cannot read simple story with 70%+ comprehension)",
                "Oral reading fluency - words read correctly per minute (ORPM) in Grade 2-3 students"
            ],
            "secondary_outcomes": [
                "Teacher phonics content knowledge (70-item multiple choice assessment)",
                "Teacher confidence in reading instruction (5-item Likert scale)",
                "Student attendance in intervention schools",
                "Gender gap in reading proficiency (percentage point difference F-M)",
                "Cost per student achieving reading proficiency"
            ],
            "measurement_timing": {
                "baseline": "April 2024 (before training begins)",
                "midline": "June 2025 (12 months into intervention)",
                "endline": "May 2026 (24 months, before school year ends)"
            },
            "instruments": [
                "Early Grade Reading Assessment (EGRA) adapted for Chichewa - 15 minutes per student",
                "Teacher Knowledge Assessment (TKA) - 30 minutes per teacher",
                "Teacher Survey on confidence and practices - 20 minutes per teacher",
                "Classroom observation using IECD protocol - 2 hours per teacher"
            ]
        },
        
        "randomization_design": {
            "numeric": {
                "design_type": "Simple randomization",
                "arms": 2,
                "treatment_arm_label": "Phonics training + coaching",
                "comparison_arm_label": "Comparison (standard practice)",
                "seed": 42,
                "strata": ["District (Lilongwe/Mchinji)", "School size (large/small)"]
            },
            "narrative": {
                "rationale": "Simple randomization with stratification balances observed school characteristics while maintaining treatment integrity. Two-armed design allows clean counterfactual comparison of intervention impact.",
                "implementation_steps": "1) List all eligible schools (n=16); 2) Stratify by district and size; 3) Within strata, randomly assign 50% to treatment; 4) Randomization conducted by independent statistician; 5) Public announcement of assignments at stakeholder meeting",
                "concealment": "Concealment not possible (schools know assignment), but outcome assessors will be blinded to treatment assignment using data collection protocols blind to school lists."
            }
        },
        
        "power_sample_size": {
            "numeric": {
                "n_per_arm": 1600,
                "total_n": 3200,
                "mde": 0.15,
                "alpha": 0.05,
                "power": 0.80,
                "intracluster_correlation": 0.15,
                "attrition_assumed": 0.15
            },
            "narrative": {
                "rationale": "With 1,600 students per arm (16 schools, ~200 per school including Grades 1-3), we achieve 80% power to detect a 15-percentage-point effect on reading proficiency (from 15% to 30%) at α=0.05, accounting for school-level clustering (ICC=0.15) and 15% attrition.",
                "power_calculation_details": "MDE of 15pp chosen as educationally meaningful based on similar interventions in East Africa. Baseline reading proficiency (15%) from Ministry data; ICC from similar settings."
            }
        },
        
        "data_collection_plan": {
            "mode": "Quantitative: Paper-based EGRA administered by trained enumerators. Teacher assessments administered in-person at schools. Qualitative: Focus group discussions with teachers (n=4 groups) at endline.",
            "survey_schedule": "Baseline (April 2024): 3 weeks to administer EGRA to all Grade 1-3 students + teacher assessments. Midline (June 2025): 2 weeks for EGRA subset + teacher survey. Endline (May 2026): 3 weeks for full EGRA + classroom observations + focus groups.",
            "qc_protocols": "Daily: Data review by field supervisor for completeness. Weekly: Enumerator meetings to address quality issues. Monthly: Re-test 10% sample and verify consent forms. Ongoing: GPS verify school visits and photo documentation of surveys."
        },
        
        "analysis_plan": {
            "estimands": {
                "itt": "Intent-to-treat: Effect of school assignment to treatment on Grade 3 reading proficiency, including all students regardless of treatment exposure",
                "tot": "Treatment-on-the-treated: Effect of actual receipt of phonics training and coaching on Grade 3 reading proficiency"
            },
            "models": {
                "regression_spec": "OLS regression with Grade 3 reading proficiency (binary: can/cannot read) as dependent variable, treatment indicator as main explanatory variable, controlling for school-level covariates (baseline literacy rate, district, school size)",
                "controls_set": ["District fixed effects", "School baseline literacy rate", "School size (students per grade)"],
                "heterogeneity_subgroups": ["Girls vs. Boys", "Urban vs. Rural schools", "High vs. Low baseline literacy schools"],
                "multiple_testing_strategy": "Bonferroni correction with pre-specified primary outcome (Grade 3 reading proficiency)",
                "missing_data_strategy": "Multiple imputation using chained equations for missing outcome data; sensitivity analysis using bounds method",
                "mediation_exploratory_flags": ["Does improved teacher knowledge mediate the effect?", "Does classroom observation quality mediate the effect?"]
            }
        },
        
        "ethics_risks_approvals": {
            "irb_status": "IRB approval obtained from University of Malawi Research Ethics Committee (REF: UM/REC/2024/001). Community engagement conducted in all 16 schools with head teachers and district officials.",
            "consent_process": "Written parental consent (mother tongue or English) required for student participation. Teachers provide written informed consent. Community leaders briefed at inception workshop.",
            "privacy_security": "Unique student IDs used instead of names in datasets. Paper surveys stored in locked cabinets. Digital data encrypted and stored on secure servers. No names in public-facing outputs. Data destroyed after 7 years per Malawi Data Protection Act.",
            "risk_matrix": [],
            "fairness_waitlist": "Comparison schools will receive the intervention in Phase 2 if results show significant positive effects",
            "adverse_event_protocol": "Protocol for managing any adverse events during data collection defined in enumerator training. Immediate reporting to study PI and Ministry of Education."
        },
        
        "timeline_milestones": {
            "preparation_phase": {
                "duration": "2 months",
                "start_date": "2024-02-01",
                "end_date": "2024-03-31",
                "milestones": [
                    "Recruit and train 12 enumerators",
                    "Finalize school recruitment and consent",
                    "Pre-test data collection tools",
                    "Train randomization team"
                ]
            },
            "baseline_phase": {
                "duration": "1 month",
                "start_date": "2024-04-01",
                "end_date": "2024-04-30",
                "milestones": [
                    "Baseline student assessments (all 3,200)",
                    "Baseline teacher assessments (48 teachers)",
                    "Randomization announcement"
                ]
            },
            "implementation_phase": {
                "duration": "24 months",
                "start_date": "2024-05-01",
                "end_date": "2026-04-30",
                "milestones": [
                    "Month 1-2: Teacher training and coach deployment",
                    "Month 3-12: Full implementation with bi-weekly coaching",
                    "Month 13: Midline data collection",
                    "Month 14-24: Continued implementation and quarterly monitoring"
                ]
            },
            "endline_phase": {
                "duration": "1 month",
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
                "milestones": [
                    "Endline student assessments (all 3,200)",
                    "Endline classroom observations (48 teachers)",
                    "Teacher focus group discussions",
                    "Data quality checks and final survey"
                ]
            },
            "analysis_dissemination": {
                "duration": "3 months",
                "start_date": "2026-06-01",
                "end_date": "2026-08-31",
                "milestones": [
                    "Data cleaning and analysis",
                    "Preliminary findings presentation to stakeholders",
                    "Technical report writing",
                    "Policy brief development for Ministry"
                ]
            }
        },
        
        "budget_summary": {
            "categories": {
                "personnel": 95000,
                "data": 48000,
                "intervention": 107000,
                "overheads": 20000,
                "contingency": 5000
            },
            "funding_gap": 0,
            "co_financing": "Ministry of Education provides in-kind support: 48 teachers and facilities worth approximately $50,000"
        },
        
        "policy_relevance_scalability": {
            "alignment_with_national_policies": "The study directly supports the Ministry of Education's commitment to the National Literacy Strategy targeting 80% of Grade 3 students reading with comprehension. Results will inform district-level literacy interventions and the Competitive Fund for Education program.",
            "scale_pathways": "If successful, the model can be scaled to 30 additional districts reaching 2 million students by 2030. Cost-effectiveness analysis will identify options for government financing versus donor support.",
            "delivery_model_comparison": "This study tests one delivery model (external coaches + MoE teachers). Phase 2 could compare with peer-to-peer teacher training and self-study options.",
            "cost_effectiveness_narrative": "Expected cost per student achieving reading proficiency: $85 (total budget $275,000 / 3,200 students). Equivalent to 2 weeks of teacher salary per student. Comparable to other successful literacy interventions in Sub-Saharan Africa."
        },
        
        "compliance_requirements": {
            "funder_requirements": [
                "Quarterly progress reports to Global Education Fund",
                "Annual independent financial audit",
                "Results shared within 6 months of endline in open-access format"
            ],
            "research_standards": [
                "Pre-registration on RIDIE (D-3030432) and OSF",
                "Open science: Code and analysis notebooks available on GitHub",
                "Data available upon request from verified researchers (IRB approval required)"
            ],
            "local_requirements": [
                "All publications require Ministry of Education review (30-day turnaround)",
                "Beneficiary community receives findings in accessible format (infographics, radio spots)",
                "Implementation partner has intellectual property rights to coaching protocol"
            ]
        },
        
        "references": [
            "Bastian, J., & Jain, S. (2024). Early Grade Literacy in Sub-Saharan Africa: A Meta-Analysis. Journal of Development Economics, 156, 103-121.",
            "Malawi Ministry of Education. (2023). National Literacy Assessment Report. Lilongwe: Government Press.",
            "Piper, B., et al. (2018). Coaching to Improve Classroom Instruction in Early Reading in Kenya. Educational Research Review, 25, 45-67.",
            "RTI International. (2020). Early Grade Reading Assessment (EGRA) Toolkit. Washington DC: USAID.",
            "World Bank. (2023). Learning Poverty Report 2023. Washington DC: World Bank Group."
        ],
        
        "references_annexes": {
            "citations": [
                "Bastian, J., & Jain, S. (2024). Early Grade Literacy in Sub-Saharan Africa: A Meta-Analysis. Journal of Development Economics, 156, 103-121.",
                "Malawi Ministry of Education. (2023). National Literacy Assessment Report. Lilongwe: Government Press.",
                "Piper, B., et al. (2018). Coaching to Improve Classroom Instruction in Early Reading in Kenya. Educational Research Review, 25, 45-67.",
                "RTI International. (2020). Early Grade Reading Assessment (EGRA) Toolkit. Washington DC: USAID.",
                "World Bank. (2023). Learning Poverty Report 2023. Washington DC: World Bank Group."
            ],
            "instruments_list": [
                "Early Grade Reading Assessment (EGRA) - student reading fluency and comprehension",
                "Teacher Knowledge Assessment (TKA) - phonics content knowledge",
                "Teacher Confidence Likert Scale - self-efficacy for reading instruction",
                "Classroom Observation Protocol - teaching quality and literacy instruction practices"
            ],
            "randomization_code_link": "https://github.com/rct_learning_lab/malawi_literacy_rct_public",
            "survey_modules": [
                "Student reading assessment module (EGRA)",
                "Teacher knowledge and practice module",
                "School administrative module"
            ],
            "partner_mou_status": "Signed MOUs with Ministry of Education, District Education Offices (Lilongwe & Mchinji), and Save the Future NGO"
        },
        
        "compliance": {
            "prereg_plan": "Study pre-registered on Open Science Framework (OSF) and RIDIE (ID: D-3030432) before baseline data collection",
            "data_management_plan": "Data stored on encrypted servers with access restricted to study team. Student data uses unique IDs only. Raw data will be destroyed 7 years post-study per Malawi Data Protection Act.",
            "back_check_protocol": "10% of surveys back-checked monthly by supervisors. GPS coordinates recorded for all school visits. Photo documentation of active data collection.",
            "transparency_commitments": [
                "Publish results regardless of findings (positive, null, or negative)",
                "Register any deviations from pre-registered analysis plan with justification",
                "Release de-identified data to researchers within 12 months of publication"
            ]
        }
    }


def get_health_sample_data() -> dict:
    """
    Generate a sample concept note for a health intervention
    (Basic structure - can be expanded)
    """
    return {
        "meta": {
            "title": "Community Health Worker Mobile Technology for Maternal Health in Ghana",
            "pis": ["Dr. Ama Owusu", "Dr. Benjamin Osei"],
            "organization": "Global Health Institute",
            "country": "Ghana",
            "funder": "Gates Foundation",
            "project_stage": "Concept note"
        },
        "problem_policy_relevance": {
            "problem_statement": "Ghana's maternal mortality ratio remains high at 308 per 100,000 live births due to delayed care seeking for pregnancy complications. Only 45% of pregnant women in rural areas access antenatal care 4+ times.",
            "evidence_gap": "Limited evidence on whether SMS reminders + community health worker engagement improves antenatal care attendance in West African contexts with high mobile phone penetration.",
            "beneficiaries": "Approximately 8,000 pregnant women in 40 health facility catchment areas across Ashanti Region",
            "policy_priority": "Ghana's Reproductive Health Policy prioritizes increasing antenatal care attendance as key to achieving SDG 3 targets."
        },
        "theory_of_change": {
            "activities": [
                "Train 80 community health workers in pregnancy danger signs and mobile app usage",
                "Establish WhatsApp peer support groups for pregnant women (5-8 per group)",
                "Implement SMS reminder system for antenatal care appointments",
                "Distribute pregnancy reminder cards with CHW contact information"
            ],
            "outputs": [
                "80 community health workers trained and active",
                "6,000 pregnant women enrolled in peer support groups",
                "80% of enrolled women receive SMS reminders throughout pregnancy"
            ],
            "short_run_outcomes": [
                "Pregnant women have increased knowledge of danger signs",
                "Antenatal care appointment attendance improves by 30%"
            ],
            "long_run_outcomes": [
                "Maternal complications detected earlier, reducing emergency obstetric cases by 20%",
                "Maternal mortality in intervention areas reduces by 25%"
            ],
            "assumptions": [
                "Community health workers will maintain regular contact with pregnant women",
                "Mobile phone coverage remains stable in rural areas",
                "Pregnant women will take action upon receiving danger sign alerts"
            ],
            "risks": [
                "Turnover of community health workers due to lack of payment",
                "Network outages during rainy season affect SMS delivery",
                "Cultural resistance to digital health interventions"
            ]
        },
        # ... (other sections would follow similar structure)
    }


def get_agriculture_sample_data() -> dict:
    """
    Generate a sample concept note for an agriculture intervention
    (Basic structure - can be expanded)
    """
    return {
        "meta": {
            "title": "Climate-Smart Agriculture and Soil Health Training Program in Kenya",
            "pis": ["Dr. James Kipchoge", "Dr. Nancy Kipchirchir"],
            "organization": "East Africa Agricultural Research Institute",
            "country": "Kenya",
            "funder": "CGIAR",
            "project_stage": "Concept note"
        },
        "problem_policy_relevance": {
            "problem_statement": "Drought-induced crop failures in semi-arid Kenya reduced small-holder farm income by 60% in 2022. Farmers lack knowledge of water-saving and soil-building practices suited to local conditions.",
            "evidence_gap": "Limited evidence on farmer adoption rates and yield impacts of integrating climate-smart agriculture training with input provision in pastoral/agro-pastoral transitions zones.",
            "beneficiaries": "2,500 small-holder farmers (60% women) in Kajiado and Narok counties",
            "policy_priority": "Kenya's Big Four Agenda prioritizes improving agricultural productivity while supporting climate resilience"
        },
        "theory_of_change": {
            "activities": [
                "Conduct 48 farmer field schools on improved soil and water conservation practices",
                "Distribute climate-adapted seed varieties and micro-irrigation equipment kits",
                "Establish farmer producer groups to enable input bundling and market linkages",
                "Create digital extension platform for seasonal climate forecasting and advisory"
            ],
            "outputs": [
                "2,500 farmers trained in conservation agriculture",
                "1,000 farmers adopt at least 2 climate-smart practices",
                "50 farmer producer groups established and registered"
            ],
            "short_run_outcomes": [
                "Farmer knowledge of soil health practices improves by 40%",
                "Adoption of water harvesting structures increases from 5% to 30%"
            ],
            "long_run_outcomes": [
                "Maize yield increases by 25% compared to baseline",
                "Farm income increases by 35% enabling improved household nutrition"
            ],
            "assumptions": [
                "Farmers will have capital to purchase improved seeds",
                "Mobile phone network enables digital extension service access",
                "Climate forecasts are timely and actionable"
            ],
            "risks": [
                "Government seed subsidy changes may disrupt input availability",
                "Extreme drought during implementation could undermine adoption",
                "Social norms may limit women farmer participation"
            ]
        },
        # ... (other sections would follow similar structure)
    }


# Dictionary mapping sector names to their sample data generators
SAMPLE_DATA_GENERATORS = {
    "education": get_education_sample_data,
    "health": get_health_sample_data,
    "agriculture": get_agriculture_sample_data,
}


def get_sample_data(sector: str = "education") -> dict:
    """
    Retrieve sample data for a given sector
    
    Args:
        sector: Sector name ('education', 'health', 'agriculture')
        
    Returns:
        Dictionary with sample concept note data
    """
    sector_lower = sector.lower().strip()
    
    if sector_lower not in SAMPLE_DATA_GENERATORS:
        raise ValueError(f"Unknown sector: {sector}. Available: {', '.join(SAMPLE_DATA_GENERATORS.keys())}")
    
    return SAMPLE_DATA_GENERATORS[sector_lower]()
