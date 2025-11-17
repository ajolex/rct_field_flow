"""
RCT Design Wizard - Main Page
Concept Note Builder with 15 sections
"""

import streamlit as st

# Import local modules - use relative imports since this is within the package
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import storage
import adapters
import narratives
import export_formats as export_module
import sample_data


def initialize_state():
    """Initialize wizard state in session"""
    if "wizard_state" not in st.session_state:
        st.session_state.wizard_state = storage.load_state()
    
    if "wizard_project_name" not in st.session_state:
        st.session_state.wizard_project_name = "default"


def sync_from_upstream():
    """Sync numeric values from randomization and power pages"""
    state = st.session_state.wizard_state
    
    # Sync randomization data
    if "rand" in st.session_state:
        rand_numeric = adapters.adapt_randomization_state(st.session_state)
        state["randomization_design"]["numeric"] = rand_numeric
        
        # Auto-generate narratives if empty
        rand_narrative = state["randomization_design"]["narrative"]
        if narratives.should_generate_narrative(rand_narrative.get("rationale")):
            auto_narrative = narratives.generate_randomization_narrative(rand_numeric)
            for key, value in auto_narrative.items():
                if value and narratives.should_generate_narrative(rand_narrative.get(key)):
                    rand_narrative[key] = value
    
    # Sync power data
    if "power" in st.session_state:
        power_numeric = adapters.adapt_power_state(st.session_state)
        state["power_sample_size"]["numeric"] = power_numeric
        
        # Auto-generate narratives if empty
        power_narrative = state["power_sample_size"]["narrative"]
        if narratives.should_generate_narrative(power_narrative.get("effect_size_justification")):
            auto_narrative = narratives.generate_power_narrative(power_numeric)
            for key, value in auto_narrative.items():
                if value and narratives.should_generate_narrative(power_narrative.get(key)):
                    power_narrative[key] = value


def render_section_0_meta():
    """Section 0: Meta/Cover"""
    with st.expander("📋 **0. Meta / Cover**", expanded=False):
        state = st.session_state.wizard_state["meta"]
        
        st.info("💡 **Tip:** Start with a clear, descriptive title that tells reviewers what the study is about in one glance.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            state["title"] = st.text_input(
                "Project Title", 
                value=state.get("title") or "",
                placeholder="Bridge to Basics: A Remedial Literacy RCT in Malawian Primary Schools"
            )
            state["country"] = st.text_input(
                "Country", 
                value=state.get("country") or "",
                placeholder="Malawi"
            )
            state["funder"] = st.text_input(
                "Funder", 
                value=state.get("funder") or "",
                placeholder="Ministry of Education & DFID"
            )
        
        with col2:
            state["project_stage"] = st.selectbox(
                "Project Stage",
                ["Concept", "Design", "Implementation", "Analysis", "Completed"],
                index=0 if not state.get("project_stage") else 
                      ["Concept", "Design", "Implementation", "Analysis", "Completed"].index(state["project_stage"])
            )
        
        # Lists
        pis_text = st.text_area(
            "Principal Investigators (one per line)", 
            value="\n".join(state.get("pis") or []),
            placeholder="Dr. Sarah Mwanza, University of Malawi\nDr. James Banda, Education Policy Institute"
        )
        state["pis"] = [pi.strip() for pi in pis_text.split("\n") if pi.strip()]
        
        affiliations_text = st.text_area(
            "Affiliations (one per line)", 
            value="\n".join(state.get("affiliations") or []),
            placeholder="University of Malawi, Centre for Educational Research\nEducation Policy Institute, Lilongwe"
        )
        state["affiliations"] = [aff.strip() for aff in affiliations_text.split("\n") if aff.strip()]
        
        partners_text = st.text_area(
            "Partners (one per line)", 
            value="\n".join(state.get("partners") or []),
            placeholder="Ministry of Education, Science and Technology\nEducation Foundation NGO\nDistrict Education Offices (3 districts)"
        )
        state["partners"] = [p.strip() for p in partners_text.split("\n") if p.strip()]


def render_section_1_problem():
    """Section 1: Problem & Policy Relevance"""
    with st.expander("🎯 **1. Problem & Policy Relevance**", expanded=False):
        state = st.session_state.wizard_state["problem_policy_relevance"]
        
        st.info("💡 **Tip:** Keep your problem statement specific to one population and one outcome. Avoid trying to solve multiple problems simultaneously.")
        
        state["problem_statement"] = st.text_area(
            "Problem Statement",
            value=state.get("problem_statement") or "",
            height=150,
            help="Concise articulation of the information/credit constraint",
            placeholder="Example: In Malawi's rural districts, 55% of grade 4 students read below grade level despite universal primary education. "
                       "This learning crisis stems from large class sizes (averaging 85 students per teacher), lack of remedial instruction, "
                       "and limited parental engagement in supporting home reading practice."
        )
        
        state["evidence_gap"] = st.text_area(
            "Evidence Gap",
            value=state.get("evidence_gap") or "",
            height=100,
            help="Why is new evidence needed?",
            placeholder="Example: While there is evidence that reduced class sizes improve learning in high-income contexts, "
                       "we lack rigorous evidence on whether structured small-group remedial instruction delivered by trained facilitators "
                       "can close literacy gaps in resource-constrained settings. Existing interventions show mixed results, with "
                       "implementation fidelity varying widely across contexts."
        )
        
        state["beneficiaries"] = st.text_area(
            "Beneficiaries",
            value=state.get("beneficiaries") or "",
            height=100,
            help="Who will benefit from this research?",
            placeholder="Example: Primary beneficiaries are 3,200 below-grade-level grade 4 students across 40 schools in three districts. "
                       "Secondary beneficiaries include their families, teachers receiving training in remedial instruction methods, "
                       "and policymakers who will use findings to inform national literacy programs reaching 2.5 million students annually."
        )
        
        state["policy_priority"] = st.text_area(
            "Policy Priority",
            value=state.get("policy_priority") or "",
            height=100,
            help="Why is evidence needed now?",
            placeholder="Example: The Ministry of Education's 2024-2030 National Education Strategy prioritizes foundational literacy and "
                       "allocates $15M for remedial programs. The school board requires evidence within two academic terms to decide whether "
                       "to expand this approach district-wide. Without credible evaluation, this investment risks being ineffective or misallocated."
        )


def render_section_2_toc():
    """Section 2: Theory of Change"""
    with st.expander("🔄 **2. Theory of Change**", expanded=False):
        state = st.session_state.wizard_state["theory_of_change"]
        
        st.info("💡 **Tip:** If you cannot draw a tight line from activity to outcome, consider narrowing scope. Flag assumptions you're least confident about or that are most critical for success.")
        
        st.markdown("**Example ToC Structure:**")
        st.caption("Activities → Outputs → Short-run Outcomes → Long-run Outcomes")
        
        activities_text = st.text_area(
            "Activities (one per line)", 
            value="\n".join(state.get("activities") or []),
            height=100,
            placeholder="Daily 45-minute literacy sessions in small groups of 12 students\nParent engagement workshops every two weeks\nWeekly SMS tips to caregivers with home reading activities"
        )
        state["activities"] = [a.strip() for a in activities_text.split("\n") if a.strip()]
        
        outputs_text = st.text_area(
            "Outputs (one per line)", 
            value="\n".join(state.get("outputs") or []),
            height=100,
            placeholder="3,200 students attend 80% of remedial sessions\n60 teachers trained in structured literacy pedagogy\n2,400 parents receive weekly reading support messages"
        )
        state["outputs"] = [o.strip() for o in outputs_text.split("\n") if o.strip()]
        
        short_outcomes_text = st.text_area(
            "Short-Run Outcomes (one per line)", 
            value="\n".join(state.get("short_run_outcomes") or []),
            height=100,
            placeholder="Students' reading fluency increases by 30 words per minute within 6 months\nParent involvement in home reading doubles\nTeachers adopt new pedagogical methods in regular classes"
        )
        state["short_run_outcomes"] = [s.strip() for s in short_outcomes_text.split("\n") if s.strip()]
        
        long_outcomes_text = st.text_area(
            "Long-Run Outcomes (one per line)", 
            value="\n".join(state.get("long_run_outcomes") or []),
            height=100,
            placeholder="70% of participants achieve grade-level reading proficiency by year-end\nReduced dropout rates in grades 5-8\nImproved academic performance across subjects requiring reading comprehension"
        )
        state["long_run_outcomes"] = [outcome.strip() for outcome in long_outcomes_text.split("\n") if outcome.strip()]
        
        st.divider()
        
        assumptions_text = st.text_area(
            "Key Assumptions (one per line)", 
            value="\n".join(state.get("assumptions") or []),
            height=100,
            placeholder="Students will attend 80% of sessions (may drop if sessions conflict with harvest season)\nTeachers will maintain instruction quality without continuous supervision\nParents can support home reading despite low literacy levels themselves\nSchools will provide adequate space for small-group instruction"
        )
        state["assumptions"] = [a.strip() for a in assumptions_text.split("\n") if a.strip()]
        
        risks_text = st.text_area(
            "Risks (one per line)", 
            value="\n".join(state.get("risks") or []),
            height=100,
            placeholder="Spillover effects: Control group students may benefit from trained teachers in regular classes\nAttendance may be lower than expected during agricultural peak seasons\nTeacher turnover could disrupt program continuity\nParental SMS engagement may be limited by phone access"
        )
        state["risks"] = [r.strip() for r in risks_text.split("\n") if r.strip()]


def render_section_3_intervention():
    """Section 3: Intervention & Implementation"""
    with st.expander("🔧 **3. Intervention & Implementation**", expanded=False):
        state = st.session_state.wizard_state["intervention_implementation"]
        
        st.info("💡 **Tip:** Provide enough detail that someone else could replicate your intervention. Think about the 'who, what, when, where, how' of delivery.")
        
        state["description"] = st.text_area(
            "Intervention Description",
            value=state.get("description") or "",
            height=200,
            help="Detailed description of the intervention",
            placeholder="The 'Bridge to Basics' remedial literacy program delivers structured small-group instruction to below-grade-level readers. "
                       "Trained facilitators conduct daily 45-minute sessions with groups of 12 students using tiered lesson plans aligned to "
                       "foundational reading benchmarks. Sessions take place during after-school hours in designated classrooms. The curriculum "
                       "combines phonics drills, guided reading, and comprehension exercises. Parents receive bi-weekly SMS messages with home "
                       "practice activities and tips for supporting their child's reading development."
        )
        
        components_text = st.text_area(
            "Intervention Components (one per line)", 
            value="\n".join(state.get("components") or []),
            placeholder="Small-group literacy instruction (core component)\nTeacher training in remedial pedagogy\nParent engagement via SMS\nPeriodic assessment and progress tracking\nSupplementary reading materials and workbooks"
        )
        state["components"] = [c.strip() for c in components_text.split("\n") if c.strip()]
        
        channels_text = st.text_area(
            "Delivery Channels (one per line)", 
            value="\n".join(state.get("delivery_channels") or []),
            placeholder="In-person instruction in school classrooms\nSMS messaging platform (parent engagement)\nMonthly teacher support workshops\nMobile assessment app for progress monitoring"
        )
        state["delivery_channels"] = [d.strip() for d in channels_text.split("\n") if d.strip()]
        
        state["frequency_intensity"] = st.text_input(
            "Frequency/Intensity",
            value=state.get("frequency_intensity") or "",
            placeholder="Daily 45-minute sessions, 5 days/week for 12 months; Parent SMS twice weekly"
        )
        
        state["eligibility_criteria"] = st.text_area(
            "Eligibility Criteria",
            value=state.get("eligibility_criteria") or "",
            placeholder="Inclusion: Grade 4 students scoring below grade-level on baseline reading fluency assessment (<60 words per minute)\n"
                       "Exclusion: Students with diagnosed learning disabilities requiring specialized support, students absent >30 days in prior term"
        )
        
        implementers_text = st.text_area(
            "Implementers (one per line)", 
            value="\n".join(state.get("implementers") or []),
            placeholder="Municipal government (program sponsor)\nPartner NGO Education Foundation (training and coordination)\n60 trained remedial facilitators (seconded teachers)\n40 school principals (site supervisors)"
        )
        state["implementers"] = [i.strip() for i in implementers_text.split("\n") if i.strip()]
        
        constraints_text = st.text_area(
            "Operational Constraints (one per line)", 
            value="\n".join(state.get("operational_constraints") or []),
            placeholder="Limited classroom space requires rotating schedule for different groups\nRural schools may have unreliable electricity affecting SMS delivery\nTeacher availability constrained during exam periods\nBudget limits number of trained facilitators to 60"
        )
        state["operational_constraints"] = [c.strip() for c in constraints_text.split("\n") if c.strip()]


def render_section_4_population():
    """Section 4: Study Population & Sampling Frame"""
    with st.expander("👥 **4. Study Population & Sampling Frame**", expanded=False):
        state = st.session_state.wizard_state["study_population_sampling"]
        
        st.info("💡 **Tip:** If spillovers feel unavoidable, consider cluster-level assignment. Ensure your sampling frame is exhaustive and up-to-date.")
        
        state["unit_of_randomization"] = st.text_input(
            "Unit of Randomization",
            value=state.get("unit_of_randomization") or "",
            help="e.g., individual, household, village, school",
            placeholder="Individual students (to minimize spillovers within schools, we randomize students rather than classrooms)"
        )
        
        state["sampling_frame_source"] = st.text_area(
            "Sampling Frame Source",
            value=state.get("sampling_frame_source") or "",
            height=100,
            placeholder="District Education Management Information System (EMIS) database of all grade 4 students in 40 participating schools, "
                       "cross-validated with school enrollment registers. Frame includes student ID, school, gender, age, and baseline test scores. "
                       "Data validated and updated one month before randomization."
        )
        
        state["inclusion_exclusion"] = st.text_area(
            "Inclusion/Exclusion Criteria",
            value=state.get("inclusion_exclusion") or "",
            height=100,
            placeholder="Inclusion: Grade 4 students (ages 9-11) in participating schools who score <60 words/min on baseline reading assessment\n"
                       "Exclusion: Students with diagnosed cognitive disabilities requiring specialized education; students who have repeated grade 4 more than once; "
                       "students with <80% attendance in prior term (indicating high dropout risk)"
        )
        
        state["expected_total_n"] = st.number_input(
            "Expected Total N",
            min_value=0,
            value=int(state.get("expected_total_n") or 0)
        )
        
        state["coverage_limitations"] = st.text_area(
            "Coverage Limitations",
            value=state.get("coverage_limitations") or "",
            height=100,
            placeholder="Study covers 40 of 120 schools in the district (those with adequate facilities for after-school programs). "
                       "Excludes remote schools >30km from district center due to logistical constraints. Sample may under-represent "
                       "most disadvantaged students who have already dropped out. Generalizability limited to semi-urban and peri-urban schools "
                       "with basic infrastructure."
        )


def render_section_5_outcomes():
    """Section 5: Outcomes & Measurement"""
    with st.expander("📊 **5. Outcomes & Measurement**", expanded=False):
        state = st.session_state.wizard_state["outcomes_measurement"]
        
        st.info("💡 **Tip:** If an indicator feels fuzzy, add a quick definition or example. Prioritize outcomes you can measure credibly with your resources.")
        
        primary_text = st.text_area(
            "Primary Outcomes (≤3, one per line)", 
            value="\n".join(state.get("primary_outcomes") or []),
            help="Maximum 3 primary outcomes recommended",
            placeholder="Reading fluency (words read correctly per minute on standardized assessment)\n"
                       "Reading comprehension (score on 10-question comprehension test, 0-100 scale)\n"
                       "Grade progression (percentage of students promoted to grade 5)"
        )
        state["primary_outcomes"] = [p.strip() for p in primary_text.split("\n") if p.strip()]
        
        # Warning if more than 3 primary outcomes
        if len(state["primary_outcomes"]) > 3:
            st.warning("⚠️ You have more than 3 primary outcomes. Consider reducing to strengthen statistical power.")
        
        secondary_text = st.text_area(
            "Secondary Outcomes (one per line)", 
            value="\n".join(state.get("secondary_outcomes") or []),
            placeholder="Student attendance rate\nParental engagement (parent-reported time spent on home reading)\n"
                       "Student confidence (self-reported reading self-efficacy scale)\n"
                       "Teacher practices (adoption of remedial methods in regular instruction)"
        )
        state["secondary_outcomes"] = [s.strip() for s in secondary_text.split("\n") if s.strip()]
        
        st.subheader("Measurement Timing")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state["measurement_timing"]["baseline"] = st.text_input(
                "Baseline",
                value=state["measurement_timing"].get("baseline") or ""
            )
        
        with col2:
            state["measurement_timing"]["midline"] = st.text_input(
                "Midline",
                value=state["measurement_timing"].get("midline") or ""
            )
        
        with col3:
            state["measurement_timing"]["endline"] = st.text_input(
                "Endline",
                value=state["measurement_timing"].get("endline") or ""
            )
        
        instruments_text = st.text_area(
            "Measurement Instruments (one per line)", 
            value="\n".join(state.get("instruments") or []),
            placeholder="EGRA (Early Grade Reading Assessment) - standardized 1-minute oral reading fluency test\n"
                       "Reading comprehension test (validated 10-item assessment developed by national curriculum board)\n"
                       "Student attendance tracked via school registers\n"
                       "Parent survey (phone-based, 15 minutes) on home reading practices"
        )
        state["instruments"] = [i.strip() for i in instruments_text.split("\n") if i.strip()]
        
        indices_text = st.text_area(
            "Indices/Composites (one per line)", 
            value="\n".join(state.get("indices_composites") or []),
            placeholder="Overall literacy index (z-score combining fluency and comprehension)\n"
                       "Engagement index (combining attendance, parent involvement, and student self-efficacy)\n"
                       "Implementation fidelity index (session attendance, content coverage, materials distribution)"
        )
        state["indices_composites"] = [i.strip() for i in indices_text.split("\n") if i.strip()]
        
        qual_text = st.text_area(
            "Qualitative Modules (one per line)", 
            value="\n".join(state.get("qualitative_modules") or []),
            placeholder="Focus group discussions with 10 students from each school (treatment and control) on learning experiences\n"
                       "Semi-structured interviews with 30 teachers on program implementation challenges\n"
                       "Observation protocol for 20 randomly selected instruction sessions"
        )
        state["qualitative_modules"] = [q.strip() for q in qual_text.split("\n") if q.strip()]


def render_section_6_randomization():
    """Section 6: Randomization Design"""
    with st.expander("🎲 **6. Randomization Design**", expanded=False):
        state = st.session_state.wizard_state["randomization_design"]
        
        st.subheader("Numeric Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            state["numeric"]["design_type"] = st.selectbox(
                "Design Type",
                ["Simple", "Stratified", "Cluster", "Stratified Cluster"],
                index=0 if not state["numeric"].get("design_type") else
                      ["Simple", "Stratified", "Cluster", "Stratified Cluster"].index(
                          state["numeric"]["design_type"].title() if state["numeric"].get("design_type") else "Simple"
                      )
            )
            
            state["numeric"]["arms"] = st.number_input(
                "Number of Arms",
                min_value=2,
                value=int(state["numeric"].get("arms") or 2)
            )
        
        with col2:
            state["numeric"]["seed"] = st.number_input(
                "Randomization Seed",
                min_value=0,
                value=int(state["numeric"].get("seed") or 12345)
            )
        
        strata_text = st.text_area(
            "Strata Variables (one per line)", 
            value="\n".join(state["numeric"].get("strata") or []) if isinstance(state["numeric"].get("strata"), list)
                  else str(state["numeric"].get("strata") or "")
        )
        state["numeric"]["strata"] = [s.strip() for s in strata_text.split("\n") if s.strip()]
        
        state["numeric"]["balance_summary"] = st.text_area(
            "Balance Summary",
            value=state["numeric"].get("balance_summary") or "",
            help="Summary of balance checks"
        )
        
        st.divider()
        st.subheader("Narrative")
        
        # Button to regenerate narratives
        if st.button("🔄 Regenerate Auto-Narratives", key="regen_rand"):
            auto_narrative = narratives.generate_randomization_narrative(state["numeric"])
            for key, value in auto_narrative.items():
                if value:
                    state["narrative"][key] = value
            st.success("Narratives regenerated!")
            st.rerun()
        
        state["narrative"]["rationale"] = st.text_area(
            "Rationale",
            value=state["narrative"].get("rationale") or "",
            height=150,
            help="Why this design? Variance reduction, fairness, etc."
        )
        
        state["narrative"]["implementation_steps"] = st.text_area(
            "Implementation Steps",
            value=state["narrative"].get("implementation_steps") or "",
            height=150
        )
        
        state["narrative"]["concealment"] = st.text_area(
            "Concealment Strategy",
            value=state["narrative"].get("concealment") or "",
            height=100
        )
        
        state["narrative"]["contamination_mitigation"] = st.text_area(
            "Contamination Mitigation",
            value=state["narrative"].get("contamination_mitigation") or "",
            height=100
        )
        
        if "cluster" in state["numeric"].get("design_type", "").lower():
            state["narrative"]["clustering_justification"] = st.text_area(
                "Clustering Justification",
                value=state["narrative"].get("clustering_justification") or "",
                height=100
            )


def render_section_7_power():
    """Section 7: Power & Sample Size"""
    with st.expander("⚡ **7. Power & Sample Size**", expanded=False):
        state = st.session_state.wizard_state["power_sample_size"]
        
        st.subheader("Numeric Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state["numeric"]["n_per_arm"] = st.number_input(
                "N per Arm",
                min_value=0,
                value=int(state["numeric"].get("n_per_arm") or 0)
            )
            
            state["numeric"]["alpha"] = st.number_input(
                "Alpha (Significance Level)",
                min_value=0.0,
                max_value=1.0,
                value=float(state["numeric"].get("alpha") or 0.05),
                step=0.01
            )
        
        with col2:
            state["numeric"]["mde"] = st.number_input(
                "MDE (Minimum Detectable Effect)",
                min_value=0.0,
                value=float(state["numeric"].get("mde") or 0.0),
                step=0.01,
                format="%.3f"
            )
            
            state["numeric"]["power"] = st.number_input(
                "Power",
                min_value=0.0,
                max_value=1.0,
                value=float(state["numeric"].get("power") or 0.80),
                step=0.01
            )
        
        with col3:
            state["numeric"]["icc"] = st.number_input(
                "ICC (if clustered)",
                min_value=0.0,
                max_value=1.0,
                value=float(state["numeric"].get("icc") or 0.0),
                step=0.01,
                format="%.3f"
            )
            
            state["numeric"]["variance"] = st.number_input(
                "Variance",
                min_value=0.0,
                value=float(state["numeric"].get("variance") or 0.0),
                step=0.01
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            state["numeric"]["attrition"] = st.number_input(
                "Expected Attrition Rate",
                min_value=0.0,
                max_value=1.0,
                value=float(state["numeric"].get("attrition") or 0.0),
                step=0.01,
                format="%.2f"
            )
        
        with col5:
            state["numeric"]["take_up"] = st.number_input(
                "Expected Take-up Rate",
                min_value=0.0,
                max_value=1.0,
                value=float(state["numeric"].get("take_up") or 1.0),
                step=0.01,
                format="%.2f"
            )
        
        # Warnings
        if state["numeric"].get("n_per_arm") == 0:
            st.warning("⚠️ N per arm is 0. Please specify sample size.")
        
        if state["numeric"].get("icc") and state["numeric"]["icc"] > 0:
            if "cluster" not in st.session_state.wizard_state["randomization_design"]["numeric"].get("design_type", "").lower():
                st.warning("⚠️ ICC is specified but design type is not clustered.")
        
        st.divider()
        st.subheader("Narrative")
        
        # Button to regenerate narratives
        if st.button("🔄 Regenerate Auto-Narratives", key="regen_power"):
            auto_narrative = narratives.generate_power_narrative(state["numeric"])
            for key, value in auto_narrative.items():
                if value:
                    state["narrative"][key] = value
            st.success("Narratives regenerated!")
            st.rerun()
        
        state["narrative"]["effect_size_justification"] = st.text_area(
            "Effect Size Justification",
            value=state["narrative"].get("effect_size_justification") or "",
            height=150
        )
        
        state["narrative"]["variance_source"] = st.text_area(
            "Variance Source",
            value=state["narrative"].get("variance_source") or "",
            height=100
        )
        
        state["narrative"]["attrition_inflation"] = st.text_area(
            "Attrition Inflation Strategy",
            value=state["narrative"].get("attrition_inflation") or "",
            height=100
        )
        
        state["narrative"]["sensitivity_analyses"] = st.text_area(
            "Sensitivity Analyses",
            value=state["narrative"].get("sensitivity_analyses") or "",
            height=150
        )
        
        if state["numeric"].get("icc") and state["numeric"]["icc"] > 0:
            state["narrative"]["design_effect_explanation"] = st.text_area(
                "Design Effect Explanation",
                value=state["narrative"].get("design_effect_explanation") or "",
                height=100
            )


def render_section_8_data_collection():
    """Section 8: Data Collection Plan"""
    with st.expander("📝 **8. Data Collection Plan**", expanded=False):
        state = st.session_state.wizard_state["data_collection_plan"]
        
        st.info("💡 **Tip:** Build in quality checks early. Set up systems to catch data issues within 24 hours, not after data collection is complete.")
        
        state["survey_schedule"] = st.text_area(
            "Survey Schedule",
            value=state.get("survey_schedule") or "",
            height=100,
            placeholder="Baseline: First 2 weeks of school year (January 2025), before program starts\n"
                       "Midline: End of Term 2 (June 2025), 6 months into program\n"
                       "Endline: End of school year (November 2025), 12 months after baseline\n"
                       "Each wave: 2-week data collection window per school, staggered across 40 schools"
        )
        
        state["enumerator_training"] = st.text_area(
            "Enumerator Training Plan",
            value=state.get("enumerator_training") or "",
            height=100,
            placeholder="5-day training for 20 enumerators covering: assessment administration, CAPI system, child safeguarding protocols, "
                       "and inter-rater reliability exercises. Day 5: field pilot in 2 non-study schools. Enumerators must achieve 90% "
                       "inter-rater reliability on fluency scoring before deployment. Refresher training before each survey wave."
        )
        
        state["tracking_attrition"] = st.text_area(
            "Tracking & Attrition Protocols",
            value=state.get("tracking_attrition") or "",
            height=100,
            placeholder="Collect 3 phone contacts per student at baseline (parent, guardian, neighbor). Track absent students within 48 hours. "
                       "For movers: attempt to locate via school admin and community leaders, conduct phone-based assessment if necessary. "
                       "For dropouts: brief exit interview to understand reasons. Weekly attrition tracking meetings to flag patterns early. "
                       "Budget includes transport costs for up to 3 follow-up attempts per student."
        )
        
        state["qc_protocols"] = st.text_area(
            "Quality Control Protocols",
            value=state.get("qc_protocols") or "",
            height=100,
            placeholder="Daily automated checks: missing data, outliers, duplicate IDs. Supervisors audit 10% of assessments through live observation. "
                       "Back-checks on 5% of parent surveys within 72 hours. High-frequency checks (HFCs) flag enumerators with >5% error rates for retraining. "
                       "Data dashboard shows daily completion rates, assessment score distributions, and enumerator performance metrics. "
                       "Weekly team meetings to review quality metrics and address issues."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            state["mode"] = st.selectbox(
                "Data Collection Mode",
                ["CAPI", "PAPI", "Phone/SMS", "Mixed", "Other"],
                index=0 if not state.get("mode") else
                      ["CAPI", "PAPI", "Phone/SMS", "Mixed", "Other"].index(state["mode"])
                      if state["mode"] in ["CAPI", "PAPI", "Phone/SMS", "Mixed", "Other"] else 0
            )
        
        state["data_security"] = st.text_area(
            "Data Security & Privacy",
            value=state.get("data_security") or "",
            height=100
        )


def render_section_9_analysis():
    """Section 9: Analysis Plan"""
    with st.expander("📈 **9. Analysis Plan**", expanded=False):
        state = st.session_state.wizard_state["analysis_plan"]
        
        st.subheader("Estimands")
        
        col1, col2 = st.columns(2)
        
        with col1:
            state["estimands"]["itt"] = st.text_area(
                "Intent-to-Treat (ITT)",
                value=state["estimands"].get("itt") or "",
                height=100
            )
        
        with col2:
            state["estimands"]["tot"] = st.text_area(
                "Treatment-on-Treated (ToT)",
                value=state["estimands"].get("tot") or "",
                height=100
            )
        
        st.divider()
        st.subheader("Models")
        
        state["models"]["regression_spec"] = st.text_area(
            "Regression Specification",
            value=state["models"].get("regression_spec") or "",
            height=150,
            help="Specify the main regression equation"
        )
        
        controls_text = st.text_area("Control Variables (one per line)", 
                                     value="\n".join(state["models"].get("controls_set") or []))
        state["models"]["controls_set"] = [c.strip() for c in controls_text.split("\n") if c.strip()]
        
        hetero_text = st.text_area("Heterogeneity Subgroups (one per line)", 
                                   value="\n".join(state["models"].get("heterogeneity_subgroups") or []))
        state["models"]["heterogeneity_subgroups"] = [h.strip() for h in hetero_text.split("\n") if h.strip()]
        
        state["models"]["multiple_testing_strategy"] = st.text_area(
            "Multiple Testing Strategy",
            value=state["models"].get("multiple_testing_strategy") or "",
            height=100,
            help="e.g., Bonferroni, FDR, step-down"
        )
        
        state["models"]["missing_data_strategy"] = st.text_area(
            "Missing Data Strategy",
            value=state["models"].get("missing_data_strategy") or "",
            height=100
        )
        
        mediation_text = st.text_area("Mediation/Exploratory Analyses (one per line)", 
                                      value="\n".join(state["models"].get("mediation_exploratory_flags") or []))
        state["models"]["mediation_exploratory_flags"] = [m.strip() for m in mediation_text.split("\n") if m.strip()]


def render_section_10_ethics():
    """Section 10: Ethics, Risks & Approvals"""
    with st.expander("⚖️ **10. Ethics, Risks & Approvals**", expanded=False):
        state = st.session_state.wizard_state["ethics_risks_approvals"]
        
        st.info("💡 **Tip:** Assign a lead person to each risk so follow-up happens quickly. Consider what could go wrong and how you'll protect participants.")
        
        state["irb_status"] = st.selectbox(
            "IRB Status",
            ["Not yet submitted", "Submitted", "Approved", "Exempt"],
            index=0 if not state.get("irb_status") else
                  ["Not yet submitted", "Submitted", "Approved", "Exempt"].index(state["irb_status"])
                  if state["irb_status"] in ["Not yet submitted", "Submitted", "Approved", "Exempt"] else 0
        )
        
        state["consent_process"] = st.text_area(
            "Consent Process",
            value=state.get("consent_process") or "",
            height=100,
            placeholder="Parental consent obtained via information sessions at each school, conducted in local language with translated materials. "
                       "Parents receive written consent forms explaining study purpose, procedures, risks, benefits, and right to withdraw. "
                       "Student assent obtained verbally before each assessment. For illiterate parents, consent process includes witnessed thumb print. "
                       "Consent forms stored in locked cabinet, separate from study data."
        )
        
        state["adverse_event_protocol"] = st.text_area(
            "Adverse Event Protocol",
            value=state.get("adverse_event_protocol") or "",
            height=100,
            placeholder="Study coordinator serves as designated safety officer. Any adverse events (student distress, assessment-related anxiety, "
                       "disclosure of abuse) reported within 24 hours to PI and ethics committee. Enumerators trained in child safeguarding protocols "
                       "and referral pathways to social services. Emergency contact list for local child protection services maintained. "
                       "Monthly safety reviews conducted by ethics board designee."
        )
        
        state["privacy_security"] = st.text_area(
            "Privacy & Security Measures",
            value=state.get("privacy_security") or "",
            height=100,
            placeholder="Data encrypted at rest and in transit. Access restricted to core research team via password-protected servers. "
                       "Student names replaced with unique IDs immediately after data collection; ID-name linkage file stored separately on encrypted drive. "
                       "No personally identifiable information shared outside research team. Assessment conducted in private spaces to ensure confidentiality. "
                       "Data retention: anonymized data stored for 7 years; identifying information destroyed after final data cleaning."
        )
        
        st.subheader("Risk Matrix")
        st.caption("Add risks with their ratings (1-5) and mitigation strategies")
        
        # Simple risk matrix input
        risk_matrix_text = st.text_area(
            "Risk Matrix (Format: Risk Type | Rating (1-5) | Mitigation)",
            value="\n".join([f"{r.get('type', '')} | {r.get('rating', '')} | {r.get('mitigation', '')}" 
                            for r in state.get("risk_matrix") or []]),
            height=150,
            help="Example: Data breach | 3 | Encrypted storage and limited access"
        )
        
        # Parse risk matrix
        risks = []
        for line in risk_matrix_text.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        risks.append({
                            "type": parts[0],
                            "rating": int(parts[1]) if parts[1].isdigit() else 0,
                            "mitigation": parts[2]
                        })
                    except Exception:
                        pass
        state["risk_matrix"] = risks
        
        state["fairness_waitlist"] = st.text_area(
            "Fairness & Waitlist Control",
            value=state.get("fairness_waitlist") or "",
            height=100
        )


def render_section_11_timeline():
    """Section 11: Timeline & Milestones"""
    with st.expander("📅 **11. Timeline & Milestones**", expanded=False):
        state = st.session_state.wizard_state["timeline_milestones"]
        
        st.caption("Add milestones with start and end dates")
        
        milestones_text = st.text_area(
            "Milestones (Format: Milestone Name | Start Date | End Date)",
            value="\n".join([f"{m.get('name', '')} | {m.get('start', '')} | {m.get('end', '')}" 
                            for m in state.get("milestones") or []]),
            height=200,
            help="Example: Baseline Survey | 2024-01-15 | 2024-03-30"
        )
        
        # Parse milestones
        milestones = []
        for line in milestones_text.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    milestones.append({
                        "name": parts[0],
                        "start": parts[1],
                        "end": parts[2]
                    })
        state["milestones"] = milestones
        
        state["dependencies"] = st.text_area(
            "Dependencies",
            value=state.get("dependencies") or "",
            height=100,
            help="Describe task dependencies and sequencing"
        )
        
        state["critical_path"] = st.text_area(
            "Critical Path",
            value=state.get("critical_path") or "",
            height=100,
            help="Identify the critical path through the project"
        )


def render_section_12_budget():
    """Section 12: Budget Summary"""
    with st.expander("💰 **12. Budget Summary**", expanded=False):
        state = st.session_state.wizard_state["budget_summary"]
        
        st.subheader("Budget Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            state["categories"]["personnel"] = st.number_input(
                "Personnel",
                min_value=0.0,
                value=float(state["categories"].get("personnel") or 0.0),
                step=1000.0
            )
            
            state["categories"]["data"] = st.number_input(
                "Data Collection",
                min_value=0.0,
                value=float(state["categories"].get("data") or 0.0),
                step=1000.0
            )
            
            state["categories"]["intervention"] = st.number_input(
                "Intervention Costs",
                min_value=0.0,
                value=float(state["categories"].get("intervention") or 0.0),
                step=1000.0
            )
        
        with col2:
            state["categories"]["overheads"] = st.number_input(
                "Overheads",
                min_value=0.0,
                value=float(state["categories"].get("overheads") or 0.0),
                step=1000.0
            )
            
            state["categories"]["contingency"] = st.number_input(
                "Contingency",
                min_value=0.0,
                value=float(state["categories"].get("contingency") or 0.0),
                step=1000.0
            )
        
        # Calculate total
        total = sum([
            state["categories"].get("personnel") or 0,
            state["categories"].get("data") or 0,
            state["categories"].get("intervention") or 0,
            state["categories"].get("overheads") or 0,
            state["categories"].get("contingency") or 0
        ])
        
        st.metric("Total Budget", f"${total:,.2f}")
        
        # Show distribution
        if total > 0:
            st.caption("Budget Distribution:")
            for category, amount in state["categories"].items():
                if amount:
                    pct = (amount / total) * 100
                    st.write(f"- {category.title()}: ${amount:,.2f} ({pct:.1f}%)")
        
        st.divider()
        
        state["funding_gap"] = st.number_input(
            "Funding Gap",
            min_value=0.0,
            value=float(state.get("funding_gap") or 0.0),
            step=1000.0
        )
        
        state["co_financing"] = st.text_area(
            "Co-financing Details",
            value=state.get("co_financing") or "",
            height=100
        )


def render_section_13_policy():
    """Section 13: Policy Relevance & Scalability"""
    with st.expander("🌍 **13. Policy Relevance & Scalability**", expanded=False):
        state = st.session_state.wizard_state["policy_relevance_scalability"]
        
        state["alignment_with_national_policies"] = st.text_area(
            "Alignment with National Policies",
            value=state.get("alignment_with_national_policies") or "",
            height=150
        )
        
        state["scale_pathways"] = st.text_area(
            "Pathways to Scale",
            value=state.get("scale_pathways") or "",
            height=150
        )
        
        state["delivery_model_comparison"] = st.text_area(
            "Delivery Model Comparison",
            value=state.get("delivery_model_comparison") or "",
            height=100
        )
        
        state["cost_effectiveness_narrative"] = st.text_area(
            "Cost-Effectiveness Narrative",
            value=state.get("cost_effectiveness_narrative") or "",
            height=100
        )


def render_section_14_references():
    """Section 14: References & Annexes"""
    with st.expander("📚 **14. References & Annexes**", expanded=False):
        state = st.session_state.wizard_state["references_annexes"]
        
        citations_text = st.text_area(
            "Citations (one per line)",
            value="\n".join(state.get("citations") or []),
            height=150,
            help="List key references"
        )
        state["citations"] = [c.strip() for c in citations_text.split("\n") if c.strip()]
        
        instruments_text = st.text_area(
            "Instruments List (one per line)",
            value="\n".join(state.get("instruments_list") or [])
        )
        state["instruments_list"] = [i.strip() for i in instruments_text.split("\n") if i.strip()]
        
        state["randomization_code_link"] = st.text_input(
            "Randomization Code Link",
            value=state.get("randomization_code_link") or ""
        )
        
        survey_modules_text = st.text_area(
            "Survey Modules (one per line)",
            value="\n".join(state.get("survey_modules") or [])
        )
        state["survey_modules"] = [s.strip() for s in survey_modules_text.split("\n") if s.strip()]
        
        state["partner_mou_status"] = st.text_input(
            "Partner MOU Status",
            value=state.get("partner_mou_status") or ""
        )


def render_section_15_compliance():
    """Section 15: Compliance (Org Standards)"""
    with st.expander("✅ **15. Compliance**", expanded=False):
        state = st.session_state.wizard_state["compliance"]
        
        state["prereg_plan"] = st.text_area(
            "Pre-registration Plan",
            value=state.get("prereg_plan") or "",
            height=100,
            help="e.g., AEA RCT Registry, OSF"
        )
        
        state["data_management_plan"] = st.text_area(
            "Data Management Plan",
            value=state.get("data_management_plan") or "",
            height=100
        )
        
        state["back_check_protocol"] = st.text_area(
            "Back-check Protocol",
            value=state.get("back_check_protocol") or "",
            height=100
        )
        
        transparency_text = st.text_area(
            "Transparency Commitments (one per line)",
            value="\n".join(state.get("transparency_commitments") or [])
        )
        state["transparency_commitments"] = [t.strip() for t in transparency_text.split("\n") if t.strip()]


def render_all_sections():
    """Render all 15 sections"""
    render_section_0_meta()
    render_section_1_problem()
    render_section_2_toc()
    render_section_3_intervention()
    render_section_4_population()
    render_section_5_outcomes()
    render_section_6_randomization()
    render_section_7_power()
    render_section_8_data_collection()
    render_section_9_analysis()
    render_section_10_ethics()
    render_section_11_timeline()
    render_section_12_budget()
    render_section_13_policy()
    render_section_14_references()
    render_section_15_compliance()


def main():
    """Main wizard page"""
    st.title("📝 RCT Design Wizard")
    st.caption("Concept Note Builder - 15 Sections")
    
    # Initialize
    initialize_state()
    
    # Sync from upstream pages
    sync_from_upstream()
    
    # Quick Start Guide
    with st.expander("📖 **Quick Start Guide**", expanded=False):
        st.markdown("""
        ### Welcome to the RCT Design Wizard!
        
        This tool helps you create a comprehensive concept note for your randomized controlled trial. 
        
        **How to use this wizard:**
        1. **Move through sections in order** - Each section builds on the previous one
        2. **Use the examples** - Placeholder text shows realistic examples from education, health, and agriculture projects in social development
        3. **Save frequently** - Click "💾 Save Progress" to save your work
        4. **Auto-generate narratives** - Click regenerate buttons in Sections 6 & 7 to auto-fill technical descriptions
        5. **Export when ready** - Click "📥 Export Markdown" to download your concept note
        
        **Tips for success:**
        - Be specific: Define your population, outcomes, and intervention clearly
        - Keep it focused: Limit to 1-3 primary outcomes for statistical power
        - Think implementation: Consider what could go wrong and how you'll detect issues early
        - Check assumptions: Flag the assumptions you're least confident about
        
        **Example program contexts (see placeholders):**
        - 📚 Education: Remedial literacy program in Malawian primary schools
        - 🏥 Health: Maternal health home visits by community workers
        - 🌾 Agriculture: Drip irrigation + SMS advisory for smallholder farmers
        """)
    
    # Integration status
    with st.expander("🔗 Integration Status", expanded=False):
        integration_status = adapters.get_integration_status(st.session_state)
        st.info(integration_status)
    
    # Project selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        projects = storage.list_projects()
        if not projects:
            projects = ["default"]
        
        selected_project = st.selectbox(
            "Select Project",
            projects,
            index=projects.index(st.session_state.wizard_project_name) 
                  if st.session_state.wizard_project_name in projects else 0
        )
        
        if selected_project != st.session_state.wizard_project_name:
            st.session_state.wizard_project_name = selected_project
            st.session_state.wizard_state = storage.load_state(selected_project)
            st.rerun()
    
    with col2:
        new_project_name = st.text_input("New project name", placeholder="my_project")
    
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("➕ Create New") and new_project_name:
            st.session_state.wizard_project_name = new_project_name
            st.session_state.wizard_state = storage.get_default_state()
            storage.save_state(st.session_state.wizard_state, new_project_name)
            st.success(f"Created project: {new_project_name}")
            st.rerun()
    
    st.divider()
    
    # Render all sections
    render_all_sections()
    
    st.divider()
    
    # Sample Data & Export Section
    with st.expander("📋 **Sample & Export**", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 View Sample Concept Note")
            st.markdown("Generate a complete concept note with example data to see how your output will look.")
            
            sector = st.radio(
                "Choose sector for sample:",
                ["education", "health", "agriculture"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            if st.button("👁️ View Sample", use_container_width=True, key="view_sample"):
                try:
                    sample_state = sample_data.get_sample_data(sector)
                    
                    # Create tabs for preview
                    tab1, tab2, tab3 = st.tabs(["📄 Markdown Preview", "📊 JSON Data", "✅ Fields Completed"])
                    
                    with tab1:
                        sample_markdown = export_module.render_markdown(sample_state)
                        st.markdown(sample_markdown)
                    
                    with tab2:
                        st.json(sample_state)
                    
                    with tab3:
                        # Count filled fields
                        filled_count = 0
                        total_count = 0
                        
                        def count_fields(obj):
                            nonlocal filled_count, total_count
                            if isinstance(obj, dict):
                                for v in obj.values():
                                    count_fields(v)
                            elif isinstance(obj, list):
                                for item in obj:
                                    count_fields(item)
                            elif obj is not None and obj != "" and obj != []:
                                filled_count += 1
                            total_count += 1
                        
                        count_fields(sample_state)
                        
                        st.metric("Fields Completed", f"{filled_count} / {total_count}", 
                                 delta=f"{(filled_count/total_count*100):.0f}%")
                        st.caption(f"This is an example. Your project currently has {len([k for k in st.session_state.wizard_state if st.session_state.wizard_state[k]])} sections with data.")
                    
                except Exception as e:
                    st.error(f"Error loading sample: {e}")
        
        with col2:
            st.subheader("📥 Export Your Concept Note")
            st.markdown("Download your concept note in your preferred format:")
            
            export_format = st.radio(
                "Select export format:",
                ["Markdown (.md)", "Word Document (.docx)", "PDF (.pdf)"],
                label_visibility="collapsed"
            )
            
            # Map format labels to format codes
            format_map = {
                "Markdown (.md)": "markdown",
                "Word Document (.docx)": "docx",
                "PDF (.pdf)": "pdf"
            }
            
            format_code = format_map[export_format]
            
            if st.button("📥 Generate Export", use_container_width=True, key="export_button"):
                try:
                    project_name = st.session_state.wizard_project_name
                    
                    if format_code == "markdown":
                        markdown = export_module.render_markdown(st.session_state.wizard_state)
                        st.download_button(
                            label=f"⬇️ Download {export_format}",
                            data=markdown,
                            file_name=f"{project_name}_concept_note.md",
                            mime="text/markdown",
                            key="download_md"
                        )
                    
                    elif format_code == "docx":
                        docx_bytes = export_module.export_to_docx(st.session_state.wizard_state)
                        st.download_button(
                            label=f"⬇️ Download {export_format}",
                            data=docx_bytes,
                            file_name=f"{project_name}_concept_note.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_docx"
                        )
                    
                    elif format_code == "pdf":
                        try:
                            pdf_bytes = export_module.export_to_pdf(st.session_state.wizard_state)
                            st.download_button(
                                label=f"⬇️ Download {export_format}",
                                data=pdf_bytes,
                                file_name=f"{project_name}_concept_note.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                        except ImportError as pdf_error:
                            st.warning("📦 PDF export requires additional dependencies")
                            st.code("pip install weasyprint", language="bash")
                            with st.expander("Why PDF export needs extra setup:"):
                                st.write("""
                                Weasyprint requires system libraries (GTK/GObject) that aren't included in standard Python.
                                
                                **Workarounds:**
                                1. **Use Markdown** (.md) - export and convert to PDF elsewhere
                                2. **Use DOCX** (.docx) - open in Word and "Save As PDF"
                                3. **Install weasyprint** - see instructions above
                                """)
                            raise
                    
                    st.toast(f"✓ {export_format} ready for download!", icon="✅")
                
                except ImportError as e:
                    pass  # Already handled by PDF-specific error above
                except Exception as e:
                    st.error(f"Export failed: {e}")
                    st.toast("❌ Export failed", icon="❌")
            
            st.divider()
            st.caption("""
            **Format recommendations:**
            - **Markdown**: Best for version control & git tracking
            - **DOCX**: Easy to edit in Word/Google Docs outside the app
            - **PDF**: Professional appearance & guaranteed formatting
            """)
    
    st.divider()
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("� Save Progress", use_container_width=True):
            success = storage.save_state(
                st.session_state.wizard_state,
                st.session_state.wizard_project_name
            )
            if success:
                st.toast("✓ Progress saved!", icon="✅")
            else:
                st.toast("❌ Save failed", icon="❌")
    
    with col2:
        if st.button("�🔄 Sync from Pages", use_container_width=True):
            sync_from_upstream()
            st.toast("✓ Synced from Randomization & Power pages", icon="✅")
            st.rerun()
    
    with col3:
        if st.button("📊 View JSON", use_container_width=True):
            with st.expander("Current State (JSON)", expanded=True):
                st.json(st.session_state.wizard_state)
    
    with col4:
        st.write("")


if __name__ == "__main__":
    main()
