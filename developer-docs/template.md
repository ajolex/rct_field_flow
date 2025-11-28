# {{ meta.title or "RCT Design Concept Note" }}

{% if meta.pis %}
**Principal Investigators:** {{ meta.pis | join(", ") }}
{% endif %}

{% if meta.affiliations %}
**Affiliations:** {{ meta.affiliations | join("; ") }}
{% endif %}

{% if meta.country %}
**Country:** {{ meta.country }}
{% endif %}

{% if meta.funder %}
**Funder:** {{ meta.funder }}
{% endif %}

{% if meta.project_stage %}
**Project Stage:** {{ meta.project_stage }}
{% endif %}

{% if meta.partners %}
**Partners:** {{ meta.partners | join(", ") }}
{% endif %}

---

## 1. Problem & Policy Relevance

{% if problem_policy_relevance.problem_statement %}
### Problem Statement
{{ problem_policy_relevance.problem_statement }}
{% endif %}

{% if problem_policy_relevance.evidence_gap %}
### Evidence Gap
{{ problem_policy_relevance.evidence_gap }}
{% endif %}

{% if problem_policy_relevance.beneficiaries %}
### Beneficiaries
{{ problem_policy_relevance.beneficiaries }}
{% endif %}

{% if problem_policy_relevance.policy_priority %}
### Policy Priority
{{ problem_policy_relevance.policy_priority }}
{% endif %}

---

## 2. Theory of Change

{% if theory_of_change.activities %}
### Activities
{% for activity in theory_of_change.activities %}
- {{ activity }}
{% endfor %}
{% endif %}

{% if theory_of_change.outputs %}
### Outputs
{% for output in theory_of_change.outputs %}
- {{ output }}
{% endfor %}
{% endif %}

{% if theory_of_change.short_run_outcomes %}
### Short-Run Outcomes
{% for outcome in theory_of_change.short_run_outcomes %}
- {{ outcome }}
{% endfor %}
{% endif %}

{% if theory_of_change.long_run_outcomes %}
### Long-Run Outcomes
{% for outcome in theory_of_change.long_run_outcomes %}
- {{ outcome }}
{% endfor %}
{% endif %}

{% if theory_of_change.assumptions %}
### Key Assumptions
{% for assumption in theory_of_change.assumptions %}
- {{ assumption }}
{% endfor %}
{% endif %}

{% if theory_of_change.risks %}
### Risks
{% for risk in theory_of_change.risks %}
- {{ risk }}
{% endfor %}
{% endif %}

---

## 3. Intervention & Implementation

{% if intervention_implementation.description %}
### Description
{{ intervention_implementation.description }}
{% endif %}

{% if intervention_implementation.components %}
### Intervention Components
{% for component in intervention_implementation.components %}
- {{ component }}
{% endfor %}
{% endif %}

{% if intervention_implementation.delivery_channels %}
### Delivery Channels
{% for channel in intervention_implementation.delivery_channels %}
- {{ channel }}
{% endfor %}
{% endif %}

{% if intervention_implementation.frequency_intensity %}
### Frequency/Intensity
{{ intervention_implementation.frequency_intensity }}
{% endif %}

{% if intervention_implementation.eligibility_criteria %}
### Eligibility Criteria
{{ intervention_implementation.eligibility_criteria }}
{% endif %}

{% if intervention_implementation.implementers %}
### Implementers
{% for implementer in intervention_implementation.implementers %}
- {{ implementer }}
{% endfor %}
{% endif %}

{% if intervention_implementation.operational_constraints %}
### Operational Constraints
{% for constraint in intervention_implementation.operational_constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

---

## 4. Study Population & Sampling Frame

{% if study_population_sampling.unit_of_randomization %}
**Unit of Randomization:** {{ study_population_sampling.unit_of_randomization }}
{% endif %}

{% if study_population_sampling.expected_total_n %}
**Expected Total N:** {{ study_population_sampling.expected_total_n }}
{% endif %}

{% if study_population_sampling.sampling_frame_source %}
### Sampling Frame Source
{{ study_population_sampling.sampling_frame_source }}
{% endif %}

{% if study_population_sampling.inclusion_exclusion %}
### Inclusion/Exclusion Criteria
{{ study_population_sampling.inclusion_exclusion }}
{% endif %}

{% if study_population_sampling.coverage_limitations %}
### Coverage Limitations
{{ study_population_sampling.coverage_limitations }}
{% endif %}

---

## 5. Outcomes & Measurement

{% if outcomes_measurement.primary_outcomes %}
### Primary Outcomes
{% for outcome in outcomes_measurement.primary_outcomes %}
- {{ outcome }}
{% endfor %}
{% endif %}

{% if outcomes_measurement.secondary_outcomes %}
### Secondary Outcomes
{% for outcome in outcomes_measurement.secondary_outcomes %}
- {{ outcome }}
{% endfor %}
{% endif %}

{% if outcomes_measurement.measurement_timing %}
### Measurement Timing
{% if outcomes_measurement.measurement_timing.baseline %}
- **Baseline:** {{ outcomes_measurement.measurement_timing.baseline }}
{% endif %}
{% if outcomes_measurement.measurement_timing.midline %}
- **Midline:** {{ outcomes_measurement.measurement_timing.midline }}
{% endif %}
{% if outcomes_measurement.measurement_timing.endline %}
- **Endline:** {{ outcomes_measurement.measurement_timing.endline }}
{% endif %}
{% endif %}

{% if outcomes_measurement.instruments %}
### Measurement Instruments
{% for instrument in outcomes_measurement.instruments %}
- {{ instrument }}
{% endfor %}
{% endif %}

{% if outcomes_measurement.indices_composites %}
### Indices/Composites
{% for index in outcomes_measurement.indices_composites %}
- {{ index }}
{% endfor %}
{% endif %}

{% if outcomes_measurement.qualitative_modules %}
### Qualitative Modules
{% for module in outcomes_measurement.qualitative_modules %}
- {{ module }}
{% endfor %}
{% endif %}

---

## 6. Randomization Design

### Numeric Parameters

{% if randomization_design.numeric.design_type %}
- **Design Type:** {{ randomization_design.numeric.design_type }}
{% endif %}
{% if randomization_design.numeric.arms %}
- **Number of Arms:** {{ randomization_design.numeric.arms }}
{% endif %}
{% if randomization_design.numeric.seed %}
- **Randomization Seed:** {{ randomization_design.numeric.seed }}
{% endif %}
{% if randomization_design.numeric.strata %}
- **Strata Variables:** {{ randomization_design.numeric.strata | join(", ") }}
{% endif %}

{% if randomization_design.numeric.balance_summary %}
### Balance Summary
{{ randomization_design.numeric.balance_summary }}
{% endif %}

{% if randomization_design.narrative.rationale %}
### Rationale
{{ randomization_design.narrative.rationale }}
{% endif %}

{% if randomization_design.narrative.implementation_steps %}
### Implementation Steps
{{ randomization_design.narrative.implementation_steps }}
{% endif %}

{% if randomization_design.narrative.concealment %}
### Concealment Strategy
{{ randomization_design.narrative.concealment }}
{% endif %}

{% if randomization_design.narrative.contamination_mitigation %}
### Contamination Mitigation
{{ randomization_design.narrative.contamination_mitigation }}
{% endif %}

{% if randomization_design.narrative.clustering_justification %}
### Clustering Justification
{{ randomization_design.narrative.clustering_justification }}
{% endif %}

---

## 7. Power & Sample Size

### Numeric Parameters

{% if power_sample_size.numeric.n_per_arm %}
- **N per Arm:** {{ power_sample_size.numeric.n_per_arm }}
{% endif %}
{% if power_sample_size.numeric.mde %}
- **MDE (Minimum Detectable Effect):** {{ power_sample_size.numeric.mde }}
{% endif %}
{% if power_sample_size.numeric.alpha %}
- **Alpha:** {{ power_sample_size.numeric.alpha }}
{% endif %}
{% if power_sample_size.numeric.power %}
- **Power:** {{ power_sample_size.numeric.power }}
{% endif %}
{% if power_sample_size.numeric.icc %}
- **ICC:** {{ power_sample_size.numeric.icc }}
{% endif %}
{% if power_sample_size.numeric.variance %}
- **Variance:** {{ power_sample_size.numeric.variance }}
{% endif %}
{% if power_sample_size.numeric.attrition %}
- **Expected Attrition:** {{ power_sample_size.numeric.attrition }}
{% endif %}
{% if power_sample_size.numeric.take_up %}
- **Expected Take-up:** {{ power_sample_size.numeric.take_up }}
{% endif %}

{% if power_sample_size.narrative.effect_size_justification %}
### Effect Size Justification
{{ power_sample_size.narrative.effect_size_justification }}
{% endif %}

{% if power_sample_size.narrative.variance_source %}
### Variance Source
{{ power_sample_size.narrative.variance_source }}
{% endif %}

{% if power_sample_size.narrative.attrition_inflation %}
### Attrition Inflation Strategy
{{ power_sample_size.narrative.attrition_inflation }}
{% endif %}

{% if power_sample_size.narrative.sensitivity_analyses %}
### Sensitivity Analyses
{{ power_sample_size.narrative.sensitivity_analyses }}
{% endif %}

{% if power_sample_size.narrative.design_effect_explanation %}
### Design Effect Explanation
{{ power_sample_size.narrative.design_effect_explanation }}
{% endif %}

---

## 8. Data Collection Plan

{% if data_collection_plan.mode %}
**Data Collection Mode:** {{ data_collection_plan.mode }}
{% endif %}

{% if data_collection_plan.survey_schedule %}
### Survey Schedule
{{ data_collection_plan.survey_schedule }}
{% endif %}

{% if data_collection_plan.enumerator_training %}
### Enumerator Training
{{ data_collection_plan.enumerator_training }}
{% endif %}

{% if data_collection_plan.tracking_attrition %}
### Tracking & Attrition Protocols
{{ data_collection_plan.tracking_attrition }}
{% endif %}

{% if data_collection_plan.qc_protocols %}
### Quality Control Protocols
{{ data_collection_plan.qc_protocols }}
{% endif %}

{% if data_collection_plan.data_security %}
### Data Security & Privacy
{{ data_collection_plan.data_security }}
{% endif %}

---

## 9. Analysis Plan

### Estimands

{% if analysis_plan.estimands.itt %}
**Intent-to-Treat (ITT):**
{{ analysis_plan.estimands.itt }}
{% endif %}

{% if analysis_plan.estimands.tot %}
**Treatment-on-Treated (ToT):**
{{ analysis_plan.estimands.tot }}
{% endif %}

### Models

{% if analysis_plan.models.regression_spec %}
**Regression Specification:**
{{ analysis_plan.models.regression_spec }}
{% endif %}

{% if analysis_plan.models.controls_set %}
**Control Variables:**
{% for control in analysis_plan.models.controls_set %}
- {{ control }}
{% endfor %}
{% endif %}

{% if analysis_plan.models.heterogeneity_subgroups %}
**Heterogeneity Subgroups:**
{% for subgroup in analysis_plan.models.heterogeneity_subgroups %}
- {{ subgroup }}
{% endfor %}
{% endif %}

{% if analysis_plan.models.multiple_testing_strategy %}
**Multiple Testing Strategy:**
{{ analysis_plan.models.multiple_testing_strategy }}
{% endif %}

{% if analysis_plan.models.missing_data_strategy %}
**Missing Data Strategy:**
{{ analysis_plan.models.missing_data_strategy }}
{% endif %}

{% if analysis_plan.models.mediation_exploratory_flags %}
**Mediation/Exploratory Analyses:**
{% for analysis in analysis_plan.models.mediation_exploratory_flags %}
- {{ analysis }}
{% endfor %}
{% endif %}

---

## 10. Ethics, Risks & Approvals

{% if ethics_risks_approvals.irb_status %}
**IRB Status:** {{ ethics_risks_approvals.irb_status }}
{% endif %}

{% if ethics_risks_approvals.consent_process %}
### Consent Process
{{ ethics_risks_approvals.consent_process }}
{% endif %}

{% if ethics_risks_approvals.adverse_event_protocol %}
### Adverse Event Protocol
{{ ethics_risks_approvals.adverse_event_protocol }}
{% endif %}

{% if ethics_risks_approvals.privacy_security %}
### Privacy & Security
{{ ethics_risks_approvals.privacy_security }}
{% endif %}

{% if ethics_risks_approvals.risk_matrix %}
### Risk Matrix

| Risk Type | Rating | Mitigation |
|-----------|--------|------------|
{% for risk in ethics_risks_approvals.risk_matrix %}
| {{ risk.type }} | {{ risk.rating }} | {{ risk.mitigation }} |
{% endfor %}
{% endif %}

{% if ethics_risks_approvals.fairness_waitlist %}
### Fairness & Waitlist Control
{{ ethics_risks_approvals.fairness_waitlist }}
{% endif %}

---

## 11. Timeline & Milestones

{% if timeline_milestones.milestones %}
### Milestones

| Milestone | Start Date | End Date |
|-----------|------------|----------|
{% for milestone in timeline_milestones.milestones %}
| {{ milestone.name }} | {{ milestone.start }} | {{ milestone.end }} |
{% endfor %}
{% endif %}

{% if timeline_milestones.dependencies %}
### Dependencies
{{ timeline_milestones.dependencies }}
{% endif %}

{% if timeline_milestones.critical_path %}
### Critical Path
{{ timeline_milestones.critical_path }}
{% endif %}

---

## 12. Budget Summary

{% if budget_summary.categories %}
### Budget Breakdown

| Category | Amount |
|----------|--------|
{% if budget_summary.categories.personnel %}
| Personnel | ${{ "{:,.2f}".format(budget_summary.categories.personnel) }} |
{% endif %}
{% if budget_summary.categories.data %}
| Data Collection | ${{ "{:,.2f}".format(budget_summary.categories.data) }} |
{% endif %}
{% if budget_summary.categories.intervention %}
| Intervention | ${{ "{:,.2f}".format(budget_summary.categories.intervention) }} |
{% endif %}
{% if budget_summary.categories.overheads %}
| Overheads | ${{ "{:,.2f}".format(budget_summary.categories.overheads) }} |
{% endif %}
{% if budget_summary.categories.contingency %}
| Contingency | ${{ "{:,.2f}".format(budget_summary.categories.contingency) }} |
{% endif %}
| **Total** | **${{ "{:,.2f}".format((budget_summary.categories.personnel or 0) + (budget_summary.categories.data or 0) + (budget_summary.categories.intervention or 0) + (budget_summary.categories.overheads or 0) + (budget_summary.categories.contingency or 0)) }}** |
{% endif %}

{% if budget_summary.funding_gap %}
**Funding Gap:** ${{ "{:,.2f}".format(budget_summary.funding_gap) }}
{% endif %}

{% if budget_summary.co_financing %}
### Co-financing
{{ budget_summary.co_financing }}
{% endif %}

---

## 13. Policy Relevance & Scalability

{% if policy_relevance_scalability.alignment_with_national_policies %}
### Alignment with National Policies
{{ policy_relevance_scalability.alignment_with_national_policies }}
{% endif %}

{% if policy_relevance_scalability.scale_pathways %}
### Pathways to Scale
{{ policy_relevance_scalability.scale_pathways }}
{% endif %}

{% if policy_relevance_scalability.delivery_model_comparison %}
### Delivery Model Comparison
{{ policy_relevance_scalability.delivery_model_comparison }}
{% endif %}

{% if policy_relevance_scalability.cost_effectiveness_narrative %}
### Cost-Effectiveness
{{ policy_relevance_scalability.cost_effectiveness_narrative }}
{% endif %}

---

## 14. References & Annexes

{% if references_annexes.citations %}
### References
{% for citation in references_annexes.citations %}
- {{ citation }}
{% endfor %}
{% endif %}

{% if references_annexes.instruments_list %}
### Instruments
{% for instrument in references_annexes.instruments_list %}
- {{ instrument }}
{% endfor %}
{% endif %}

{% if references_annexes.randomization_code_link %}
**Randomization Code:** {{ references_annexes.randomization_code_link }}
{% endif %}

{% if references_annexes.survey_modules %}
### Survey Modules
{% for module in references_annexes.survey_modules %}
- {{ module }}
{% endfor %}
{% endif %}

{% if references_annexes.partner_mou_status %}
**Partner MOU Status:** {{ references_annexes.partner_mou_status }}
{% endif %}

---

## 15. Compliance

{% if compliance.prereg_plan %}
### Pre-registration Plan
{{ compliance.prereg_plan }}
{% endif %}

{% if compliance.data_management_plan %}
### Data Management Plan
{{ compliance.data_management_plan }}
{% endif %}

{% if compliance.back_check_protocol %}
### Back-check Protocol
{{ compliance.back_check_protocol }}
{% endif %}

{% if compliance.transparency_commitments %}
### Transparency Commitments
{% for commitment in compliance.transparency_commitments %}
- {{ commitment }}
{% endfor %}
{% endif %}

---

*Generated by RCT Design Wizard v{{ version }}*
{% if _last_saved %}
*Last saved: {{ _last_saved }}*
{% endif %}
