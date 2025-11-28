# RCT Design Wizard –> Concept Note

I want to redesign the RCT Design. Read and understand the structure below and perform the tasks.

## 0. Meta / Cover

Data: title, PIs, affiliations, country, funder, project stage, partners.
Output: header block (table or key-value).

## 1. Problem & Policy Relevance

Fields: problem_statement, evidence_gap, beneficiaries, policy_priority.
Goal: concise articulation of an information/credit constraint + its magnitude + why evidence is needed now.

## 2. Theory of Change (ToC)

Elements: activities, outputs, short_run_outcomes, long_run_outcomes, assumptions, risks.
Goal: causal pathway + testable links + key assumptions.

## 3. Intervention & Implementation

Fields: description, components, delivery_channel(s), frequency/intensity, eligibility_criteria, implementers, operational_constraints.
Goal: reproducible intervention specification.

## 4. Study Population & Sampling Frame

Fields: unit_of_randomization, sampling_frame_source, inclusion_exclusion, expected_total_N, coverage_limitations.
Goal: define population & frame robustness.

## 5. Outcomes & Measurement

Primary (≤3) + secondary, measurement_timing (baseline/midline/endline), instruments, indices/composites, qualitative_modules.
Goal: pre-specified, unbiased, time-bound measures.

## 6. Randomization Design

Numeric: design_type, arms, seed, strata, balance_summary.
Narrative: rationale (variance reduction, fairness), implementation steps, concealment, contamination mitigation, clustering justification.

## 7. Power & Sample Size

Numeric: N_per_arm, ICC (if clustered), MDE, assumptions (alpha, power, variance, attrition, take-up).
Narrative: effect size justification, variance source, attrition inflation, sensitivity analyses, design effect explanation.

## 8. Data Collection Plan

Fields: survey_schedule, enumerator_training, tracking_attrition, QC_protocols, mode (CAPI/SMS), data_security.
Goal: reliability & minimizing missingness.

## 9. Analysis Plan

Estimands: ITT, ToT.
Models: regression_spec, controls_set, heterogeneity (subgroups), multiple_testing_strategy, missing_data_strategy, mediation/exploratory flags.
Goal: transparent pre-analysis logic.

## 10. Ethics, Risks & Approvals

Fields: IRB_status, consent_process, adverse_event_protocol, privacy_security, risk_matrix (type, rating, mitigation), fairness (waitlist).
Goal: compliance & participant protection.

## 11. Timeline & Milestones

Fields: milestone_list (start/end dates), dependencies, critical_path.
Goal: feasibility demonstration.

## 12. Budget Summary

Fields: categories (personnel, data, intervention, overheads, contingency), amounts, funding_gap, co-financing.
Goal: cost transparency & scalability signals.

## 13. Policy Relevance & Scalability

Fields: alignment_with_national_policies, scale_pathways, delivery_model_comparison, cost_effectiveness_narrative.
Goal: external impact potential.

## 14. References & Annexes

Fields: citations (bib/bare), instruments_list, randomization_code_link, survey_modules, partner_MOU_status.
Goal: reproducibility & credibility.

## 15. Compliance (Org Standards)

Fields: prereg_plan, data_management_plan, back_check_protocol, transparency_commitments.

---

# Agent Task Plan

| ID  | Task                                                      | Priority | Category        | Inputs                    | Output / Acceptance Criteria                                        |
| --- | --------------------------------------------------------- | -------- | --------------- | ------------------------- | ------------------------------------------------------------------- |
| T0  | Put the current design into archive folder. Do not delete | High     |                 |                           |                                                                     |
| T1  | Create JSON schema (v1) for all sections                  | High     | Data Model      | Outline above             | schema.json with all keys & null defaults                           |
| T2  | Implement load/save helpers (storage.py)                  | High     | Backend         | schema.json               | Functions load_state(), save_state()                                |
| T3  | Build session adapter for Randomization page              | High     | Integration     | st.session_state.rand     | Populates numeric fields if empty                                   |
| T4  | Build session adapter for Power page                      | High     | Integration     | st.session_state.power    | Populates power numeric fields if empty                             |
| T5  | Auto-narrative generator (randomization)                  | High     | Logic           | design_type, strata, arms | Function returns default narrative if blank                         |
| T6  | Auto-narrative generator (power)                          | High     | Logic           | n_per_arm, mde, icc       | Function returns default narrative if blank                         |
| T7  | Streamlit page layout with expanders                      | High     | UI              | schema                    | All sections rendered; placeholders present                         |
| T8  | Validation layer (soft warnings)                          | Medium   | UX              | current state             | Warnings: >3 primary outcomes, missing ICC for cluster, N_per_arm=0 |
| T9  | Export Jinja2 template (Markdown)                         | High     | Export          | state                     | File content includes numeric + narrative blocks                    |
| T10 | Download button implementation                            | High     | UI              | rendered markdown         | User can download concept_note.md                                   |
| T11 | Add schema version & migration stub                       | Medium   | Maintainability | schema.json               | version key + migrate() placeholder                                 |
| T12 | Budget table summarizer                                   | Medium   | Utility         | category list             | Auto total & % distribution                                         |
| T13 | Timeline serialization                                    | Medium   | Utility         | milestone inputs          | Normalized list with ISO dates                                      |
| T14 | Risk matrix component                                     | Medium   | UI              | risk entries              | Table-like rendering + validation rating 1–5                       |
| T15 | Reference input normalization                             | Low      | Utility         | free-text citations       | Simple list; stub for future BibTeX                                 |
| T16 | Save progress button & autosave hook                      | High     | UX              | state changes             | JSON updated; toast confirmation                                    |
| T17 | Unit tests for storage & auto-narratives                  | Medium   | QA              | functions                 | pytest suite passing                                                |
| T18 | Documentation README for Wizard module                    | Medium   | Docs            | architecture summary      | README with usage + integration contract                            |
| T19 | Optional LaTeX export scaffold                            | Low      | Export          | markdown state            | latex_template.tex draft                                            |
| T20 | Integration checklist with existing pages                 | High     | Integration     | contracts                 | Table of required session keys validated at load                    |

---

## Task Sequencing (Sprint 1 Suggested)

1: T0, T1, T2, T3, T4, T5, T6, T7, T9, T16, T18, T20
Sprint 2: T8, T12, T13, T14, T10 (enhancements), T11, T17
Sprint 3: T15, T19 + refinements, performance review.

## Integration Contracts

- Randomization page must expose st.session_state["rand"] with keys: design, arms, seed, strata(list or CSV), balance_summary.
- Power page must expose st.session_state["power"] with keys: n_per_arm (or n_each), mde, icc (if clustered), assumptions, notes.
- Wizard only writes back to its own state; does not mutate upstream pages.

## Acceptance Criteria (High-Level)

- User can open page, see placeholders, fill narratives, import numeric values automatically.
- Exported Markdown includes all filled sections in correct order.
- No crash if upstream pages absent; numeric fields remain editable.
- Warnings display but do not block export.

## Risk Mitigation

- Missing upstream data: graceful fallback defaults.
- Schema evolution: version tagging.
- Oversized narratives: soft character count advisory (>1200 chars).

## Future Extensions (Defer)

- LaTeX export aligned with sample EOI.
- Multiple project selection.
- Pre-registration text generator (AEA registry stub).
- Optional RAG suggestions (re-integrate later).
