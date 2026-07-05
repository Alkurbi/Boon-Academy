# Eval report

## Note-analysis goldens (field-level exact match)

- S005.last_contact_method: PASS
- S005.contact_outcome: PASS
- S005.intervention_succeeded: PASS
- S012.last_contact_method: PASS
- S012.contact_outcome: PASS
- S012.intervention_succeeded: PASS
- S004.contact_outcome: PASS
- S004.intervention_succeeded: PASS
- S019.last_contact_method: PASS
- S019.contact_outcome: PASS
- S019.intervention_succeeded: PASS
- S030.last_contact_method: PASS
- S030.contact_outcome: PASS
- S030.intervention_succeeded: PASS
- S075.last_contact_method: PASS
- S075.contact_outcome: PASS
- S075.intervention_succeeded: PASS
- S110.last_contact_method: PASS
- S110.contact_outcome: PASS
- S110.intervention_succeeded: PASS

**Accuracy: 20/20 = 100%** - OK

## Message-draft quality (LLM-as-judge, sample of 5)

- S113: grounded 5, tone 4, arabic 5, actionable 3
- S123: grounded 5, tone 3, arabic 5, actionable 2 - issue: No clear actionable next step for parent; 'لا يحل أسئلة يومياً' risks sounding accusatory to a previously defensive parent
- S012: grounded 3, tone 4, arabic 4, actionable 5
- S084: grounded 5, tone 3, arabic 4, actionable 4
- S149: grounded 5, tone 5, arabic 4, actionable 5

**Mean judge score: 4.2/5**