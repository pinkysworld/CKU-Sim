# Empirical Follow-up Summary

This note summarizes the paper-facing empirical position of the current repository.

## Primary Evidence

### 1. Frozen audited external replication

Result directory:

- `data/results/e20_external_replication__expanded7_with_gitea__audited_v1`

Headline numbers:

- 9 training repositories
- 7 disjoint holdout repositories
- 50 holdout positive files
- `baseline_plus_composite` versus `baseline_history_plus_structure`:
  - ROC AUC: `0.836` versus `0.828`
  - average precision: `0.026` versus `0.023`
  - Brier score: `0.005554` versus `0.005604`

Interpretation:

- Opacity adds modest but externally validated predictive value beyond the baseline history-and-structure model.

### 2. Direct external quantification-failure diagnostics

Result directory:

- `data/results/e24_external_quantification_failure__expanded7_with_gitea__audited_v1`

Headline numbers:

- 43 positive files
- 86 positive prediction rows
- 6 repositories represented in the grouped low/high opacity comparison
- `direct_quantification_failure_gate = true`

Interpretation:

- High-opacity positive files are more underpredicted and more likely to be missed by top-k ranking diagnostics.
- This is the strongest direct empirical evidence in the repository for CKU's quantification-limits mechanism.

### 3. Focused audited intervention expansion

Result directory:

- `data/results/e26_external_intervention_securityfile_enriched__focused4_h1825_v1`

Headline numbers:

- 33 accepted audited intervention rows
- 30 matched intervention-control pairs
- 18 positive pair rows
- `delta_absolute_error_did < 0` with non-zero clustered confidence intervals
- `delta_underprediction_loss_did < 0` with non-zero clustered confidence intervals
- `delta_positive_log_loss_did < 0` with non-zero clustered confidence intervals
- `delta_score_range_did` negative overall, but mixed across clustering levels
- the older pooled proxy gate remains false in the saved summary

Interpretation:

- Structural simplification improves several direct quantification outcomes.
- The older stricter proxy gate remains conservative because the pooled composite-change endpoint is still mixed.

## Recommended Manuscript Framing

Use the following claim boundary:

- "The follow-up provides empirical confirmation of the operational CKU claim that higher-opacity software is harder to quantify reliably and that opacity-aware models capture externally validated risk signal missed by opacity-blind baselines."

Avoid the following overclaim:

- "The follow-up empirically proves undecidability or fully proves the original CKU theorem."

## Publication-Facing Position

The current repository supports:

- a strong empirical follow-up paper;
- externally validated predictive evidence;
- direct evidence for quantification failure on high-opacity positive files;
- materially improved intervention-side evidence after audited refactoring enrichment.

The current repository does not support:

- a claim that every CKU proxy endpoint now points in the same direction;
- a claim that the mathematical undecidability argument itself has been empirically demonstrated.
