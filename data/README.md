This directory contains the released data artifacts for `CKU-Sim`.

## Included

- `data/results/`: generated tables, figures, model outputs, and experiment summaries.
- `data/processed/security_event_file_audit_curated.csv`: curated accepted security event-to-file audit table used by the audited file-level program.
- `data/processed/ordinary_bugfix_control_audit_curated.csv`: curated audit table for the ordinary bug-fix negative controls.

## Excluded From Public Release

- `data/raw/`: large local clones of upstream source repositories.
- transient cache files such as raw NVD responses, OSV query caches, and local intermediate hydration artifacts.

## Primary Result Directories

The current empirical follow-up is centered on:

- `data/results/e20_external_replication__expanded7_with_gitea__audited_v1`
- `data/results/e24_external_quantification_failure__expanded7_with_gitea__audited_v1`
- `data/results/e26_external_intervention_securityfile_enriched__focused4_h1825_v1`

These directories provide the strongest audited external evidence for the follow-up paper:

- frozen external predictive validation on a disjoint holdout;
- direct quantification-failure diagnostics on positive holdout files;
- focused audited intervention evidence using deterministic security-file refactoring enrichment.

Older result directories are retained for methodological traceability, appendix material, and robustness reporting.
