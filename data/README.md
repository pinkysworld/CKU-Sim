This repository includes generated experiment outputs in `data/results/`.

The public repository intentionally excludes:

- `data/raw/`: large cloned upstream source repositories
- `data/processed/`: transient caches such as NVD and OSV responses

Current contents:

- `data/results/e01_synthetic/`: synthetic separation outputs
- `data/results/e02_corpus/`: corpus-level opacity measurements
- `data/results/e03_cve/`: opacity/CVE merged outputs and figures
- `data/results/e04_temporal/`: temporal opacity series and aggregate figure
- `data/results/e05_insurance/`: Monte Carlo insurance outputs
- `data/results/e06_file_case_control/`: matched file-level case-control outputs and figures
- `data/results/e06_file_case_control__strict_nvd_event/`: event-collapsed NVD case-control outputs
- `data/results/e06_file_case_control__balanced_explicit_id_event/`: expanded explicit-ID case-control outputs
- `data/results/e06_file_case_control__expanded_advisory_event/`: advisory-expanded case-control outputs
- `data/results/e07_predictive_validation/`: held-out prediction outputs and comparison figures
- `data/results/e07_predictive_validation__e06_file_case_control__strict_nvd_event/`: predictive validation for the event-collapsed NVD specification
- `data/results/e07_predictive_validation__e06_file_case_control__balanced_explicit_id_event/`: predictive validation for the expanded explicit-ID specification
- `data/results/e07_predictive_validation__e06_file_case_control__expanded_advisory_event/`: predictive validation for the advisory-expanded specification
- `data/results/e08_policy_comparison/`: side-by-side policy robustness summaries and figures
- `data/results/e09_negative_control_bugfix/`: security-versus-ordinary-bugfix negative-control outputs
- `data/results/e09_negative_control_bugfix__e06_file_case_control__expanded_advisory_event/`: negative-control outputs using the advisory-expanded security dataset
- `data/results/e10_forward_release_panel__core8/`: forward-looking release-panel outputs for the current tag-rich core subset
- `data/results/e10_forward_corpus_screen__curated15_h730_l10/`: screening table and summary for the broader forward-looking candidate set
- `data/results/e10_forward_release_panel__light8_h730_l10/`: focused eight-repository forward-looking panel outputs with a fully observed two-year horizon
- `data/results/e11_large_corpus__filtered/`: GitHub-discovered expanded-corpus candidate list, validated subset, and manifest
- `data/results/e12_prospective_file_panel__light8_h730_l10/`: focused eight-repository prospective file-level panel outputs
- `data/results/e12_prospective_file_panel__curated15_h730_l10/`: screened 15-repository prospective file-level panel outputs with two sampled tags per repository
- `data/results/e12_prospective_file_panel__curated15_h730_l10_t5/`: screened 15-repository prospective file-level panel outputs with five sampled tags per repository
- `data/results/e12_prospective_file_panel__curated15_h365_l10_t5/`: screened 15-repository prospective file-level panel outputs at a one-year horizon
- `data/results/e12_prospective_file_panel__curated15_h365_l10_t5__high_critical/`: one-year prospective panel restricted to high/critical-severity future events
- `data/results/e12_prospective_file_panel__curated15_h730_l10_t5__high_critical/`: two-year prospective panel restricted to high/critical-severity future events
- `data/results/e13_prospective_label_audit__curated15_h730_l10_t5/`: reviewed audit sample and audit summaries for the main prospective file-level panel
- `data/results/e13_prospective_label_audit__curated15_h730_l10_t5__stratified120/`: larger source-stratified audit sample and review summaries for the main prospective file-level panel
- `data/results/e14_prospective_robustness__curated15/`: side-by-side horizon/severity robustness summaries for the prospective file-level panel
- `data/results/e15_negative_control_strict__expanded_advisory__light6/`: stricter same-subsystem security-versus-bugfix matched comparison on the lighter six-repository subset
- `data/results/e12_prospective_file_panel__external_holdout_flask_requests_h730_l10_t5/`: external-holdout prospective panel for Flask and Requests using the frozen prospective specification
- `data/results/e16_external_holdout__curated15_to_external_flask_requests/`: frozen train/test evaluation from the curated prospective corpus to the external Flask/Requests holdout

The file-level outputs correspond to the following event definitions:

- `nvd_commit_refs`: one observation per locally accessible NVD-linked fixing commit
- `strict_nvd_event`: one primary fixing event per NVD-linked vulnerability
- `balanced_explicit_id_event`: the event-collapsed NVD set augmented with locally explicit `CVE-...` and `GHSA-...` commit identifiers
- `expanded_advisory_event`: the event-collapsed NVD set augmented with OSV-linked repository advisories and locally explicit `CVE-...` and `GHSA-...` commit identifiers

The `e09_negative_control_bugfix` outputs compare security-fix files with ordinary bug-fix files drawn from the same repository histories. The policy-specific `e09_negative_control_bugfix__e06_file_case_control__expanded_advisory_event` outputs use the advisory-expanded `e06` security dataset.

The `e10_forward_release_panel__core8` outputs use release tags as pre-event snapshots and OSV-linked future advisories as later outcomes over a fixed horizon.

The `e10_forward_corpus_screen__curated15_h730_l10` outputs document the screening step used to identify the focused forward-panel subset. The `e10_forward_release_panel__light8_h730_l10` outputs correspond to the completed panel run on the screened eight-repository subset.

The `e11_large_corpus__filtered` outputs document a reproducible GitHub discovery pass for constructing a broader follow-on repository corpus. These files are intended to support later curation rather than to define a final publication corpus automatically.

The `e12` outputs are the forward-looking file-level analogue of the earlier matched case-control study. They use release snapshots as pre-event baselines, label future file involvement from advisory-linked fixing events, and compare future-case files against matched untouched controls from the same snapshot. The `e12_prospective_file_panel__curated15_h730_l10_t5` run is the densest completed specification and retains up to five sampled release tags per repository within the fully observed ten-year window. The `high_critical` variants restrict future events to observations with severity metadata mapped to high or critical labels.

The `e13_prospective_label_audit__curated15_h730_l10_t5` outputs summarize the original reviewed sample from the main `e12` run. The `e13_prospective_label_audit__curated15_h730_l10_t5__stratified120` outputs extend that audit to a 120-observation source-stratified sample, making it possible to distinguish stronger explicit/reference-backed labels from `osv_range`-only mappings while preserving ambiguous cases rather than forcing binary confirmation.

The `e14_prospective_robustness__curated15` outputs provide a compact comparison across one-year versus two-year horizons and all-severity versus high/critical-severity outcome definitions.

The `e15_negative_control_strict__expanded_advisory__light6` outputs tighten the negative-control design by requiring nearly exact subsystem and suffix alignment between security-fix files and ordinary bug-fix controls before comparing opacity.

The `e12_prospective_file_panel__external_holdout_flask_requests_h730_l10_t5` and `e16_external_holdout__curated15_to_external_flask_requests` outputs provide a small but fully external check in which models are frozen on the curated `e12` corpus and evaluated unchanged on Flask and Requests.

Readers who wish to regenerate the results should consult `REPRODUCIBILITY.md`.
