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

The file-level outputs correspond to the following event definitions:

- `nvd_commit_refs`: one observation per locally accessible NVD-linked fixing commit
- `strict_nvd_event`: one primary fixing event per NVD-linked vulnerability
- `balanced_explicit_id_event`: the event-collapsed NVD set augmented with locally explicit `CVE-...` and `GHSA-...` commit identifiers
- `expanded_advisory_event`: the event-collapsed NVD set augmented with OSV-linked repository advisories and locally explicit `CVE-...` and `GHSA-...` commit identifiers

The `e09_negative_control_bugfix` outputs compare security-fix files with ordinary bug-fix files drawn from the same repository histories. The policy-specific `e09_negative_control_bugfix__e06_file_case_control__expanded_advisory_event` outputs use the advisory-expanded `e06` security dataset.

Readers who wish to regenerate the results should consult `REPRODUCIBILITY.md`.
