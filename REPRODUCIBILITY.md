# Reproducibility

This repository is intended to support reproducible computational results for the associated journal paper.

## Environment

- Recommended interpreter: Python 3.11
- Installation:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Configuration

Start from:

```bash
cp experiments/config.example.yaml experiments/config.yaml
```

Key configuration fields:

- `data_dir`: local data root.
- `nvd_api_key`: optional NVD API key.
- `nvd_rate_limit`: delay between NVD requests.
- `osv_rate_limit`: delay between OSV requests.
- `osv_query_batch_size`: number of repository tags submitted per OSV batch query.
- `corpus`: repository list and product identifiers.
- `monte_carlo_runs`: simulation count for the insurance model.

Environment variables used by optional experiments:

- `GITHUB_TOKEN`: optional GitHub API token for `e11_large_corpus_builder`; recommended to reduce rate-limit pressure during corpus discovery.

## Data Policy

The public repository includes:

- Experiment source code.
- Generated result tables and figures in `data/results/`.

The public repository excludes:

- Local raw repository clones in `data/raw/`.
- Transient caches in `data/processed/`, including NVD and OSV cache files.

This keeps the published repository compact while preserving the outputs referenced in the paper.

## Experiment Order

The intended run order is:

```bash
python -m experiments.e01_synthetic_replication --config experiments/config.yaml
python -m experiments.e02_real_codebase_survey --config experiments/config.yaml
python -m experiments.e03_cve_correlation --config experiments/config.yaml
python -m experiments.e04_temporal_evolution --config experiments/config.yaml
python -m experiments.e05_insurance_simulation --config experiments/config.yaml
python -m experiments.e06_file_level_case_control --config experiments/config.yaml
python -m experiments.e07_predictive_validation --config experiments/config.yaml
python -m experiments.e06_file_level_case_control --config experiments/config.yaml --ground-truth-policy strict_nvd_event
python -m experiments.e07_predictive_validation --config experiments/config.yaml --e06-subdir e06_file_case_control__strict_nvd_event
python -m experiments.e06_file_level_case_control --config experiments/config.yaml --ground-truth-policy balanced_explicit_id_event
python -m experiments.e07_predictive_validation --config experiments/config.yaml --e06-subdir e06_file_case_control__balanced_explicit_id_event
python -m experiments.e06_file_level_case_control --config experiments/config.yaml --ground-truth-policy expanded_advisory_event
python -m experiments.e07_predictive_validation --config experiments/config.yaml --e06-subdir e06_file_case_control__expanded_advisory_event
python -m experiments.e08_policy_comparison --config experiments/config.yaml
python -m experiments.e09_negative_control_bugfix --config experiments/config.yaml
python -m experiments.e09_negative_control_bugfix --config experiments/config.yaml --security-e06-subdir e06_file_case_control__expanded_advisory_event
python -m experiments.e10_forward_release_panel --config experiments/config.yaml --repos openssl,libxml2,curl,redis,sqlite,zlib,git,openssh --max-tags 4 --min-tag-gap-days 180 --horizon-days 365 --results-subdir e10_forward_release_panel__core8
python -m experiments.e10_forward_release_panel --config experiments/config.forward_panel_light8.yaml --max-tags 3 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e10_forward_release_panel__light8_h730_l10
GITHUB_TOKEN=... python -m experiments.e11_large_corpus_builder --config experiments/config.yaml --per-language 24 --min-stars 4000 --min-remote-tags 10 --results-subdir e11_large_corpus__filtered
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e12_prospective_file_panel__curated15_h730_l10_t5
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --ground-truth-policy supported_advisory_plus_explicit --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e12_prospective_file_panel__curated15_h730_l10_t5__supported
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --max-tags 5 --min-tag-gap-days 365 --horizon-days 365 --lookback-years 10 --results-subdir e12_prospective_file_panel__curated15_h365_l10_t5
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --severity-band high_critical --results-subdir e12_prospective_file_panel__curated15_h730_l10_t5__high_critical
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --max-tags 5 --min-tag-gap-days 365 --horizon-days 365 --lookback-years 10 --severity-band high_critical --results-subdir e12_prospective_file_panel__curated15_h365_l10_t5__high_critical
python -m experiments.e13_prospective_label_audit --config experiments/config.forward_panel_curated.yaml --e12-subdir e12_prospective_file_panel__curated15_h730_l10_t5 --audit-input audit_full.csv --sample-size 120 --sampling stratified --stratify-by ground_truth_source --results-subdir e13_prospective_label_audit__curated15_h730_l10_t5__stratified120
python -m experiments.e14_prospective_robustness --config experiments/config.forward_panel_curated.yaml --runs e12_prospective_file_panel__curated15_h365_l10_t5,e12_prospective_file_panel__curated15_h365_l10_t5__high_critical,e12_prospective_file_panel__curated15_h730_l10_t5,e12_prospective_file_panel__curated15_h730_l10_t5__high_critical --results-subdir e14_prospective_robustness__curated15
python -m experiments.e15_negative_control_strict --config experiments/config.yaml --repos libxml2,openssh,sqlite,jq,zlib,nginx --security-e06-subdir e06_file_case_control__expanded_advisory_event --max-bugfix-commits 200 --results-subdir e15_negative_control_strict__expanded_advisory__light6
python -m experiments.e17_bugfix_control_audit --config experiments/config.yaml --e15-subdir e15_negative_control_strict__expanded_advisory__light6 --results-subdir e17_bugfix_control_audit__e15_light6
python -m experiments.e12_prospective_file_panel --config experiments/config.external_holdout.yaml --repos django-django,pallets-flask,psf-requests,fastapi-fastapi,scrapy-scrapy --ground-truth-policy supported_advisory_plus_explicit --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e12_prospective_file_panel__external_python5_h730_l10_t5__supported
python -m experiments.e16_external_holdout --config experiments/config.forward_panel_curated.yaml --train-e12-subdir e12_prospective_file_panel__curated15_h730_l10_t5__supported --holdout-e12-subdir e12_prospective_file_panel__external_python5_h730_l10_t5__supported --results-subdir e16_external_holdout__supported_to_external_python5
python -m experiments.e18_quantification_limits --config experiments/config.forward_panel_curated.yaml --e12-subdir e12_prospective_file_panel__curated15_h730_l10_t5__supported --results-subdir e18_quantification_limits__curated15_h730_l10_t5__supported --n-bootstrap 1000
```

Notes:

- `e03` depends on `e02`.
- `e05` uses `e02` outputs when available.
- `e04` requires full Git histories for the selected temporal projects.
- `e06` requires NVD cache entries with usable GitHub commit references and local repository histories that contain those commits.
- `e07` depends on `e06`.
- `e08` compares multiple `e06`/`e07` policy runs and is only as complete as the upstream policy-specific outputs.
- `e09` depends on an existing `e06` security dataset and on local repository history for the negative-control bug-fix pool.
- `e10` depends on sufficiently deep local Git histories with usable release tags and on OSV records that can be resolved to fixing commits in those local histories.
- `e11` queries the GitHub repository search API and validates candidate repositories by remote tag counts; the generated manifest is intended as an input to later corpus curation rather than as an automatically final study set.
- `e12` depends on the screened forward-panel corpus, sufficiently deep local Git histories with usable release tags, and future advisory events that can be resolved to fixing commits and files in those local histories.
- `e13` depends on an existing `e12` audit sample and is intended to estimate label precision conservatively rather than to certify every event observation in the full panel.
- `e14` depends on completed `e12` runs with comparable matching settings and summarizes horizon/severity sensitivity across those runs.
- `e15` depends on an existing `e06` security dataset and on local history mining for ordinary bug-fix controls under tighter same-subsystem and same-suffix matching rules.
- `e16` depends on a frozen training `e12` dataset and a separately generated external-holdout `e12` dataset built with the same prospective feature construction.
- `e17` depends on an existing `e15` strict negative-control run and screens the matched ordinary bug-fix controls for residual security-related message signals.
- `e18` depends on an existing `e12` prospective panel that already includes leave-one-repository-out held-out predictions for each model specification.

Focused forward-panel notes:

- `experiments/config.forward_panel_curated.yaml` stores the broader screened forward-panel corpus.
- `experiments/config.forward_panel_light8.yaml` stores the focused eight-repository subset used for the completed light-panel run.
- The focused `e10_forward_release_panel__light8_h730_l10` run enforces a fully observed two-year outcome horizon by excluding snapshots that fall within two years of the analysis date.
- The same run also restricts candidate tags to the trailing ten-year window before the horizon cutoff to better align release sampling with modern advisory coverage.
- The prospective `e12_prospective_file_panel__curated15_h730_l10_t5` run uses the same fully observed two-year horizon and ten-year lookback window, but samples up to five release tags per repository to retain intermediate release windows with non-trivial future-event density.
- The `supported_advisory_plus_explicit` prospective policy excludes range-only mappings that are not backed by an explicit identifier or an explicit reference resolution.
- The `e12` prospective study also supports severity-restricted outcomes through `--severity-band high_critical`; this currently retains events with severity information mapped to high or critical labels from NVD or OSV metadata.
- The one-year `e12_prospective_file_panel__curated15_h365_l10_t5` run provides a shorter-horizon sensitivity check using the same release-sampling and matching design.
- The `e13_prospective_label_audit__curated15_h730_l10_t5__stratified120` audit reviews a 120-observation sample from the main `e12` run, stratified by ground-truth source combination, and reports reviewed event-to-commit and file-touch precision by source class.
- The `e14_prospective_robustness__curated15` summary compares one-year versus two-year horizons and all-severity versus high/critical-severity versions of the same prospective file-level design.
- The `e15_negative_control_strict__expanded_advisory__light6` run tightens the negative-control design by requiring same-subsystem matches and nearly always same-suffix matches between security-fix files and ordinary bug-fix controls.
- The `e12_prospective_file_panel__external_python5_h730_l10_t5__supported` plus `e16_external_holdout__supported_to_external_python5` outputs provide a frozen external holdout based on screened repositories that are outside the curated prospective training corpus.
- The `e17_bugfix_control_audit__e15_light6` audit screens the ordinary bug-fix controls used in the strict negative-control comparison.
- The `e18_quantification_limits__curated15_h730_l10_t5__supported` diagnostics use the supported-source `e12` run as input and summarize calibration, forecast error, and cross-model score dispersion by opacity quartile with cluster-bootstrap high-versus-low comparisons.

## Ground-Truth Policy Tiers

The file-level study now supports four event-definition policies:

- `nvd_commit_refs`: each usable NVD-linked fixing commit is treated as one observation.
- `strict_nvd_event`: each NVD-linked vulnerability is mapped to one primary single-parent source-touching fix.
- `balanced_explicit_id_event`: the event-collapsed NVD set is augmented with locally explicit `CVE-...` and `GHSA-...` commit identifiers.
- `expanded_advisory_event`: the event-collapsed NVD set is augmented with OSV-linked repository advisories and locally explicit `CVE-...` and `GHSA-...` commit identifiers.

## Negative-Control Design

Experiment `e09_negative_control_bugfix` compares security-fix files with ordinary bug-fix files drawn from the same repository histories.

- Security examples come from an existing `e06` security-fix dataset.
- Ordinary bug-fix controls are collected conservatively from local Git history using bug-fix subject keywords, while excluding commits that look security-related.
- Matching is done within repository and prefers similar file suffix and file size.

## External Dependencies And Stability

- NVD-backed results depend on the state and availability of the NVD API at run time.
- OSV-backed results depend on the state and availability of the OSV API at run time and on local repository tag coverage.
- GitHub-discovery outputs depend on the state of the GitHub repository search API, token rate limits, and the evolving star counts, topics, and tag structures of public repositories.
- Repository histories may evolve after publication; exact reruns should pin commits if strict archival replication is required.
- Temporal outputs depend on the local clone depth and branch state.

## Verification

Run:

```bash
. .venv/bin/activate
python -m pytest -q
```

The included test suite is intended as a code sanity check, not as a full statistical validation harness.
