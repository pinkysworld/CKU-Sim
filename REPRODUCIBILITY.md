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
```

Notes:

- `e03` depends on `e02`.
- `e05` uses `e02` outputs when available.
- `e04` requires full Git histories for the selected temporal projects.
- `e06` requires NVD cache entries with usable GitHub commit references and local repository histories that contain those commits.
- `e07` depends on `e06`.
- `e08` compares multiple `e06`/`e07` policy runs and is only as complete as the upstream policy-specific outputs.
- `e09` depends on an existing `e06` security dataset and on local repository history for the negative-control bug-fix pool.

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
- Repository histories may evolve after publication; exact reruns should pin commits if strict archival replication is required.
- Temporal outputs depend on the local clone depth and branch state.

## Verification

Run:

```bash
. .venv/bin/activate
python -m pytest -q
```

The included test suite is intended as a code sanity check, not as a full statistical validation harness.
