# Reproducibility

This repository is intended to reproduce the companion empirical follow-up to the CKU paper.

## Environment

- Recommended interpreter: Python 3.11

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

Useful configuration fields:

- `data_dir`: local data root
- `nvd_api_key`: optional NVD API key
- `nvd_rate_limit`: delay between NVD requests
- `osv_rate_limit`: delay between OSV requests
- `osv_query_batch_size`: repository-tag batch size for OSV queries
- `corpus`: repository list and product identifiers
- `monte_carlo_runs`: number of insurance-simulation runs

Optional environment variables:

- `GITHUB_TOKEN`: useful for `e11_large_corpus_builder`

## Data Policy

Included in the public repository:

- experiment source code
- generated result files under `data/results/`
- curated audit tables under `data/processed/`

Excluded from the public repository:

- local raw repository clones under `data/raw/`
- transient cache files such as raw NVD responses, OSV responses, and intermediate hydration artifacts

## Primary Reproduction Path

The primary released empirical package is:

1. audited frozen external replication
2. direct external quantification-failure diagnostics
3. focused audited intervention enrichment

Run those steps as follows:

```bash
python -m experiments.e20_external_replication \
  --train-config experiments/config.forward_panel_curated.yaml \
  --holdout-config experiments/config.external_holdout_expanded.yaml \
  --train-dataset-path data/results/e20_external_replication__expanded7_no_gitea__audited_v1/train_file_level_dataset.parquet \
  --holdout-repos django-django,fastapi-fastapi,prometheus-prometheus,psf-requests,scrapy-scrapy,traefik-traefik,go-gitea-gitea \
  --results-subdir e20_external_replication__expanded7_with_gitea__audited_v1

python -m experiments.e24_external_quantification_failure \
  --e20-subdir e20_external_replication__expanded7_with_gitea__audited_v1 \
  --results-subdir e24_external_quantification_failure__expanded7_with_gitea__audited_v1

python -m experiments.e26_external_intervention_securityfile_enriched \
  --repos django-django,traefik-traefik,prometheus-prometheus,go-gitea-gitea \
  --repo-row-caps django-django=20,traefik-traefik=15,prometheus-prometheus=10,go-gitea-gitea=15 \
  --max-enrichment-rows-per-repo 20 \
  --max-enrichment-rows-per-file 1 \
  --max-control-commits-per-repo 200 \
  --max-control-files-per-commit 1 \
  --results-subdir e26_external_intervention_securityfile_enriched__focused4_h1825_v1
```

## Interpretation Notes

- `e20_external_replication__expanded7_with_gitea__audited_v1` is the primary external predictive package.
- `e24_external_quantification_failure__expanded7_with_gitea__audited_v1` is the primary direct test of the quantification-limits mechanism on positive holdout files.
- `e26_external_intervention_securityfile_enriched__focused4_h1825_v1` is the focused intervention expansion built from deterministic refactoring screens on accepted security-linked files.

## Curated Audit Inputs

The audited empirical program uses:

- `data/processed/security_event_file_audit_curated.csv`
- `data/processed/ordinary_bugfix_control_audit_curated.csv`

These are part of the reproducibility package and should be versioned with care.

## Legacy Runs

Earlier result directories are retained for:

- measurement validity
- negative controls
- prospective panel development
- external holdout development
- robustness appendices

They remain useful for the manuscript, but the paper's core empirical narrative should be anchored in the current audited external package above.

## Verification

```bash
. .venv/bin/activate
python -m pytest -q
```

The test suite is a code and pipeline sanity check, not a substitute for statistical interpretation of the released result directories.
