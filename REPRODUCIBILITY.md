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
- `corpus`: repository list and product identifiers.
- `monte_carlo_runs`: simulation count for the insurance model.

## Data Policy

The public repository includes:

- Experiment source code.
- Generated result tables and figures in `data/results/`.

The public repository excludes:

- Local raw repository clones in `data/raw/`.
- Transient caches in `data/processed/`, including NVD cache files.

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
```

Notes:

- `e03` depends on `e02`.
- `e05` uses `e02` outputs when available.
- `e04` requires full Git histories for the selected temporal projects.
- `e06` requires NVD cache entries with usable GitHub commit references and local repository histories that contain those commits.

## External Dependencies And Stability

- NVD-backed results depend on the state and availability of the NVD API at run time.
- Repository histories may evolve after publication; exact reruns should pin commits if strict archival replication is required.
- Temporal outputs depend on the local clone depth and branch state.

## Verification

Run:

```bash
. .venv/bin/activate
python -m pytest -q
```

The included test suite is intended as a code sanity check, not as a full statistical validation harness.
