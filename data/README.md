This repository includes generated experiment outputs in `data/results/`.

The public repository intentionally excludes:

- `data/raw/`: large cloned upstream source repositories
- `data/processed/`: transient caches such as NVD responses

Current contents:

- `data/results/e01_synthetic/`: synthetic separation outputs
- `data/results/e02_corpus/`: corpus-level opacity measurements
- `data/results/e03_cve/`: opacity/CVE merged outputs and figures
- `data/results/e04_temporal/`: temporal opacity series and aggregate figure
- `data/results/e05_insurance/`: Monte Carlo insurance outputs
- `data/results/e06_file_case_control/`: matched file-level case-control outputs and figures

Readers who wish to regenerate the results should consult `REPRODUCIBILITY.md`.
