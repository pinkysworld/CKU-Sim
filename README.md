# CKU Simulation Toolkit

CKU Simulation Toolkit (`CKU-Sim`) is the companion codebase for:

> Nguyen, M. (2026). Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms. *International Journal of Research in Computing*, 5(I), 41-56. Retrieved from https://www.ijrcom.org/index.php/ijrc/article/view/192

This repository now supports an audited file-level empirical follow-up centered on external validation rather than repository-level cross-sections alone.

## Current Empirical Package

The primary released empirical package centers on four layers:

1. Measurement validity:
   synthetic replication, compressor robustness, and corpus-wide opacity measurement.
2. Audited prospective prediction:
   frozen file-level training on the curated in-corpus set and audited external replication on disjoint holdout repositories.
3. Direct quantification-failure diagnostics:
   tests of whether high-opacity positive files are more underpredicted and more often missed in the ranking tail.
4. Audited intervention evidence:
   matched pre/post refactoring studies testing whether structural simplification improves later quantification performance.

## Primary Results Included

The strongest paper-facing directories currently included in `data/results/` are:

- `e20_external_replication__expanded7_with_gitea__audited_v1`
- `e24_external_quantification_failure__expanded7_with_gitea__audited_v1`
- `e26_external_intervention_securityfile_enriched__focused4_h1825_v1`

Key checkpoints from those runs:

- Frozen audited external replication:
  9 training repositories, 7 holdout repositories, and 50 holdout positive files.
- External predictive lift:
  `baseline_plus_composite` improved over `baseline_history_plus_structure` on ROC AUC (`0.828 -> 0.836`), average precision (`0.023 -> 0.026`), and slightly on Brier score.
- Direct quantification-failure signal:
  on the refreshed external holdout, high-opacity positive files were more underpredicted and more frequently missed in top-k ranking diagnostics, with non-zero bootstrap gaps.
- Focused external intervention expansion:
  33 accepted audited intervention rows, 30 matched pairs, and 18 positive pair rows across `django-django`, `traefik-traefik`, `prometheus-prometheus`, and `go-gitea-gitea`.
- Intervention-side direction:
  absolute error, underprediction loss, and positive log loss all fell after audited simplification in the focused intervention run; score-range moved in the expected direction overall but remained mixed at one clustering level; the direct failure diagnostics are therefore more decisive than the pooled composite-change proxy.

## Claim Boundary

The repository is strongest on the operational, empirically testable CKU claims:

- higher pre-event structural opacity predicts worse later audited security outcomes than a baseline history-and-structure model alone;
- high-opacity positive files are harder to quantify reliably, not merely higher-risk on average;
- opacity-aware modeling captures externally validated signal that is missed by opacity-blind specifications.

The repository does **not** claim to empirically prove undecidability itself. The theoretical CKU argument remains grounded in the original paper. The follow-up paper should present these results as empirical confirmation of CKU's measurable risk-quantification consequences, not as a stand-alone proof of the underlying computability argument.

## Repository Layout

- `cku_sim/`: metric computation, audit utilities, predictive evaluation, intervention analysis, and plotting logic.
- `experiments/`: runnable entry points for the released experiment families.
- `tests/`: unit tests and regression checks.
- `data/results/`: generated result tables, figures, and experiment outputs included with the repository.
- `data/processed/`: curated audit tables needed for the audited empirical program.
- `data/raw/`: local source-repository clones used to regenerate the results; these are not intended for public release.

## Reproducing The Primary Follow-up

Create the environment:

```bash
git clone https://github.com/pinkysworld/CKU-Sim.git
cd CKU-Sim
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run the paper-facing external package:

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

## Included Audit Tables

The audited empirical program relies on the curated CSVs in `data/processed/`:

- `security_event_file_audit_curated.csv`
- `ordinary_bugfix_control_audit_curated.csv`

These files are part of the released reproducibility package and should be treated as curated research data, not as transient caches.

## Additional Results

The repository also includes the earlier measurement, case-control, negative-control, forward-panel, calibration, and intervention directories that document the progression from exploratory evidence to the current audited external package. Those outputs remain useful for robustness sections, appendices, and methodological traceability.

## Testing

```bash
. .venv/bin/activate
python -m pytest -q
```

## How to Cite

If you use this repository in academic work, please cite the associated paper. Citation metadata is also provided in `CITATION.cff`.

Nguyen, M. (2026). Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms. *International Journal of Research in Computing*, 5(I), 41-56. Retrieved from https://www.ijrcom.org/index.php/ijrc/article/view/192

```bibtex
@article{nguyen2026cku,
  title={Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms},
  author={Nguyen, Mich{\\'e}l},
  journal={International Journal of Research in Computing},
  volume={5},
  number={I},
  pages={41--56},
  year={2026},
  url={https://www.ijrcom.org/index.php/ijrc/article/view/192}
}
```

## License

MIT
