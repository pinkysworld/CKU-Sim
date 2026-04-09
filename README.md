# CKU Simulation Toolkit

CKU Simulation Toolkit (`CKU-Sim`) is the companion codebase for the paper:

> Nguyen, M. (2026). Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms. *International Journal of Research in Computing*, 5(I), 41-56. Retrieved from https://www.ijrcom.org/index.php/ijrc/article/view/192

The repository implements a measurement and simulation workflow for structural opacity as a proxy for computational Knightian uncertainty in software-intensive systems.

## Scope

The codebase supports nine experiment families:

1. Synthetic opacity separation.
2. Structural opacity measurement across a curated software corpus.
3. Correlation analysis between opacity metrics and NVD CVE data.
4. Temporal evolution of opacity over repository history.
5. Monte Carlo cyber insurance simulation under opacity-aware and opacity-blind assumptions.
6. File-level matched case-control analysis for vulnerability-fixing commits.
7. Leave-one-repository-out predictive validation for file-level risk scoring.
8. Ground-truth policy comparison across alternative event definitions.
9. Negative-control comparison between security-fix files and ordinary bug-fix files.

## Repository Contents

This repository is organized for reproducible research publication:

- `cku_sim/`: library code for metrics, collection, analysis, simulation, and plotting.
- `experiments/`: runnable experiment entry points.
- `tests/`: unit tests.
- `data/results/`: generated tables, figures, and experiment outputs included with the repository.
- `data/raw/`: placeholder only in the public repository; large upstream source trees are intentionally excluded.
- `data/processed/`: placeholder only in the public repository; transient caches are intentionally excluded.

## Included Corpus

The current experiment configuration uses a 15-project corpus:

- High-opacity label: `openssl`, `imagemagick`, `libxml2`, `php-src`, `wireshark`
- Low-opacity label: `redis`, `sqlite`, `curl`, `zlib`, `jq`, `libsodium`, `busybox`
- Mixed/control label: `nginx`, `git`, `openssh`

The exact corpus configuration is defined in `experiments/config.example.yaml`.

## Quick Start

```bash
git clone https://github.com/pinkysworld/CKU-Sim.git
cd CKU-Sim
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
cp experiments/config.example.yaml experiments/config.yaml
```

Run the experiments:

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

For the file-level analyses, the event-definition variants are:

- `nvd_commit_refs`: one observation per locally accessible NVD-linked fixing commit.
- `strict_nvd_event`: one primary source-touching fixing commit per NVD-linked vulnerability.
- `balanced_explicit_id_event`: the event-collapsed NVD set augmented with locally explicit `CVE-...` and `GHSA-...` commit identifiers.
- `expanded_advisory_event`: the event-collapsed NVD set augmented with OSV-linked repository advisories and locally explicit `CVE-...` and `GHSA-...` commit identifiers.
- `e09_negative_control_bugfix`: a comparison of security-fix files against ordinary bug-fix files selected under a conservative text filter.

## Results Included In This Repository

The repository ships with generated outputs under `data/results/`:

- `e01_synthetic`: synthetic separation tables and figures.
- `e02_corpus`: corpus-level opacity metrics, robustness analysis, and figures.
- `e03_cve`: merged opacity/CVE dataset, correlation outputs, regression outputs, and figures.
- `e04_temporal`: sampled historical opacity series and aggregate figure.
- `e05_insurance`: simulation summaries, convergence check, quartile analysis, and figures.
- `e06_file_case_control`: matched pre-fix file-level case-control outputs, commit summaries, and figures.
- `e06_file_case_control__strict_nvd_event`: event-collapsed NVD case-control outputs.
- `e06_file_case_control__balanced_explicit_id_event`: expanded explicit-ID case-control outputs.
- `e06_file_case_control__expanded_advisory_event`: advisory-expanded case-control outputs.
- `e07_predictive_validation`: leave-one-repository-out predictive validation outputs and held-out model comparisons.
- `e07_predictive_validation__e06_file_case_control__strict_nvd_event`: predictive validation for the event-collapsed NVD specification.
- `e07_predictive_validation__e06_file_case_control__balanced_explicit_id_event`: predictive validation for the expanded explicit-ID specification.
- `e07_predictive_validation__e06_file_case_control__expanded_advisory_event`: predictive validation for the advisory-expanded specification.
- `e08_policy_comparison`: side-by-side policy comparison tables and figures.
- `e09_negative_control_bugfix`: security-versus-ordinary-bugfix matched comparisons and held-out classification outputs.
- `e09_negative_control_bugfix__e06_file_case_control__expanded_advisory_event`: negative-control outputs using the advisory-expanded security dataset.

## Reproducibility Notes

- Large cloned upstream repositories are not committed.
- NVD responses are not committed because they are transient external data pulls.
- OSV query caches are not committed because they are transient external data pulls.
- The temporal analysis requires full repository history for the selected projects.
- CVE results depend on NVD availability and may change as NVD records evolve.
- The file-level case-control study depends on which NVD entries expose usable fixing-commit references for locally available repository histories.
- The `strict_nvd_event` specification maps each NVD-linked vulnerability to one primary fixing event.
- The `balanced_explicit_id_event` specification augments the NVD-linked event set with locally explicit `CVE-...` and `GHSA-...` identifiers found in commit history.
- The `expanded_advisory_event` specification additionally incorporates OSV-linked repository advisories resolved against local repository tags and histories.
- The predictive validation layer evaluates incremental discrimination beyond a size-only reference model within the curated corpus.
- The negative-control experiment compares security-fix files with ordinary bug-fix files selected under a conservative text filter; the advisory-expanded variant uses the `e06_file_case_control__expanded_advisory_event` security dataset.

Additional procedural details are documented in `REPRODUCIBILITY.md`.

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
