# CKU-Sim

CKU-Sim is the companion codebase for the paper:

> Nguyen, M. H. T. D. (2026). Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms. *International Journal of Research in Computing*, 5(I), 41-56.

The repository implements a measurement and simulation workflow for structural opacity as a proxy for computational Knightian uncertainty in software-intensive systems.

## Scope

The codebase supports five experiment families:

1. Synthetic opacity separation.
2. Structural opacity measurement across a curated software corpus.
3. Correlation analysis between opacity metrics and NVD CVE data.
4. Temporal evolution of opacity over repository history.
5. Monte Carlo cyber insurance simulation under opacity-aware and opacity-blind assumptions.

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
```

## Results Included In This Repository

The repository ships with generated outputs under `data/results/`:

- `e01_synthetic`: synthetic separation tables and figures.
- `e02_corpus`: corpus-level opacity metrics, robustness analysis, and figures.
- `e03_cve`: merged opacity/CVE dataset, correlation outputs, regression outputs, and figures.
- `e04_temporal`: sampled historical opacity series and aggregate figure.
- `e05_insurance`: simulation summaries, convergence check, quartile analysis, and figures.

## Reproducibility Notes

- Large cloned upstream repositories are not committed.
- NVD responses are not committed because they are transient external data pulls.
- The temporal analysis requires full repository history for the selected projects.
- CVE results depend on NVD availability and may change as NVD records evolve.

Additional procedural details are documented in `REPRODUCIBILITY.md`.

## Testing

```bash
. .venv/bin/activate
python -m pytest -q
```

## Citation

If you use this repository in academic work, please cite the associated paper. Citation metadata is also provided in `CITATION.cff`.

```bibtex
@article{nguyen2026cku,
  title={Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms},
  author={Nguyen, Mich{\\'e}l H. T. D.},
  journal={International Journal of Research in Computing},
  volume={5},
  number={I},
  pages={41--56},
  year={2026}
}
```

## License

MIT
