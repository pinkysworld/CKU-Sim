# CKU-Sim

**Toolkit for Quantifying Structural Opacity and Computational Knightian Uncertainty in Software-Intensive Systems**

Companion software for:

> Nguyen, M. H. T. D. (2026). Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms. *International Journal of Research in Computing*, 5(I), 41–56.

## Overview

CKU-Sim operationalises the concept of *structural opacity* — the degree to which a codebase resists compression into regular patterns — as a measurable proxy for Computational Knightian Uncertainty (CKU). It provides:

- **Opacity metrics**: Compressibility index (gzip/LZMA/Zstandard), Shannon entropy, cyclomatic density, Halstead volume, and a configurable composite score
- **Data collectors**: Automated Git repository cloning and NVD/CVE retrieval
- **Simulation engine**: Replication of original CKU paper simulations at scale, plus a novel Monte Carlo cyber insurance model
- **Statistical analysis**: Correlation, regression, classification, and robustness testing
- **Visualisation**: Publication-quality figures and LaTeX-ready tables

## Quick Start

```bash
# Install
git clone https://github.com/pinkysworld/CKU-Sim.git
cd CKU-Sim
pip install -e .

# Configure corpus
cp experiments/config.example.yaml experiments/config.yaml
# Edit config.yaml to select repos and NVD API key

# Run experiments
python -m experiments.e01_synthetic_replication
python -m experiments.e02_real_codebase_survey
python -m experiments.e03_cve_correlation
python -m experiments.e04_temporal_evolution
python -m experiments.e05_insurance_simulation
```

## Corpus

Default corpus of ~25 open-source C/C++ projects spanning the opacity spectrum:

| Category | Projects |
|----------|----------|
| High opacity (expected) | Linux net/, OpenSSL, FFmpeg, PHP, Wireshark, ImageMagick, libxml2, Samba |
| Low opacity (expected) | Redis, SQLite, curl, zlib, musl, busybox, jq, libsodium |
| Mixed / control | CPython, Nginx, PostgreSQL, Git, OpenSSH |

## Project Structure

```
cku_sim/
├── core/          # Configuration, codebase model, opacity dataclass
├── collectors/    # Git and NVD data acquisition
├── metrics/       # Opacity metric implementations
├── simulation/    # Synthetic generation, replication sims, Monte Carlo
├── analysis/      # Statistical tests, regression, classification
└── viz/           # Plotting and LaTeX report generation
experiments/       # Reproducible experiment scripts
tests/             # Unit and integration tests
data/              # Raw, processed, and result data (gitignored)
```

## Citation

If you use CKU-Sim in your research, please cite:

```bibtex
@article{nguyen2026cku,
  title={Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms},
  author={Nguyen, Mich\'{e}l H. T. D.},
  journal={International Journal of Research in Computing},
  volume={5},
  number={I},
  pages={41--56},
  year={2026}
}
```

## License

MIT
