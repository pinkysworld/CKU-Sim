# CKU Simulation Toolkit

CKU Simulation Toolkit (`CKU-Sim`) is the companion codebase for the paper:

> Nguyen, M. (2026). Computational Knightian Uncertainty: Undecidability and the Limits of Cyber Risk Quantification in Software-Intensive Firms. *International Journal of Research in Computing*, 5(I), 41-56. Retrieved from https://www.ijrcom.org/index.php/ijrc/article/view/192

The repository implements a measurement and simulation workflow for structural opacity as a proxy for computational Knightian uncertainty in software-intensive systems.

## Scope

The codebase supports seventeen experiment families:

1. Synthetic opacity separation.
2. Structural opacity measurement across a curated software corpus.
3. Correlation analysis between opacity metrics and NVD CVE data.
4. Temporal evolution of opacity over repository history.
5. Monte Carlo cyber insurance simulation under opacity-aware and opacity-blind assumptions.
6. File-level matched case-control analysis for vulnerability-fixing commits.
7. Leave-one-repository-out predictive validation for file-level risk scoring.
8. Ground-truth policy comparison across alternative event definitions.
9. Negative-control comparison between security-fix files and ordinary bug-fix files.
10. Forward-looking release-level panel analysis using pre-event opacity and later advisory outcomes.
11. Larger external corpus discovery and manifest construction for expanded follow-on studies.
12. Prospective file-level panel analysis using pre-release file opacity and later security-fix involvement.
13. Reviewed label-audit summaries for the prospective file-level panel.
14. Horizon and severity robustness summaries for the prospective file-level panel.
15. Strict negative-control comparison between security-fix files and tightly matched ordinary bug-fix files.
16. Frozen external-holdout validation using an out-of-corpus prospective file-level panel.
17. Screening audit for the ordinary bug-fix controls used in the strict negative-control comparison.

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
python -m experiments.e10_forward_release_panel --config experiments/config.yaml --repos openssl,libxml2,curl,redis,sqlite,zlib,git,openssh --max-tags 4 --min-tag-gap-days 180 --horizon-days 365 --results-subdir e10_forward_release_panel__core8
python -m experiments.e10_forward_release_panel --config experiments/config.forward_panel_light8.yaml --max-tags 3 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e10_forward_release_panel__light8_h730_l10
GITHUB_TOKEN=... python -m experiments.e11_large_corpus_builder --config experiments/config.yaml --per-language 24 --min-stars 4000 --min-remote-tags 10 --results-subdir e11_large_corpus__filtered
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e12_prospective_file_panel__curated15_h730_l10_t5
python -m experiments.e12_prospective_file_panel --config experiments/config.forward_panel_curated.yaml --ground-truth-policy supported_advisory_plus_explicit --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e12_prospective_file_panel__curated15_h730_l10_t5__supported
python -m experiments.e13_prospective_label_audit --config experiments/config.forward_panel_curated.yaml --e12-subdir e12_prospective_file_panel__curated15_h730_l10_t5 --audit-input audit_full.csv --sample-size 120 --sampling stratified --stratify-by ground_truth_source --results-subdir e13_prospective_label_audit__curated15_h730_l10_t5__stratified120
python -m experiments.e14_prospective_robustness --config experiments/config.forward_panel_curated.yaml --runs e12_prospective_file_panel__curated15_h365_l10_t5,e12_prospective_file_panel__curated15_h365_l10_t5__high_critical,e12_prospective_file_panel__curated15_h730_l10_t5,e12_prospective_file_panel__curated15_h730_l10_t5__high_critical --results-subdir e14_prospective_robustness__curated15
python -m experiments.e15_negative_control_strict --config experiments/config.yaml --repos libxml2,openssh,sqlite,jq,zlib,nginx --security-e06-subdir e06_file_case_control__expanded_advisory_event --max-bugfix-commits 200 --results-subdir e15_negative_control_strict__expanded_advisory__light6
python -m experiments.e17_bugfix_control_audit --config experiments/config.yaml --e15-subdir e15_negative_control_strict__expanded_advisory__light6 --results-subdir e17_bugfix_control_audit__e15_light6
python -m experiments.e12_prospective_file_panel --config experiments/config.external_holdout.yaml --repos django-django,pallets-flask,psf-requests,fastapi-fastapi,scrapy-scrapy --ground-truth-policy supported_advisory_plus_explicit --max-tags 5 --min-tag-gap-days 365 --horizon-days 730 --lookback-years 10 --results-subdir e12_prospective_file_panel__external_python5_h730_l10_t5__supported
python -m experiments.e16_external_holdout --config experiments/config.forward_panel_curated.yaml --train-e12-subdir e12_prospective_file_panel__curated15_h730_l10_t5__supported --holdout-e12-subdir e12_prospective_file_panel__external_python5_h730_l10_t5__supported --results-subdir e16_external_holdout__supported_to_external_python5
```

For the file-level analyses, the event-definition variants are:

- `nvd_commit_refs`: one observation per locally accessible NVD-linked fixing commit.
- `strict_nvd_event`: one primary source-touching fixing commit per NVD-linked vulnerability.
- `balanced_explicit_id_event`: the event-collapsed NVD set augmented with locally explicit `CVE-...` and `GHSA-...` commit identifiers.
- `expanded_advisory_event`: the event-collapsed NVD set augmented with OSV-linked repository advisories and locally explicit `CVE-...` and `GHSA-...` commit identifiers.
- `supported_advisory_plus_explicit`: the prospective event set excluding range-only mappings that lack explicit identifier or reference support.
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
- `e10_forward_release_panel__core8`: forward-looking release-panel outputs for the current tag-rich core subset.
- `e10_forward_corpus_screen__curated15_h730_l10`: screening table for the expanded forward-looking candidate set.
- `e10_forward_release_panel__light8_h730_l10`: completed forward-looking panel for the focused eight-repository subset with full-horizon observation and a ten-year lookback window.
- `e11_large_corpus__filtered`: GitHub-discovered expanded-corpus candidate list, validated subset, and manifest.
- `e12_prospective_file_panel__light8_h730_l10`: prospective file-level panel outputs for the focused eight-repository subset.
- `e12_prospective_file_panel__curated15_h730_l10`: prospective file-level panel outputs for the screened 15-repository subset with two sampled tags per repository.
- `e12_prospective_file_panel__curated15_h730_l10_t5`: prospective file-level panel outputs for the screened 15-repository subset with five sampled tags per repository.
- `e12_prospective_file_panel__curated15_h365_l10_t5`: prospective file-level panel outputs for the screened 15-repository subset at a one-year horizon.
- `e12_prospective_file_panel__curated15_h365_l10_t5__high_critical`: one-year prospective panel restricted to high/critical-severity future events.
- `e12_prospective_file_panel__curated15_h730_l10_t5__high_critical`: two-year prospective panel restricted to high/critical-severity future events.
- `e12_prospective_file_panel__curated15_h730_l10_t5__supported`: prospective panel using only explicit or reference-supported future-event mappings.
- `e12_prospective_file_panel__external_python5_h730_l10_t5__supported`: external prospective holdout across the screened Python subset under the supported-source policy.
- `e13_prospective_label_audit__curated15_h730_l10_t5`: reviewed audit sample and audit-summary tables for the main prospective panel.
- `e13_prospective_label_audit__curated15_h730_l10_t5__stratified120`: larger stratified audit sample and source-stratified review summaries for the main prospective panel.
- `e14_prospective_robustness__curated15`: side-by-side horizon/severity robustness summary for the prospective panel.
- `e15_negative_control_strict__expanded_advisory__light6`: stricter same-subsystem security-versus-bugfix matched comparison on the lighter six-repository subset.
- `e12_prospective_file_panel__external_holdout_flask_requests_h730_l10_t5`: frozen-spec prospective file-level holdout panel for the external Flask and Requests subset.
- `e16_external_holdout__curated15_to_external_flask_requests`: frozen train/test validation from the curated prospective panel to the external Flask/Requests holdout.
- `e16_external_holdout__supported_to_external_python5`: frozen train/test validation from the supported-source prospective panel to the screened external Python holdout.
- `e17_bugfix_control_audit__e15_light6`: screening audit for the ordinary bug-fix controls used in the strict negative-control run.

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
- The `supported_advisory_plus_explicit` prospective specification excludes range-only mappings and retains only events with explicit identifier or reference support.
- The predictive validation layer evaluates incremental discrimination beyond a size-only reference model within the curated corpus.
- The negative-control experiment compares security-fix files with ordinary bug-fix files selected under a conservative text filter; the advisory-expanded variant uses the `e06_file_case_control__expanded_advisory_event` security dataset.
- The forward-looking panel currently uses release tags as snapshot units and OSV-linked fixing events as later outcomes; its power depends heavily on tag coverage and advisory density in local histories.
- The focused `e10_forward_release_panel__light8_h730_l10` run uses only fully observed two-year horizons and restricts snapshot sampling to the trailing ten-year window before the horizon cutoff.
- The prospective `e12` study uses release snapshots as pre-event file baselines, labels future file involvement from advisory-linked fixing events, and compares matched future-case files against untouched controls from the same snapshot.
- The denser `e12_prospective_file_panel__curated15_h730_l10_t5` run uses up to five sampled release tags per repository within the fully observed ten-year window in order to recover intermediate release windows with usable future-event density.
- The stratified `e13_prospective_label_audit__curated15_h730_l10_t5__stratified120` audit reviews a 120-observation sample from the main `e12` run and separates stronger explicit/reference-backed labels from `osv_range`-only mappings.
- The `e14` robustness summary compares one-year versus two-year horizons and all-severity versus high/critical-severity specifications using the same prospective file-level design.
- The strict `e15_negative_control_strict__expanded_advisory__light6` design requires same-subsystem and almost always same-suffix matches between security-fix files and ordinary bug-fix controls.
- The `e16_external_holdout__curated15_to_external_flask_requests` outputs freeze models on the curated `e12` corpus and score them unchanged on an external two-repository holdout.
- The `e17_bugfix_control_audit__e15_light6` outputs screen the ordinary bug-fix controls used in the strict negative-control analysis for residual security-related message signals.
- The large-corpus builder is a reproducible discovery tool for follow-on studies, not a substitute for final substantive curation of a publication corpus.

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
