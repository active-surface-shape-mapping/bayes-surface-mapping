# Overview
This is the repo for the paper, ""Accelerating Materials Discovery with Active Thermodynamic Surface Shape Mapping" at the AI4Mat workshop at NeurIPS 2026.
# Installation
```sh
git clone git@github.com: active-surface-shape-mapping/bayes-surface-mapping.git
cd bayes-mapping
uv sync
uv pip install -e .
```
# Usage
To run using the [hydra](https://hydra.cc/) experiment config library, e.g.:
```sh
uv run python -m bayes_mapping 'experiment=disp_eig'
```
To run multiple experiments at once (seeds for the paper results are 0-29 inclusive):
```sh
uv run python -m bayes_mapping -m 'experiment=disp_gospa_rand_pair_p_w,disp_sobol,disp_eig,disp_random' 'experiment.global_seed=range(0,30)'
```
