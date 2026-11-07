# Overview
This is the repo for the Bayes-mapping project.
# Installation
```sh
git clone git@github.com: active-surface-shape-mapping/bayes-mapping.git
cd bayes-phase
uv sync
uv pip install -e .
```
# Usage
To run using the [hydra](https://hydra.cc/) experiment config library, e.g.:
```sh
uv run python -m bayes_phase 'experiment=disp_eig'
```
To run multiple experiments at once, e.g.:
```sh
uv run python -m bayes_phase -m 'experiment=disp_eig,disp_random' 'experiment.global_seed=range(0,5)'
```