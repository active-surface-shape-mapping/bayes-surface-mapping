# Overview
This is the repo for the Bayes-phase project (work-in-progress).
# Installation
```sh
git clone git@github.com:Bobby-Huggins/bayes-phase.git
cd bayes-phase
uv sync
uv pip install -e .
```
# Usage
To run using the [hydra](https://hydra.cc/) experiment config library:
```sh
uv run python -m bayes_phase 'experiment=test_eig'
```
To run multiple experiments at once:
```sh
uv run python -m bayes_phase -m 'experiment=test_eig,test_mutual_information_ep' 'experiment.global_seed=range(0,5)'
```