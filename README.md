# Overview
This is the repo for the Bayes-phase project (work-in-progress).
## TODOs:
### Current:
  1. Implement mutual-information acquisition function.
     1. Write tests for acquisition function.
  2. Implement active learning loop.
  3. Basic shakedown run with synthetic data.
  4. Compare different approaches to generating phase diagram (optimization vs. thresholding of GP partial derivatives).
### Eventual:
  1. Flesh out comments and docstrings.
### Potential:
  1. Use ``gpytorch.kernels.RBFGradGrad`` in batch-mode for faster computation of posterior variance.