from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch

from bayes_phase.active_learning.active_learner import (
    ActiveLearner,
    ActiveLearnerConfig,
)

parser = ArgumentParser()
parser.add_argument("--config", "-c", type=Path, help="The path to the config file.")
if __name__ == "__main__":
    # Get configuration
    args = parser.parse_args()
    schema = OmegaConf.structured(ActiveLearnerConfig)
    if args.config is not None:
        conf = OmegaConf.load(args.config)
        config = OmegaConf.merge(schema, conf)
    else:
        config = schema

    # Update savepath to include name of acquisition function (for easy bookkeeping)
    base_savepath = Path(config.base_savepath)
    config.base_savepath = str(
        base_savepath.parent / f"{base_savepath.name}_{config.acquisition_function}"
    )
    # Save copy of config
    Path(config.base_savepath).mkdir(exist_ok=True, parents=True)
    with open(Path(config.base_savepath) / "config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)

    # Set random seed(s)
    if config.global_seed is not None:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.global_seed)
        np.random.seed(config.global_seed)

    # Build active learner and run it
    assert isinstance(config, DictConfig)
    learner = ActiveLearner(config)
    learner.optimize()
