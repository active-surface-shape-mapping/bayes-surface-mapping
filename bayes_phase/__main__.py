from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from pathlib import Path

from bayes_phase.active_learning.active_learner import (
    ActiveLearner,
    ActiveLearnerConfig,
)

parser = ArgumentParser()
parser.add_argument("--config", "-c", type=Path, help="The path to the config file.")
if __name__ == "__main__":
    args = parser.parse_args()
    schema = OmegaConf.structured(ActiveLearnerConfig)
    if args.config is not None:
        conf = OmegaConf.load(args.config)
        config = OmegaConf.merge(schema, conf)
    else:
        config = schema
    assert isinstance(config, DictConfig)
    learner = ActiveLearner(config)
    learner.optimize()
