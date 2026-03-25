from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import hydra
import random
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from bayes_phase.active_learning.active_learner import (
    ActiveLearner,
    ActiveLearnerConfig,
)
def get_save_path() -> Path:
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    else:
        date = datetime.now()
        save_path = Path(f"./runs/{date:%Y-%m-%d_%H:%M:%S}")
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path
    

@hydra.main(version_base=None, config_path="./active_learning/config", config_name="base_config")
def run_hydra(cfg: DictConfig):
    schema = OmegaConf.structured(ActiveLearnerConfig)
    config: ActiveLearnerConfig = OmegaConf.merge(schema, cfg.experiment)
    print(config)

    # Save copy of config
    Path(config.base_savepath).mkdir(exist_ok=True, parents=True)
    with open(Path(config.base_savepath) / "config.yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)

    # Set random seed(s)
    if config.global_seed is not None:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.global_seed)
        random.seed(cfg.experiment.global_seed)
        np.random.seed(config.global_seed)

    # Build active learner and run it
    assert isinstance(config, DictConfig)
    learner = ActiveLearner(config)
    learner.optimize()
        
if __name__ == "__main__":
    OmegaConf.register_new_resolver(
        "base_savepath", get_save_path, use_cache=False, 
    )
    run_hydra()
