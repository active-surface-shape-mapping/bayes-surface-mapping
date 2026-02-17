from typing import Sequence, Literal
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class InitializationConfig:
    n_points: int
    seed: int = 42
    strategy: Literal["Sobol"] = "Sobol"


@dataclass
class ActiveLearnerConfig:
    n_iters: int
    init: InitializationConfig
    dataset_paths: Sequence[Path] = [
        PROJECT_ROOT / "../data/111_Cv_run1.arrow",
        PROJECT_ROOT / "../data/111_Cv_run2.arrow",
        PROJECT_ROOT / "../data/111_Cv_run3.arrow",
    ]
    init_data_strategy: Literal["Sobol"] = "Sobol"
