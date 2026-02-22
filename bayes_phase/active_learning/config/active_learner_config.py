from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def timestamp_path() -> Path:
    now = datetime.now().strftime("%y-%m-%d-%H:%M:%S.%f")
    return PROJECT_ROOT / "runs" / now


def default_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "data/111_Cv_run1.arrow",
        PROJECT_ROOT / "data/111_Cv_run2.arrow",
        PROJECT_ROOT / "data/111_Cv_run3.arrow",
    ]


class InitDataStrategy(StrEnum):
    SOBOL = "sobol"
    MARGINAL_SOBOL = "marginal_sobol"


class MeanModule(StrEnum):
    CONSTANT = "ConstantMean"
    ZERO = "ZeroMean"


class BaseKernel(StrEnum):
    RBF = "RBFKernelWithGrad"
    MATERN = "MaternKernelWithGrad"


class AcquisitionFunction(StrEnum):
    RANDOM = "random"
    MI = "mutual_information"


@dataclass
class ActiveLearnerConfig:
    n_iters: int = 10
    init_n_points: int = 5
    init_seed: int = 42
    iter_n_candidates: int = 300 # number of candidate points per iteration
    n_mc_samples: int = 20 # number of MC samples for acquisition function
    dataset_paths: list[Path] = field(default_factory=default_paths)
    init_data_strategy: InitDataStrategy = InitDataStrategy.SOBOL
    mean_module: MeanModule = MeanModule.CONSTANT
    mean_module_kwargs: dict[str, Any] = field(default_factory=dict)
    base_kernel: BaseKernel = BaseKernel.MATERN
    base_kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    acquisition_function: AcquisitionFunction = AcquisitionFunction.MI
    callbacks: list[str] = field(default_factory=list)
    use_log_transform: bool = True
    base_savepath: Path = field(default_factory=timestamp_path)
