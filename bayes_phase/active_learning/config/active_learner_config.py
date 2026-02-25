from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def timestamp_path() -> str:
    now = datetime.now().strftime("%y-%m-%d-%H:%M:%S.%f")
    return f"./runs/{now}/"


def default_paths() -> list[str]:
    return [
        "data/111_Cv_run1.arrow",
        "data/111_Cv_run2.arrow",
        "data/111_Cv_run3.arrow",
    ]


class InitDataStrategy(StrEnum):
    SOBOL = "sobol"
    MARGINAL_SOBOL = "marginal_sobol"
    GRID = "grid"


class CandDataStrategy(StrEnum):
    SOBOL = "sobol"
    MARGINAL_SOBOL = "marginal_sobol"


class MeanModule(StrEnum):
    ZERO = "ZeroMean"
    CONSTANT = "ConstantMean"
    LINEAR = "LinearMean"
    QUADRATIC = "QuadraticMean"


class BaseKernel(StrEnum):
    RBF = "RBFKernelWithGrad"
    MATERN = "Matern52KernelWithGrad"


class AcquisitionFunction(StrEnum):
    RANDOM = "random"
    EIG = "eig"
    MI = "mutual_information"


class DiagramGenerator(StrEnum):
    FIND_PEAKS = "FindPeaksDiagramGenerator"
    SPLINE = "SplineDiagramGenerator"


@dataclass
class ActiveLearnerConfig:
    global_seed: Optional[int] = None
    n_iters: int = 10
    init_n_points: int = 5
    init_seed: int = 42
    iter_n_candidates: int = 300  # number of candidate points per iteration
    n_mc_samples: int = 200  # number of MC samples for MI acquisition function
    dataset_paths: list[str] = field(default_factory=default_paths)
    init_data_strategy: InitDataStrategy = InitDataStrategy.SOBOL
    cand_data_strategy: CandDataStrategy = CandDataStrategy.SOBOL
    mean_module: MeanModule = MeanModule.CONSTANT
    mean_module_kwargs: dict[str, Any] = field(default_factory=dict)
    base_kernel: BaseKernel = BaseKernel.MATERN
    base_kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    acquisition_function: AcquisitionFunction = AcquisitionFunction.MI
    callbacks: list[str] = field(default_factory=list)
    use_log_transform: bool = True
    base_savepath: str = field(default_factory=timestamp_path)
    diagram_generator: DiagramGenerator = DiagramGenerator.FIND_PEAKS
    diagram_generator_kwargs: dict[str, Any] = field(default_factory=dict)
