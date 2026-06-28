from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def default_paths() -> list[str]:
    return [
        "data/111_Cv_run1.arrow",
        "data/111_Cv_run2.arrow",
        "data/111_Cv_run3.arrow",
    ]

def default_diagram_generator_kwargs() -> dict[str, Any]:
    return {
        "peak_threshold": None,
        "max_n_peaks": None,
        "peak_height_rule": None,
    }

# Limit visual connections in phase diagram plots
def default_forbidden_tree_regions() -> list[dict[str, int | float]]:
    return [
        {
            "theta_min": 2,
            "theta_max": 6,
            "temp_max": 200,
        },
        {
            "theta_min": 7,
            "theta_max": 12,
            "temp_max": 350,
        },
        {
            "theta_min": 13,
            "theta_max": 16,
            "temp_max": 400,
        },
    ]


def default_callback_every() -> dict[str, int]:
    return {
        "plot_mst": 20,
        "score_model_plot": 1,
        "plot": 20,
        "plot3d": 50,
        "plot_acq": 50,
        "plot_hull_slices": 50,
    }

class InitDataStrategy(StrEnum):
    RANDOM = "random"
    SOBOL = "sobol"
    MARGINAL_SOBOL = "marginal_sobol"
    GRID = "grid"
    SLICE_ENDS = "slice_ends"


class CandDataStrategy(StrEnum):
    SOBOL = "sobol"
    MARGINAL_SOBOL = "marginal_sobol"
    MARGINAL_POISSON_DISK = "marginal_poisson_disk"


class MeanModule(StrEnum):
    ZERO = "ZeroMean"
    CONSTANT = "ConstantMean"
    LINEAR = "LinearMean"
    QUADRATIC = "QuadraticMean"


class BaseKernel(StrEnum):
    RBF_GRAD = "RBFKernelWithGrad"
    MATERN_GRAD = "Matern52KernelWithGrad"
    RBF = "RBFKernel"
    MATERN52 = "Matern52Kernel"


class AcquisitionFunction(StrEnum):
    RANDOM = "random"
    EIG = "eig"
    MI = "mutual_information"
    SOBOL = "sobol"
    HULL = "hull_variance"
    CELL = "peak_cell"


class DiagramGenerator(StrEnum):
    FIND_PEAKS = "FindPeaksDiagramGenerator"
    FIND_PEAKS_AND_VALLEYS = "FindPeaksAndValleysDiagramGenerator"
    SPLINE = "SplineDiagramGenerator"


class TreeGraphType(StrEnum):
    MST = "MST"
    CURVE = "MSTCurve"


class MIConditioningStrategy(StrEnum):
    EP = "EP"
    ESS = "ESS"
    DO = "DO"


class PeakHeightRule(StrEnum):
    NONE = "none"
    ABOVE_MEDIAN = "above_median"


@dataclass
class ActiveLearnerConfig:
    global_seed: Optional[int] = None
    n_iters: int = 10
    init_n_points: int = 5
    init_seed: int = 42
    iter_n_candidates: int = 300  # number of candidate points per iteration
    n_scoring_samples: int = 50 # number of samples to use for scoring at each iteration
    scoring_grid_T_range: int = 50 # size of scoring box along the T axis, in K
    connect_the_dots: bool = False
    max_tree_conn_dist: float = 0.175 # max proportion of the space across which a spanning tree phase diagram connection will be made for plotting. Only used when connect_the_dots=True.
    tree_graph_type: TreeGraphType = TreeGraphType.CURVE # Type of visuals for tree graphs produced by plot_mst
    show_points_tree_graphs: bool = True # Show scatter of ridge points or not in plots produced by plot_mst. Only used when connect_the_dots=True.
    forbidden_tree_regions: list[dict[str, int | float]] = field(
        default_factory=default_forbidden_tree_regions
    )
    init_dataset_paths: list[str] = field(default_factory=default_paths)
    sampling_dataset_paths: list[str] = field(default_factory=default_paths)
    ground_truth_dataset_paths: list[str] = field(default_factory=default_paths)
    init_data_strategy: InitDataStrategy = InitDataStrategy.SOBOL
    cand_data_strategy: CandDataStrategy = CandDataStrategy.SOBOL
    mean_module: MeanModule = MeanModule.CONSTANT
    mean_module_kwargs: dict[str, Any] = field(default_factory=dict)
    base_kernel: BaseKernel = BaseKernel.MATERN52
    base_kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    acquisition_function: AcquisitionFunction = AcquisitionFunction.CELL
    mi_conditioning_strategy: Optional[MIConditioningStrategy] = None
    mi_n_mc_samples: Optional[int] = None
    hull_n_fantasies: int = 10
    hull_n_mc_samples: int = 100
    hull_n_temp_grid: int = 75
    hull_n_temp_peak_grid: int = 100
    hull_sample_jitter: float = 1e-8
    peak_cell_n_fantasies: int = 10
    peak_cell_n_mc_samples: int = 100
    peak_cell_n_temp_eval: int = 100
    peak_cell_sample_jitter: float = 1e-7
    ess_burnin: Optional[int] = None
    ess_n_samples: Optional[int] = None
    ess_thin: Optional[int] = None
    callbacks: list[str] = field(default_factory=list)
    callback_every: dict[str, int] = field(
        default_factory=default_callback_every
    )
    use_log_transform: bool = True
    show_sampled_locations: bool = True
    base_savepath: str = "${base_savepath:}"
    diagram_generator: DiagramGenerator = DiagramGenerator.FIND_PEAKS
    diagram_generator_kwargs: dict[str, Any] = field(
        default_factory=default_diagram_generator_kwargs
    )
    only_use_peaks: bool = False
    addon_data_noise_std: float = 0.0 # If noise is being added to observations: standard deviation of the noise.
