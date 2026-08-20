from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any



def default_paths() -> list[str]:
    return [
        "data/111_Cv_run1.arrow",
        "data/111_Cv_run2.arrow",
        "data/111_Cv_run3.arrow",
    ]


def default_sampling_methods() -> list[dict[str, Any]]:
    return [
        {
            "name": "md",
            "cost": 1.0,
            "n_candidates": 300,
            "n_temperatures": 1,
        },
    ]


def full_sampling_methods() -> list[dict[str, Any]]:
    return [
        {
            "name": "md",
            "cost": 1.0,
            "n_candidates": 100,
            "n_temperatures": 1,
        },
        {
            "name": "nested_sampling",
            "cost": 1000.0,
            "temperature_spacing_K_nested_sampling": 25.0,
        },
        {
            "name": "remd",
            "cost": 7.0,
            "n_candidates": 100,
            "n_temperatures_remd": 10,
            "temperature_spacing_K_remd": 10.0,
        },
    ]

def default_diagram_generator_kwargs() -> dict[str, Any]:
    return {
        "peak_threshold": None,
        "max_n_peaks": None,
        "peak_height_rule": None,
        "n_temp_grid": 100,
        "jitter": 1e-12,
        "extrema": "peaks",
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
        "plot3d": 20,
        "plot_acq": 20,
    }

class InitDataStrategy(StrEnum):
    RANDOM = "random"
    SOBOL = "sobol"
    GRID = "grid"
    NS_SLICE = "ns_slice"


class CandDataStrategy(StrEnum):
    RANDOM = "random"
    SOBOL = "sobol"


class MeanModule(StrEnum):
    ZERO = "ZeroMean"
    CONSTANT = "ConstantMean"
    LINEAR = "LinearMean"
    QUADRATIC = "QuadraticMean"


class BaseKernel(StrEnum):
    RBF = "RBFKernel"
    MATERN52 = "Matern52Kernel"


class AcquisitionFunction(StrEnum):
    RANDOM = "random"
    EIG = "eig"
    SOBOL = "sobol"
    CELL = "peak_cell"
    CELL_COST = "peak_cell_cost"
    CELL_BATCH = "peak_cell_batch"
    RANDOM_BATCH = "random_batch"
    RANDOM_SLICE = "random_slice"
    DISPERSION = "dispersion"


class FeatureFinder(StrEnum):
    SLICE_PEAKS = "find_slice_peak_idxs"

class DiagramGenerator(StrEnum):
    FIND_PEAKS = "FindPeaksDiagramGenerator"


class TreeGraphType(StrEnum):
    MST = "MST"
    CURVE = "MSTCurve"


class PeakHeightRule(StrEnum):
    NONE = "none"
    ABOVE_MEDIAN = "above_median"

class DatasetType(StrEnum):
    SIMULATED = "simulated"
    SYNTHETIC = "synthetic"

class DistanceMetric(StrEnum):
    GOSPA = "gospa"


class DispersionMetric(StrEnum):
    RAND_PAIR = "rand_pair"
    PAIRWISE = "pairwise"




@dataclass
class ActiveLearnerConfig:
    global_seed: int | None = None
    n_iters: int = 10
    init_n_points: int = 5
    init_seed: int | None = None 
    iter_n_candidates: int = 300  # number of candidate points per iteration
    n_scoring_samples: int = 50 # number of samples to use for scoring at each iteration
    scoring_grid_T_range: int = 50 # size of scoring box along the T axis, in K
    connect_the_dots: bool = True
    max_tree_conn_dist: float = 0.175 # max proportion of the space across which a spanning tree phase diagram connection will be made for plotting. Only used when connect_the_dots=True.
    tree_graph_type: TreeGraphType = TreeGraphType.CURVE # Type of visuals for tree graphs produced by plot_mst
    show_points_tree_graphs: bool = True # Show scatter of ridge points or not in plots produced by plot_mst. Only used when connect_the_dots=True.
    forbidden_tree_regions: list[dict[str, int | float]] = field(
        default_factory=default_forbidden_tree_regions
    )

    dataset_type: DatasetType = DatasetType.SIMULATED
    # For simulated data:
    init_dataset_paths: list[str] = field(default_factory=default_paths)
    sampling_dataset_paths: list[str] = field(default_factory=default_paths)
    ground_truth_dataset_paths: list[str] = field(default_factory=default_paths)
    # For synthetic data (BayesO Benchmark functions):
    target_name: Optional[str] = "Branin"  # Benchmark target name
    target_scale: float = 1.0  # Multiplicative scale for output values
    target_shift: float = 0.0  # Additive translation for output values
    target_sampling_n_replicates: int = 3  # Number of replicates per evaluation
    target_sampling_noise: float = 0.0  # Noise scale (default is lognormal multiplicative noise)
    target_init_n_replicates: int = 3
    target_init_noise: float = 0.0
    target_ground_truth_n_replicates: int = 3
    target_ground_truth_noise: float = 0.0
    target_kwargs: dict = field(default_factory=dict)  # Additional kwargs forwarded to target

    init_data_strategy: InitDataStrategy = InitDataStrategy.SOBOL
    cand_data_strategy: CandDataStrategy = CandDataStrategy.SOBOL
    sampling_methods: list[dict[str, Any]] = field(
        default_factory=default_sampling_methods
    )
    mean_module: MeanModule = MeanModule.QUADRATIC
    mean_module_kwargs: dict[str, Any] = field(default_factory=dict)
    base_kernel: BaseKernel = BaseKernel.MATERN52
    base_kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    feature_finder: FeatureFinder = FeatureFinder.SLICE_PEAKS
    acquisition_function: AcquisitionFunction = AcquisitionFunction.CELL
    acquisition_n_fantasies: int = 10
    acquisition_fantasy_quadrature: bool = False # Use quadrature instead of Monte Carlo fantasies in the acquisition function.
    acquisition_fantasy_quadrature_points: int = 10
    acquisition_n_mc_samples: int = 100
    acquisition_n_temp_eval: int = 100
    acquisition_sample_jitter: float = 1e-7
    acquisition_slice_window_radius: int = 4
    distance_metric: DistanceMetric = DistanceMetric.GOSPA
    distance_metric_kwargs: dict[str, Any] = field(default_factory=dict)
    dispersion_metric: DispersionMetric = DispersionMetric.RAND_PAIR
    dispersion_metric_kwargs: dict[str, Any] = field(default_factory=dict)
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
    gp_noise_floor: float = 1e-7
    gp_noise_ceiling: float | None = None # Constrain noise to stabilize initial GP estimation with small data
    gp_lengthscale_prior: dict[str, Any] | None = field(
        default_factory=lambda: {
            "name": "LogNormalPrior",
            "args": [-1.5, 0.5],
        }
    )
    gp_outputscale_prior: dict[str, Any] | None = None
    gp_noise_prior: dict[str, Any] | None = None