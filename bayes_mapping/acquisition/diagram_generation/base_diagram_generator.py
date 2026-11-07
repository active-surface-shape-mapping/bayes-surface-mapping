from abc import ABC, abstractmethod
import torch
from jaxtyping import Float
from typing import NamedTuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bayes_mapping.active_learning import ActiveLearner


class DiagramOutput(NamedTuple):
    """Class containing potentially relevant information for conditioning on the phase boundary.

    - `peaks` are coordinates of local energy maxima.
    - `valleys` are coordinates of local energy minima.
    - `increasing` are coordinates of increasing energy.
    - `decreasing` are coordinates of decreasing energy.
    - `sample` are energy values of the corresponding posterior sample, at grid
        locations (for debugging).
    """

    peaks: Optional[Float[torch.Tensor, "n d"]]
    valleys: Optional[Float[torch.Tensor, "n d"]]
    increasing: Optional[Float[torch.Tensor, "n d"]]
    decreasing: Optional[Float[torch.Tensor, "n d"]]
    sample: Optional[Float[torch.Tensor, "n_temp n_theta"]]


class BaseDiagramGenerator(ABC):
    @abstractmethod
    def __init__(self, learner: "ActiveLearner", *args, **kwargs):
        """Initialize an object for generating samples of phase diagram boundaries.

        Parameters
        ----------
        learner
            The active learning object, for context.
        """
        self.learner = learner

    @abstractmethod
    def generate_diagram(
        self,
    ) -> DiagramOutput:
        """Generate a single sample of the phase diagram boundary.

        Returns
        -------
        diagram_output: DiagramOutput
        """
        ...

    @abstractmethod
    def _update(self):
        """Update any internal variables as necessary per batch of diagrams.

        These variables could be, e.g., the probability distribution for
        generating sample paths. If no variables need to be updated, this
        function may be a no-op."""
        ...

    def diagrams(self, n_diagrams: int, **kwargs) -> list[DiagramOutput]:
        """Get a list of phase diagram boundaries.

        Parameters
        ----------
        n_diagrams
            The number of diagrams to generate.

        Returns
        -------
        diagrams
            A list of `DiagramOutputs`.
        """
        self._update()
        return [self.generate_diagram(**kwargs) for _ in range(n_diagrams)]

    def _remove_duplicate_points(
        self, points: Float[torch.Tensor, "n 2"], delta: float = 0.005
    ):
        """Given a set of points (e.g., representing a diagram boundary), remove
        points so that none are within distance `delta` of eachother.

        Parameters
        points
            The coordinates of the points to deduplicate.
        delta
            The minimum l2 distance of the resulting tensor of points.

        Returns
        -------
        deduplicated_points
            The resulting points, where some points may have been removed so
            that no two points are within `delta` l2 distance from one another.
        """
        new_points = points[0:1]
        for i in range(1, points.shape[0]):
            next_point = points[i : i + 1]
            if torch.all(torch.cdist(new_points, next_point).amin(0) > delta):
                new_points = torch.cat([new_points, next_point], dim=0)

        return new_points
