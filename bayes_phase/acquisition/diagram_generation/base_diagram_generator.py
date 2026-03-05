from abc import ABC, abstractmethod
import torch
from jaxtyping import Float
import gpytorch
from typing import Iterator

DiagramOutput = tuple[Float[torch.Tensor, "n d"], Float[torch.Tensor, "n_temp n_theta"]]
"""A two-element tuple where the first element is a collection of points
representing the phase diagram boundary, and the second element is the GP sample
path corresponding to that boundary."""


class BaseDiagramGenerator(ABC):
    @abstractmethod
    def __init__(self, model: gpytorch.models.GP, *args, **kwargs):
        """Initialize an object for generating samples of phase diagram boundaries.

        Parameters
        ----------
        model
            The surrogate model from which to draw sample paths.
        """
        self.model = model

    @abstractmethod
    def generate_diagram(
        self,
    ) -> DiagramOutput:
        """Generate a single sample of the phase diagram boundary.

        Returns
        -------
        diagram_output: DiagramOutput
            A two-element tuple where the first element is a collection of
            points representing the phase diagram boundary, and the second
            element is the GP sample path corresponding to that boundary.
        """
        ...

    @abstractmethod
    def _update(self):
        """Update any internal variables as necessary per batch of diagrams.

        These variables could be, e.g., the probability distribution for
        generating sample paths. If no variables need to be updated, this
        function may be a no-op."""
        ...

    def diagrams(self, n_diagrams: int, **kwargs) -> Iterator[DiagramOutput]:
        """Get a lazy iterator over single phase diagram boundaries.

        Parameters
        ----------
        n_diagrams
            The number of diagrams to generate.

        Returns
        -------
        diagrams
            An iterator over `(diagram, sample_path)` pairs, of length
            `n_diagrams`.
        """
        self._update()
        for i in range(n_diagrams):
            yield self.generate_diagram(**kwargs)

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
