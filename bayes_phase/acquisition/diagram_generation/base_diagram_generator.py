from abc import ABC, abstractmethod
import torch
from jaxtyping import Float
import gpytorch
from typing import Iterator

DiagramOutput = tuple[Float[torch.Tensor, "n d"], Float[torch.Tensor, "n_temp n_theta"]]


class BaseDiagramGenerator(ABC):
    @abstractmethod
    def __init__(self, model: gpytorch.models.GP, *args, **kwargs):
        self.model = model

    @abstractmethod
    def generate_diagram(
        self,
    ) -> DiagramOutput: ...

    @abstractmethod
    def _update(self): ...

    def diagrams(self, n_diagrams: int, **kwargs) -> Iterator[DiagramOutput]:
        self._update()
        for i in range(n_diagrams):
            yield self.generate_diagram(**kwargs)

    def _normalize_peaks(
        self,
        opt_points: Float[torch.Tensor, "n_points n_opt_starts d"],
        opt_vals: Float[torch.Tensor, "n_points n_opt_starts 1"],
        threshold=0.65,
    ) -> Float[torch.Tensor, "n d"]:
        assert opt_points.shape[0] == opt_vals.shape[0]
        n_theta = opt_points.shape[0]
        all_peaks = []
        for i in range(n_theta):
            best = torch.amax(opt_vals[i])
            mask = opt_vals[i, :, 0] >= threshold * best
            peaks = opt_points[i][mask]
            all_peaks.append(peaks)
        return torch.vstack(all_peaks)

    def _remove_duplicate_points(
        self, points: Float[torch.Tensor, "n 2"], delta: float = 0.005
    ):
        new_points = points[0:1]
        for i in range(1, points.shape[0]):
            next_point = points[i : i + 1]
            if torch.all(torch.cdist(new_points, next_point).amin(0) > delta):
                new_points = torch.cat([new_points, next_point], dim=0)

        return new_points
