from abc import ABC, abstractmethod
import torch
from jaxtyping import Float
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bayes_phase.acquisition.diagram_generation import DiagramOutput
    from bayes_phase.active_learning import ActiveLearner


class BaseConditioner(ABC):
    @abstractmethod
    def condition(
        self,
        x_test: Float[torch.Tensor, "n d"],
        diagram_output: "DiagramOutput",
        learner: "ActiveLearner",
    ) -> tuple[Float[torch.Tensor, " n"], dict[str, Any]]:
        """Condition the posterior distribution on phase diagram information.

        Parameters
        ----------
        diagram_output: Diagram Ouput
            The phase diagram information.
        learner: ActiveLearner
            The active learning context object.

        Returns
        -------
        conditional_variance, info
            The posterior variance at `x_test`, conditioned on the diagram
            ouput, plus a dictionary of info for debugging.

        Raises
        ------
        StopIteration
            If there is an issue with conditioning.
        """
        ...
