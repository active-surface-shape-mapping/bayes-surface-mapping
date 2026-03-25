import torch
from jaxtyping import Float
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bayes_phase.acquisition.diagram_generation import DiagramOutput
    from bayes_phase.active_learning import ActiveLearner

from bayes_phase.acquisition.conditioning import BaseConditioner


class EPConditioner(BaseConditioner):
    def condition(
        self,
        x_test: Float[torch.Tensor, "n d"],
        diagram_output: "DiagramOutput",
        learner: "ActiveLearner",
    ) -> tuple[Float[torch.Tensor, " n"], dict[str, Any]]:
        """Condition the posterior distribution on phase diagram information, using EP.

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
        peaks, valleys, increasing, decreasing, _ = diagram_output
        assert peaks is not None
        if peaks.numel() == 0:
            raise StopIteration("No peaks in sample.")
        if (increasing is not None) or (decreasing is not None):
            damped_step_size = 0.005
            max_loops = 20
            convergence_threshold = 0.01
        else:
            damped_step_size = 0.05
            max_loops = 50
            convergence_threshold = 0.001

        # Condition on ridges using EP
        with torch.no_grad():
            out = learner.model.pred_condition_1st_and_2nd_deriv(
                x_test=x_test,
                x_ridge=peaks,
                x_inc=increasing,
                x_dec=decreasing,
                verbose=False,
                damped_step_size=damped_step_size,
                max_loops=max_loops,
                convergence_threshold=convergence_threshold,
                return_unconverged=True,
            )
        if out is None:
            # IF EP returned None -> skip
            raise StopIteration("EP returned None.")
        mu_cond, Sigma_cond = out
        m = x_test.shape[0]
        conditional_variance = Sigma_cond[:m, :m].diagonal().clamp_min(1e-30)  # (m,)
        info = {
            "posterior_mean": mu_cond[:m],
            "posterior_covariance": Sigma_cond[:m, :m],
        }
        return conditional_variance, info
