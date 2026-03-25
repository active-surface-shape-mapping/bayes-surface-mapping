import torch
from jaxtyping import Float
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bayes_phase.acquisition.diagram_generation import DiagramOutput
    from bayes_phase.active_learning import ActiveLearner

from bayes_phase.acquisition.conditioning import BaseConditioner
from bayes_phase.models.means import QuadraticMean
from bayes_phase.utils.ess import ElipticalSliceSampler


class ESSConditioner(BaseConditioner):
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
            raise StopIteration
        assert valleys is not None
        assert increasing is not None
        assert decreasing is not None

        # Condition on ridges using ESS
        with torch.no_grad():
            # Compute prior mean and covariance
            points = torch.vstack((x_test, peaks, valleys, increasing, decreasing))
            n_points = points.shape[0]
            mean, covar = learner.model.predict_with_grad(points)
            assert covar is not None
            # Build indices
            peaks_start = x_test.shape[0]
            peaks_end = peaks_start + peaks.shape[0]
            peaks_idxs = torch.arange(peaks_start, peaks_end)
            valleys_start = peaks_end
            valleys_end = valleys_start + valleys.shape[0]
            valleys_idxs = torch.arange(valleys_start, valleys_end)
            increasing_start = valleys_end
            increasing_end = increasing_start + increasing.shape[0]
            increasing_idxs = torch.arange(increasing_start, increasing_end)
            decreasing_start = increasing_end
            decreasing_end = decreasing_start + decreasing.shape[0]
            decreasing_idxs = torch.arange(decreasing_start, decreasing_end)
            assert decreasing_end == n_points
            if isinstance(learner.model.mean_module, QuadraticMean):
                # Adjust derivatives for non-constant mean
                prior_derivs = learner.model.mean_module.derivatives(points)
            else:
                prior_derivs = torch.zeros((2, n_points))

            def log_likelihood(
                preds: Float[torch.Tensor, "n_points"],
                k: float = 20.0,
                b: float = 1.0,
                p: float = 1.0,
            ) -> Float[torch.Tensor, ""]:
                # Find first derivatives which are incorrect:
                first_derivs_inc = (
                    preds[increasing_idxs + n_points] + prior_derivs[0][increasing_idxs]
                )
                first_derivs_dec = (
                    preds[decreasing_idxs + n_points] + prior_derivs[0][decreasing_idxs]
                )
                wrong_incs = first_derivs_inc[first_derivs_inc < 0.0]
                wrong_decs = first_derivs_dec[first_derivs_dec > 0.0]
                # Find second derivatives which are incorrect:
                first_derivs_peaks = (
                    preds[peaks_idxs + n_points] + prior_derivs[0][peaks_idxs]
                )
                second_derivs_peaks = (
                    preds[peaks_idxs + n_points * 2] + prior_derivs[1][peaks_idxs]
                )
                first_derivs_valleys = (
                    preds[valleys_idxs + n_points] + prior_derivs[0][valleys_idxs]
                )
                second_derivs_valleys = (
                    preds[valleys_idxs + n_points * 2] + prior_derivs[1][valleys_idxs]
                )
                wrong_peaks_1 = first_derivs_peaks[first_derivs_peaks != 0]
                wrong_valleys_1 = first_derivs_valleys[first_derivs_valleys != 0]
                wrong_peaks_2 = second_derivs_peaks[second_derivs_peaks >= 0.0]
                wrong_valleys_2 = second_derivs_valleys[second_derivs_valleys <= 0.0]
                # TODO: FIXME: Currently only using first derivatives, sampling is too slow to converge otherwise.
                increasing_penalty = torch.tensor(0.0)
                decreasing_penalty = torch.tensor(0.0)
                peak_first_deriv_penalty = torch.tensor(0.0)
                peak_second_deriv_penalty = torch.tensor(0.0)
                valley_first_deriv_penalty = torch.tensor(0.0)
                valley_second_deriv_penalty = torch.tensor(0.0)
                if wrong_incs.numel() > 0:
                    increasing_penalty = -((k * (wrong_incs.abs() + b)) ** p).mean()
                if wrong_decs.numel() > 0:
                    decreasing_penalty = -((k * (wrong_decs.abs() + b)) ** p).mean()
                if wrong_peaks_1.numel() > 0:
                    peak_first_deriv_penalty = -(
                        (k * (wrong_peaks_1.abs() + 0.0)) ** p
                    ).mean()
                if wrong_peaks_2.numel() > 0:
                    peak_second_deriv_penalty = -(
                        (k * (wrong_peaks_2.abs() + b)) ** p
                    ).mean()
                if wrong_valleys_1.numel() > 0:
                    valley_first_deriv_penalty = -(
                        (k * (wrong_valleys_1.abs() + 0.0)) ** p
                    ).mean()
                if wrong_valleys_2.numel() > 0:
                    valley_second_deriv_penalty = -(
                        (k * (wrong_valleys_2.abs() + b)) ** p
                    ).mean()
                return (
                    increasing_penalty
                    + decreasing_penalty
                    + peak_first_deriv_penalty
                    # + peak_second_deriv_penalty
                    + valley_first_deriv_penalty
                    # + valley_second_deriv_penalty
                )

            sampler = None
            for jitter in torch.tensor(10.0) ** torch.arange(-12, -8):
                try:
                    sampler = ElipticalSliceSampler(
                        covar + jitter * torch.eye(covar.shape[0]),
                        log_likelihood,
                    )
                    break
                except Exception:
                    continue
            if sampler is None:
                # Skip this sample
                raise StopIteration(
                    "Could not construct sampler (likely due to NonPosDef error)."
                )
            n_chains = 4
            assert learner.config.ess_burnin is not None, "ess_burnin must be set."
            assert learner.config.ess_n_samples is not None, (
                "ess_n_samples must be set."
            )
            assert learner.config.ess_thin is not None, "ess_thin must be set."
            samples = sampler.sample_chains(
                n_chains=n_chains,
                n_samples=learner.config.ess_n_samples,
                n_burnin=learner.config.ess_burnin,
                thin=learner.config.ess_thin,
            )
            conditional_variance: torch.Tensor = samples[:, :, : x_test.shape[0]].var(
                dim=(0, 1)
            )
            assert isinstance(conditional_variance, torch.Tensor)
            info = {"posterior_samples": samples}
            return conditional_variance, info
