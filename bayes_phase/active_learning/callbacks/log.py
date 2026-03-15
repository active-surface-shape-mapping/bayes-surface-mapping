from typing import TYPE_CHECKING
from pathlib import Path
from jsonlines import jsonlines

if TYPE_CHECKING:
    from bayes_phase.active_learning import ActiveLearner


def log(learner: "ActiveLearner"):
    """Log relevant info."""
    log = {}
    hyps = {}
    log["iteration"] = learner._i
    hyps["lengthscale"] = learner.model.base_kernel.lengthscale.tolist()
    hyps["outputscale"] = learner.model.covar_module.outputscale.item()
    hyps["noise"] = learner.model.likelihood.noise.item()
    hyps["mean"] = learner.model.mean_module.constant.item()
    log["hyperparameters"] = hyps
    log["wall_clock_time"] = learner._ti - learner._t0

    savepath = Path(learner.config.base_savepath) / "logs.json"
    savepath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(savepath, mode="a") as writer:
        writer.write(log)
