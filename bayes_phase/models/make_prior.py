from typing import Any, Optional

import gpytorch


def make_prior(
    config: Optional[dict[str, Any]],
) -> Optional[gpytorch.priors.Prior]:
    if config is None:
        return None

    prior_name = config["name"]
    args = config.get("args", [])
    kwargs = config.get("kwargs", {})

    PriorClass = getattr(gpytorch.priors, prior_name)

    return PriorClass(*args, **kwargs)