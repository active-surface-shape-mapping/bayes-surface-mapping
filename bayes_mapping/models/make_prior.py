from typing import Any

import gpytorch


def make_prior(
    config: dict[str, Any] | None,
) -> gpytorch.priors.Prior | None:
    if config is None:
        return None

    prior_name = config["name"]
    args = config.get("args", [])
    kwargs = config.get("kwargs", {})

    PriorClass = getattr(gpytorch.priors, prior_name)

    return PriorClass(*args, **kwargs)
