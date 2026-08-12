import pytest
import torch
from typing import Any, Optional

from bayes_phase.data_io import BayesODataGenerator

test_targets = [
    ("Branin", None, None, None, None, {}),
    ("Cosines", None, 1001.2, 3, None, {"dim_problem": 2}),
    ("Griewank", 100.0, 1500.1, 5, 8, {"dim_problem": 2}),
]

@pytest.mark.parametrize("target_name,min_temp,max_temp,min_theta,max_theta,target_kwargs", test_targets)
def test_bayeso_data_generator_init(target_name: str, min_temp: Optional[float], max_temp: Optional[float], min_theta: Optional[int], max_theta: Optional[int], target_kwargs: dict[str, Any]):
    oracle = BayesODataGenerator(
        target_name,
        min_temp,
        max_temp,
        min_theta,
        max_theta,
        **target_kwargs,
    )
    if min_theta is None:
        min_theta = 1
    if max_theta is None:
        max_theta = 16
    if min_temp is None:
        min_temp = 10.0
    if max_temp is None:
        max_temp = 2009.0
    assert oracle.min_temp == min_temp
    assert oracle.max_temp == max_temp
    assert oracle.min_theta == min_theta
    assert oracle.max_theta == max_theta

def test_bayeso_data_generator_init_invalid():
    with pytest.raises(NotImplementedError):
        oracle = BayesODataGenerator(
            "Griewank",
            dim_problem=4,
        )


@pytest.mark.parametrize("target_name,min_temp,max_temp,min_theta,max_theta,target_kwargs", test_targets)
def test_bayeso_data_generator_measure(target_name: str, min_temp: Optional[float], max_temp: Optional[float], min_theta: Optional[int], max_theta: Optional[int], target_kwargs: dict[str, Any]):
    oracle = BayesODataGenerator(
        target_name,
        min_temp,
        max_temp,
        min_theta,
        max_theta,
        **target_kwargs,
    )
    # Single measurement
    assert oracle.measure_single(1000.0, 7).shape == (1,)

    # Stacked measurements
    u = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0)).sample(
        (11, 2)
    )
    y1 = oracle.measure_uniform(u)
    assert y1.shape == (11, 1)
    x = oracle.untransform_inputs(u)
    y2 = oracle.measure(x[:, 0], x[:, 1])
    assert y2.shape == (11, 1)
    assert y1.allclose(y2)


@pytest.mark.parametrize("target_name,min_temp,max_temp,min_theta,max_theta,target_kwargs", test_targets)
def test_bayeso_data_generator_transform_untransform(target_name: str, min_temp: Optional[float], max_temp: Optional[float], min_theta: Optional[int], max_theta: Optional[int], target_kwargs: dict[str, Any]):
    oracle = BayesODataGenerator(
        target_name,
        min_temp,
        max_temp,
        min_theta,
        max_theta,
        **target_kwargs,
    )
    # Forward:
    x = (
        torch.distributions.Uniform(
            torch.Tensor([oracle.min_temp, oracle.min_theta]),
            torch.Tensor([oracle.max_temp, oracle.max_theta]),
        )
        .sample((10, 1))
        .squeeze()
    )
    z = oracle.transform_inputs(x)
    bounds = oracle.target.get_bounds()
    lb = torch.tensor(bounds[:, 0])
    ub = torch.tensor(bounds[:, 1])
    assert ((z >= lb) & (z <= ub)).all()
    x2 = oracle.untransform_inputs(z)
    assert x.allclose(x2)

    # Reverse:
    u = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0)).sample(
        (11, 2)
    )
    x = oracle.untransform_inputs(u)
    assert ((x[:, 0] >= oracle.min_temp) & (x[:, 0] <= oracle.max_temp)).all()
    assert ((x[:, 1] >= oracle.min_theta) & (x[:, 1] <= oracle.max_theta)).all()
    u2 = oracle.transform_inputs(x)
    assert u.allclose(u2)
