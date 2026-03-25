import pytest
import torch

from bayes_phase.data_io import SimulatedDataGenerator


def test_simulated_data_generator_init():
    oracle = SimulatedDataGenerator(
        [
            "data/111_Cv_run1.arrow",
            "data/111_Cv_run2.arrow",
            "data/111_Cv_run3.arrow",
        ]
    )
    assert oracle.min_temp == 10.0
    assert oracle.max_temp == 2009.0
    assert oracle.min_theta == 1
    assert oracle.max_theta == 16


def test_simulated_data_generator_measure():
    oracle = SimulatedDataGenerator(
        [
            "data/111_Cv_run1.arrow",
            "data/111_Cv_run2.arrow",
            "data/111_Cv_run3.arrow",
        ]
    )
    # Single measurement
    assert oracle.measure_single(1000.0, 7).shape == (3,)

    # Stacked measurements
    u = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0)).sample(
        (11, 2)
    )
    y1 = oracle.measure_uniform(u)
    assert y1.shape == (11, 3)
    x = oracle.untransform_inputs(u)
    y2 = oracle.measure(x[:, 0], x[:, 1])
    assert y2.shape == (11, 3)
    assert y1.allclose(y2)


def test_simulated_data_generator_interpolation():
    oracle = SimulatedDataGenerator(
        [
            "data/111_Cv_run1.arrow",
            "data/111_Cv_run2.arrow",
            "data/111_Cv_run3.arrow",
        ]
    )
    a1_nearest = oracle.measure_single(10.0, 15, interpolation="nearest")
    a2_nearest = oracle.measure_single(10.5, 15, interpolation="nearest")
    a3_nearest = oracle.measure_single(11.0, 15, interpolation="nearest")
    a4_nearest = oracle.measure_single(10.51, 15, interpolation="nearest")

    a1_linear = oracle.measure_single(10.0, 15, interpolation="linear")
    a2_linear = oracle.measure_single(10.5, 15, interpolation="linear")
    a3_linear = oracle.measure_single(11.0, 15, interpolation="linear")

    a1_none = oracle.measure_single(10.0, 15, interpolation="none")
    with pytest.raises(ValueError):
        a2_none = oracle.measure_single(10.5, 15, interpolation="none")  # NOQA: F841
    a3_none = oracle.measure_single(11.0, 15, interpolation="none")
    with pytest.raises(ValueError):
        a3_none = oracle.measure_single(11.0, 18, interpolation="none")

    assert torch.allclose(a1_nearest, a2_nearest)
    assert torch.allclose(a3_nearest, a4_nearest)
    assert not torch.allclose(a1_nearest, a3_nearest)

    assert torch.allclose(a1_nearest, a1_linear)
    assert torch.allclose(a3_nearest, a3_linear)
    assert not torch.allclose(a2_nearest, a2_linear)

    assert torch.allclose(a1_nearest, a1_none)
    assert torch.allclose(a3_nearest, a3_none)


def test_simulated_data_generator_transform_untransform():
    oracle = SimulatedDataGenerator(
        [
            "data/111_Cv_run1.arrow",
            "data/111_Cv_run2.arrow",
            "data/111_Cv_run3.arrow",
        ]
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
    u = oracle.transform_inputs(x)
    assert ((u >= 0.0) & (u <= 1.0)).all()
    x2 = oracle.untransform_inputs(u)
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
