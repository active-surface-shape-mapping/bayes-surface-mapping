import pytest
import torch

from bayes_phase import SimulationOracle


def test_simulation_oracle():
    oracle = SimulationOracle(
        [
            "data/111_Cv_run1.arrow",
            "data/111_Cv_run2.arrow",
            "data/111_Cv_run3.arrow",
        ]
    )

    a1_nearest = oracle.measure(10.0, 15, interpolation="nearest")
    a2_nearest = oracle.measure(10.5, 15, interpolation="nearest")
    a3_nearest = oracle.measure(11.0, 15, interpolation="nearest")
    a4_nearest = oracle.measure(10.51, 15, interpolation="nearest")

    a1_linear = oracle.measure(10.0, 15, interpolation="linear")
    a2_linear = oracle.measure(10.5, 15, interpolation="linear")
    a3_linear = oracle.measure(11.0, 15, interpolation="linear")

    a1_none = oracle.measure(10.0, 15, interpolation="none")
    with pytest.raises(ValueError):
        a2_none = oracle.measure(10.5, 15, interpolation="none")  # NOQA: F841
    a3_none = oracle.measure(11.0, 15, interpolation="none")

    assert torch.allclose(a1_nearest, a2_nearest)
    assert torch.allclose(a3_nearest, a4_nearest)
    assert not torch.allclose(a1_nearest, a3_nearest)

    assert torch.allclose(a1_nearest, a1_linear)
    assert torch.allclose(a3_nearest, a3_linear)
    assert not torch.allclose(a2_nearest, a2_linear)

    assert torch.allclose(a1_nearest, a1_none)
    assert torch.allclose(a3_nearest, a3_none)
