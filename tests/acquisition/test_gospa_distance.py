"""Compare GOSPA with the finite-set definition using exhaustive assignments."""

from itertools import permutations

import pytest
import torch

from bayes_mapping.acquisition.spatial_statistics import calc_distance


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("alpha", [1, 2])
@pytest.mark.parametrize("a, b", [
    ([0., 0.11], [0.12, 0.23]),  # Minimizing distances gives the wrong p=2 assignment.
    ([0., 0.11], [0.12, 0.23, 0.9]),  # Unmatched points add a cardinality penalty.
    ([0., 0.4], [0., 0.4]),  # Identical sets have zero distance.
    ([0.], [1.]),  # Localization error is capped at the cutoff.
    ([], [0.1, 0.5]),
    ([], []),
])
def test_gospa_matches_exhaustive_definition(a, b, p, alpha):
    cutoff = 0.2
    # Enumerate all injections from the smaller set into the larger set.
    # This independent oracle needs no assignment solver and handles empty sets.
    smaller, larger = sorted([a, b], key=len)
    localization = min(
        sum(min(abs(x - y), cutoff) ** p for x, y in zip(smaller, assignment))
        for assignment in permutations(larger, len(smaller))
    )
    cardinality = cutoff ** p / alpha * (len(larger) - len(smaller))
    expected = (localization + cardinality) ** (1 / p)
    features_a = torch.tensor(a, dtype=torch.float64)
    features_b = torch.tensor(b, dtype=torch.float64)

    # Exercise the public dispatcher, and check symmetry and order independence.
    for first, second in [(features_a, features_b), (features_b, features_a),
                          (features_a.flip(0), features_b.flip(0))]:
        result = calc_distance(first, second, "gospa", cutoff=cutoff, p=p, alpha=alpha)
        assert result.shape == ()
        assert result.dtype == features_a.dtype
        assert result.item() == pytest.approx(expected, abs=1e-12)
