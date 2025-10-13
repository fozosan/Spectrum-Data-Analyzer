"""Unit tests for trendline utilities."""
from __future__ import annotations

import numpy as np

from raman_analyzer.analysis.trendlines import (
    fit_linear,
    intersections_linear_linear,
)


def test_fit_linear_recovers_coefficients() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + rng.normal(scale=0.1, size=x.shape)
    fit = fit_linear(x, y)
    m, b = fit["coeffs"]
    assert abs(m - 2.0) < 0.1
    assert abs(b - 1.0) < 0.1
    assert fit["r2"] > 0.95


def test_linear_intersection() -> None:
    points = intersections_linear_linear(2, 1, 1, 0)
    assert len(points) == 1
    x, y = points[0]
    assert abs(x + 1) < 1e-6
    assert abs(y + 1) < 1e-6
