"""Unit tests for trendline utilities."""
from __future__ import annotations

import numpy as np

from raman_analyzer.analysis.trendlines import (
    fit_linear,
    fit_power,
    fit_quadratic,
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


def test_fit_quadratic() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(-3, 3, 80)
    y = 0.5 * x**2 - 2 * x + 3 + rng.normal(scale=0.1, size=x.shape)
    fit = fit_quadratic(x, y)
    a, b, c = fit["coeffs"]
    assert abs(a - 0.5) < 0.1
    assert abs(b + 2.0) < 0.3
    assert fit["r2"] > 0.95


def test_fit_power_positive() -> None:
    rng = np.random.default_rng(2)
    x = np.linspace(0.1, 10, 60)
    y = 3.0 * x**1.5 * np.exp(rng.normal(scale=0.02, size=x.shape))
    fit = fit_power(x, y)
    A, B = fit["coeffs"]
    assert abs(A - 3.0) < 0.3
    assert abs(B - 1.5) < 0.1
    assert fit["r2"] > 0.95
