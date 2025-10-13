"""Trendline fitting and intersection utilities."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import optimize


def _linear_r2(x: np.ndarray, y: np.ndarray, coeffs: Tuple[float, float]) -> float:
    m, b = coeffs
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Fit a linear trendline to the data."""

    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs
    r2 = _linear_r2(x, y, (m, b))
    return {"model": "linear", "coeffs": (float(m), float(b)), "r2": float(r2)}


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {"model": "quadratic", "coeffs": (float(a), float(b), float(c)), "r2": float(r2)}


def fit_power(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Fit a power-law model y = A * x^B."""

    positive_mask = (x > 0) & (y > 0)
    if not np.any(positive_mask):
        raise ValueError("Power-law fit requires positive x and y values")
    log_x = np.log(x[positive_mask])
    log_y = np.log(y[positive_mask])
    b, log_a = np.polyfit(log_x, log_y, 1)
    A = np.exp(log_a)
    y_hat = A * x[positive_mask] ** b
    ss_res = np.sum((y[positive_mask] - y_hat) ** 2)
    ss_tot = np.sum((y[positive_mask] - np.mean(y[positive_mask])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {"model": "power", "coeffs": (float(A), float(b)), "r2": float(r2), "note": "y=A*x^B"}


def eval_linear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    return m * x + b


def eval_quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x**2 + b * x + c


def eval_power(x: np.ndarray, A: float, B: float) -> np.ndarray:
    return A * x**B


def intersections_linear_linear(
    m1: float, b1: float, m2: float, b2: float
) -> List[Tuple[float, float]]:
    if np.isclose(m1, m2):
        return []
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return [(float(x), float(y))]


def intersections_poly_linear(
    a: float, b: float, c: float, m: float, b2: float
) -> List[Tuple[float, float]]:
    coeffs = [a, b - m, c - b2]
    roots = np.roots(coeffs)
    points: List[Tuple[float, float]] = []
    for root in roots:
        if np.isreal(root):
            x = float(np.real(root))
            y = float(a * x**2 + b * x + c)
            points.append((x, y))
    return points


def intersections_numeric(
    f1: Callable[[float], float],
    f2: Callable[[float], float],
    x_min: float,
    x_max: float,
    steps: int = 100,
) -> List[Tuple[float, float]]:
    """Numerically find intersections between two functions."""

    xs = np.linspace(x_min, x_max, steps)
    diff = np.vectorize(lambda x: f1(x) - f2(x))(xs)
    points: List[Tuple[float, float]] = []
    for i in range(len(xs) - 1):
        if np.sign(diff[i]) == 0:
            x_val = xs[i]
            points.append((x_val, f1(x_val)))
        elif np.sign(diff[i]) != np.sign(diff[i + 1]):
            try:
                root = optimize.brentq(lambda x: f1(x) - f2(x), xs[i], xs[i + 1])
                points.append((float(root), float(f1(root))))
            except ValueError:
                continue
    return points
