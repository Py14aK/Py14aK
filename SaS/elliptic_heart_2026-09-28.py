#!/usr/bin/env python3
"""Elliptic heart, Wicklin 2024-02-07 / Conway 2023 / Newton.

Generating ellipse:
    x**2 + y**2 - x*y = 1
Polar:
    r**2 = 1 / (1 - 0.5 * sin(2 * theta))
Heart:
    t in [-pi/2, pi/2]
    right (r cos t, r sin t)
    left  (-r cos t, r sin t) at angle pi - t
    order by polar angle, close the polygon.

Distinct from Py14aK/Py14aK SaS/Heart Shaped Box
    H(x, y) = (x**2 + y**2 - 1)**3 - x**2 * y**3

Run:
    python3 elliptic_heart_2026-09-28.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon


MAX_R = float(np.sqrt(2.0))
MIN_R = float(np.sqrt(2.0 / 3.0))
OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HeartPolygon:
    theta: np.ndarray
    x: np.ndarray
    y: np.ndarray


def ellipse_radius(theta: np.ndarray) -> np.ndarray:
    den = 1.0 - 0.5 * np.sin(2.0 * theta)
    if np.any(den <= 0):
        raise ValueError("polar denominator non-positive")
    return np.sqrt(1.0 / den)


def elliptic_heart(n: int = 401) -> HeartPolygon:
    if n < 5:
        raise ValueError("n must be at least 5")
    t = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    r = ellipse_radius(t)
    x_right = r * np.cos(t)
    y = r * np.sin(t)
    theta_right = t
    theta_left = np.pi - t
    x_left = -x_right
    theta = np.concatenate([theta_right, theta_left])
    x = np.concatenate([x_right, x_left])
    y_all = np.concatenate([y, y])
    order = np.argsort(theta, kind="mergesort")
    theta_s = theta[order]
    x_s = x[order]
    y_s = y_all[order]
    if not np.isclose(x_s[0], x_s[-1]) or not np.isclose(y_s[0], y_s[-1]):
        theta_s = np.append(theta_s, theta_s[0])
        x_s = np.append(x_s, x_s[0])
        y_s = np.append(y_s, y_s[0])
    return HeartPolygon(theta=theta_s, x=x_s, y=y_s)


def implicit_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + y**2 - x * y


def verify(heart: HeartPolygon) -> dict[str, float]:
    right = heart.x >= -1e-12
    residual = implicit_ellipse(heart.x[right], heart.y[right])
    half = heart.x > 1e-12
    left_x = -heart.x[half]
    left_y = heart.y[half]
    matched = 0
    for lx, ly in zip(left_x, left_y):
        if np.min((heart.x + lx) ** 2 + (heart.y - ly) ** 2) < 1e-12:
            matched += 1
    return {
        "n_points": float(heart.x.size),
        "max_abs_implicit_on_right": float(np.max(np.abs(residual - 1.0))),
        "x_min": float(heart.x.min()),
        "x_max": float(heart.x.max()),
        "y_min": float(heart.y.min()),
        "y_max": float(heart.y.max()),
        "closed": float(
            np.isclose(heart.x[0], heart.x[-1]) and np.isclose(heart.y[0], heart.y[-1])
        ),
        "left_matches": float(matched),
        "left_candidates": float(left_x.size),
    }


def _rotated_ellipse(ax, slope: float, **kwargs) -> None:
    angle_deg = float(np.degrees(np.arctan(slope)))
    patch = Ellipse(
        (0.0, 0.0),
        width=2.0 * MAX_R,
        height=2.0 * MIN_R,
        angle=angle_deg,
        fill=False,
        **kwargs,
    )
    ax.add_patch(patch)


def plot_heart(heart: HeartPolygon, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.add_patch(
        Polygon(
            np.column_stack([heart.x, heart.y]),
            closed=True,
            facecolor="#C3540C",
            edgecolor="#7A2200",
            linewidth=1.2,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.axis("off")
    ax.set_title("Elliptic heart")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overlay(heart: HeartPolygon, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.add_patch(
        Polygon(
            np.column_stack([heart.x, heart.y]),
            closed=True,
            facecolor="#F4B6C2",
            edgecolor="#C3540C",
            linewidth=1.0,
        )
    )
    _rotated_ellipse(ax, 1.0, edgecolor="#E08080", linewidth=1.4, alpha=0.85)
    _rotated_ellipse(ax, -1.0, edgecolor="#E08080", linewidth=1.4, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Elliptic heart with generating ellipses")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    heart = elliptic_heart()
    stats = verify(heart)
    csv_path = OUT_DIR / "elliptic_heart_points_2026-09-28.csv"
    png_heart = OUT_DIR / "elliptic_heart_2026-09-28.png"
    png_overlay = OUT_DIR / "elliptic_heart_overlay_2026-09-28.png"
    np.savetxt(
        csv_path,
        np.column_stack([heart.theta, heart.x, heart.y]),
        delimiter=",",
        header="theta,x,y",
        comments="",
    )
    plot_heart(heart, png_heart)
    plot_overlay(heart, png_overlay)
    print("VERIFIED")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"csv={csv_path}")
    print(f"png={png_heart}")
    print(f"overlay={png_overlay}")
    if stats["max_abs_implicit_on_right"] > 1e-10:
        raise SystemExit("implicit residual too large")
    if stats["closed"] != 1.0:
        raise SystemExit("polygon not closed")


if __name__ == "__main__":
    main()
