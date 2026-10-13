"""Small statistics helpers (no SciPy needed)."""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_N = 30   # groups smaller than this are flagged / hidden in rate charts


def wilson(k, n, z: float = 1.96):
    """95% Wilson score interval for a proportion (arrays allowed)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z**2 / np.maximum(n, 1)
    centre = (p + z**2 / (2 * np.maximum(n, 1))) / den
    half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1) ** 2)) / den
    return centre - half, centre + half


def rate_table(df: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    """Depression rate (%) per group with n and 95% CI. NaN groups are excluded."""
    g = df.dropna(subset=[col]).groupby(col, observed=True)["depression"].agg(n="size", k="sum").reset_index()
    g = g[g["n"] >= min_n].copy()
    g["rate"] = 100 * g["k"] / g["n"]
    lo, hi = wilson(g["k"], g["n"])
    g["lo"], g["hi"] = 100 * lo, 100 * hi
    return g
