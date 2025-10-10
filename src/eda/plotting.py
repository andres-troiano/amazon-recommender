"""Shared plotting helpers for EDA.

These are intentionally minimal; notebooks can customize styles as needed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 4)


__all__ = ["setup_style"]
