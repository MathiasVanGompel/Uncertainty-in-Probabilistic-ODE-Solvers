"""
Matplotlib styling helpers for ProbNum 2025/2026-style figures.
"""

from contextlib import contextmanager
import math
from pathlib import Path
from typing import Iterator, Optional

import matplotlib.pyplot as plt
from tueplots.bundles import probnum2025


SINGLE_COLUMN_WIDTH_PT = 240.0
FULL_WIDTH_PT = 500.0
_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


_PT_PER_INCH = 72.0

def _pt_to_in(x_pt: float) -> float:
    return float(x_pt) / _PT_PER_INCH

def _default_height_pt(width_pt: float) -> float:
    return float(width_pt) / _GOLDEN_RATIO


_FIG_FORMAT = "pdf"

def set_figure_format(fmt: str) -> None:
    global _FIG_FORMAT
    _FIG_FORMAT = str(fmt).lower().lstrip(".")

def get_figure_format() -> str:
    return _FIG_FORMAT

def setup_matplotlib(*, force: bool = False) -> None:
    """Apply the probnum2025 style globally.
    """
    if getattr(setup_matplotlib, "_did_setup", False) and not force:
        return

    # We keep figure size in rcParams (single-column) and never pass figsize=...
    plt.rcParams.update(dict(probnum2025()))

    # mark
    setattr(setup_matplotlib, "_did_setup", True)

@contextmanager
def full_width_figure(*, height_pt: Optional[float] = None) -> Iterator[None]:
    """Temporarily switch rcParams to a full-width (500 pt) figure size."""
    w_pt = FULL_WIDTH_PT
    h_pt = _default_height_pt(w_pt) if height_pt is None else float(height_pt)
    with plt.rc_context({"figure.figsize": (_pt_to_in(w_pt), _pt_to_in(h_pt))}):
        yield

def savefig(fig: "plt.Figure", outpath: str | Path, *, close: bool = True) -> Path:
    """Save figure in the globally configured format.
    """
    outpath = Path(outpath)
    fmt = get_figure_format()
    out = outpath.with_suffix(f".{fmt}") if outpath.suffix else outpath.with_suffix(f".{fmt}")

    if fmt == "pdf":
        fig.savefig(out, format="pdf")
    else:
        fig.savefig(out, format="png", dpi=200)

    if close:
        plt.close(fig)
    return out
