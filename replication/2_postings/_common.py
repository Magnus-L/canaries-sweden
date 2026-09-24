"""
_common.py: helpers shared by the scripts of this pack.

sibling(name) loads another script of the pack as a module, so that one
classification (the advertisement filter of 01) or one panel builder (the
within-employer panel of 14) is written once and imported, not copied.
set_rcparams() applies the figure style of the event-study and rival-
explanation figures; the other figures keep matplotlib's defaults.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The palette of the submitted version's figures, which the event-study and
# rival-explanation figures keep.
DARK_BLUE = "#1B3A5C"
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
DARK_TEXT = "#2C2C2C"
LIGHT_BLUE = "#DCE6F2"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#C8C8C8"
Q_COLORS = {
    "Q1 (lowest)": LIGHT_GRAY,
    "Q2": LIGHT_BLUE,
    "Q3": TEAL,
    "Q4 (highest)": ORANGE,
}


def sibling(stem: str):
    """Import <stem>.py from this folder as a module."""
    spec = importlib.util.spec_from_file_location(f"_pack2_{stem}", HERE / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_rcparams():
    """Publication settings of the submitted version's figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": DARK_TEXT,
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "text.color": DARK_TEXT,
    })


def save_pdf_png(fig, stem: str) -> None:
    """Write <stem>.pdf and <stem>.png (300 dpi) into output/figures."""
    for ext in (".pdf", ".png"):
        kw = {"dpi": 300} if ext == ".png" else {}
        fig.savefig(config.FIGURES / f"{stem}{ext}", bbox_inches="tight", **kw)
