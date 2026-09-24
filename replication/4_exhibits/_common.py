"""
_common.py: the one helper the figure builders of this pack share.

save() writes a Matplotlib figure to output/figures/ as PDF and as PNG at
300 dots per inch, cropped to its content, under the file name the
manuscript includes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES  # noqa: E402


def save(fig, stem: str, generator=None, figdir: Path | None = None,
         exts=(".pdf", ".png")) -> None:
    """Write <stem>.pdf and <stem>.png to the package's figure folder."""
    figdir = Path(figdir) if figdir else FIGURES
    figdir.mkdir(parents=True, exist_ok=True)
    for e in exts:
        kw = {"dpi": 300} if e == ".png" else {}
        fig.savefig(figdir / f"{stem}{e}", bbox_inches="tight", **kw)
