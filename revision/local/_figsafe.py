#!/usr/bin/env python3
"""
_figsafe.py -- refuse to overwrite a figure no script in the tree claims.

THE ACCIDENT THIS PREVENTS. On 21 September 2026 a finished
`figA2_first_stage.pdf` sat in `revision/figures/` with no generating
script anywhere. That absence was read as a gap to fill, a new figure
was written over it, and because `figures/` is gitignored there was no
version history; SIP then blocked recovery from the Time Machine
snapshot. The only surviving copy was a window someone happened to have
left open.

THE RULE. A figure file that exists but that NO script in the tree
mentions is an orphan: something produced it that we no longer have, so
it is irreplaceable and must not be overwritten. A figure claimed by a
DIFFERENT script belongs to that script. Only a figure this script
claims may be overwritten freely, and a figure that does not yet exist
is always fine to create.

    from _figsafe import save
    save(fig, "figA2_first_stage", __file__)

Set CANARIES_FIG_FORCE=1 to override, deliberately and once.
"""
import os
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
LOCAL = REV / "local"
SRC_DIRS = (LOCAL, REV / "mona", REV.parent / "src")


def claimants(stem: str) -> set:
    """Scripts that name this figure, i.e. that could have produced it."""
    out = set()
    for d in SRC_DIRS:
        if not d.exists():
            continue
        for p in list(d.glob("*.py")) + list(d.glob("*.R")):
            # A test that names a figure is not a thing that can
            # regenerate it, and neither is this module. Counting them
            # would report a genuine orphan as "claimed" and hide the
            # very fact the guard exists to surface.
            if p.name.startswith("test_") or p.name == "_figsafe.py":
                continue
            try:
                if stem in p.read_text(encoding="utf-8", errors="replace"):
                    out.add(p.resolve())
            except OSError:
                pass
    return out


def save(fig, stem: str, generator, figdir: Path | None = None,
         exts=(".pdf", ".png")) -> None:
    """Write <stem><ext> for each ext, unless that would destroy an orphan."""
    figdir = Path(figdir) if figdir else (REV / "figures")
    figdir.mkdir(parents=True, exist_ok=True)
    me = Path(generator).resolve()

    existing = [figdir / f"{stem}{e}" for e in exts
                if (figdir / f"{stem}{e}").exists()]
    if existing and os.environ.get("CANARIES_FIG_FORCE") != "1":
        owners = claimants(stem)
        if me not in owners:
            others = sorted(p.name for p in owners)
            why = (f"it is claimed by {', '.join(others)}" if others
                   else "NO script in the tree claims it, so whatever "
                        "produced it is lost and this file is "
                        "irreplaceable")
            print(f"\n  REFUSING to overwrite {stem}: {why}.\n"
                  f"  Existing: {', '.join(p.name for p in existing)}\n"
                  f"  This script: {me.name}\n"
                  f"  figures/ is gitignored, so an overwrite here cannot "
                  f"be undone.\n"
                  f"  Recover or inspect it first. To override once:\n"
                  f"      CANARIES_FIG_FORCE=1 python3 {me.name}\n",
                  file=sys.stderr)
            raise SystemExit(3)

    for e in exts:
        kw = {"dpi": 300} if e == ".png" else {}
        fig.savefig(figdir / f"{stem}{e}", bbox_inches="tight", **kw)
