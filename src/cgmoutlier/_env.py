"""One environment check, run by every script before it does any work.

The failure this exists to prevent: a numpy in the user site directory
(`~/.local/lib/pythonX.Y/site-packages`) silently shadows the one in the active conda
environment. Python puts user site ahead of the environment on sys.path, so `conda
activate` does not protect you. The symptom is an ImportError from deep inside
scikit-learn about dtype sizes, which names neither numpy nor the shadowing.

This turns that into a message that says what to do.
"""
from __future__ import annotations

import os
import site
import sys


def check(strict: bool = True) -> None:
    import numpy as np

    major = int(np.__version__.split(".")[0])
    user_site = getattr(site, "USER_SITE", "") or ""
    shadowed = bool(user_site) and np.__file__.startswith(user_site)

    if major < 2 and not shadowed:
        return

    lines = [f"numpy {np.__version__} loaded from {np.__file__}"]
    if shadowed:
        lines.append(
            f"  This is the USER site directory ({user_site}), which takes priority\n"
            f"  over the active environment. `conda activate` does not override it.")
    if major >= 2:
        lines.append(
            "  numpy 2.x breaks the binary interface that scikit-learn 1.1 and the\n"
            "  vendored generators were compiled against.")
    lines.append("\nFix, cheapest first:\n"
                 "    PYTHONNOUSERSITE=1 python <your command>      # this run only\n"
                 "    export PYTHONNOUSERSITE=1                     # this shell\n"
                 "    pip install -r requirements.txt               # in a clean venv\n")
    msg = "\n".join(lines)
    if strict and os.environ.get("CGMOUTLIER_IGNORE_ENV") != "1":
        raise RuntimeError(msg + "\nSet CGMOUTLIER_IGNORE_ENV=1 to proceed anyway.")
    print("WARNING: " + msg, file=sys.stderr)
