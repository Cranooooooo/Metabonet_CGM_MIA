"""TS2Vec import shim.

The Context-FID metrics need TS2Vec but we do not duplicate the package; we
register sys.modules aliases so the existing eval_ref_1/wMissing_scripts/ts2vec
package is importable as both `ts2vec` and `Models.ts2vec.*` (the latter is
expected by the original Context-FID code path).

Call `load_ts2vec()` once before importing TS2Vec.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_LOADED = False


def load_ts2vec() -> None:
    global _LOADED
    if _LOADED:
        return

    # eval_ref_1/wMissing_scripts/ts2vec lives one directory up from utils/.
    here = Path(__file__).resolve().parent
    ts2vec_root = here.parent / "eval_ref_1" / "wMissing_scripts"
    if not (ts2vec_root / "ts2vec").exists():
        # Fall back to the woMissing_scripts copy.
        ts2vec_root = here.parent / "eval_ref_1" / "woMissing_scripts"
    if not (ts2vec_root / "ts2vec").exists():
        raise ImportError(
            f"Could not locate ts2vec package under {ts2vec_root}. "
            "Context-FID needs eval_ref_1/{w,wo}Missing_scripts/ts2vec/."
        )

    if str(ts2vec_root) not in sys.path:
        sys.path.insert(0, str(ts2vec_root))

    import ts2vec
    import ts2vec.utils
    import ts2vec.models
    import ts2vec.models.encoder
    import ts2vec.models.losses

    sys.modules.setdefault("Models", types.ModuleType("Models"))
    for k, v in {
        "Models.ts2vec":               ts2vec,
        "Models.ts2vec.utils":         ts2vec.utils,
        "Models.ts2vec.models":        ts2vec.models,
        "Models.ts2vec.models.encoder": ts2vec.models.encoder,
        "Models.ts2vec.models.losses": ts2vec.models.losses,
    }.items():
        sys.modules[k] = v

    import ts2vec.ts2vec  # noqa: F401
    sys.modules["Models.ts2vec.ts2vec"] = ts2vec.ts2vec

    _LOADED = True


def get_TS2Vec_class():
    """Return the TS2Vec class object after ensuring the package is loaded."""
    load_ts2vec()
    from Models.ts2vec.ts2vec import TS2Vec
    return TS2Vec
