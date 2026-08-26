#!/usr/bin/env python
"""Pack a built cohort into the single compressed file that goes into git.

    python scripts/pack_cohort.py data/cohort/metabonet875
    python scripts/pack_cohort.py data/cohort/metabonet875 --verify

`windows.npy` for the 875-subject cohort is 210 MB, which does not belong in a
repository. Compressed it is 58 MB, and CGM traces compress well enough that dropping
to float16 would save only a further 6 MB -- not a trade worth making, because then the
shipped data would no longer be the data the checked-in results were computed from.
So the pack is **lossless**: same float32 values, deflate on top.

`data/cohort.load()` reads either form, so nothing downstream changes. `.gitignore`
excludes `windows.npy` and keeps `cohort.npz`.

58 MB is over GitHub's 50 MB advisory threshold and well under its 100 MB hard limit:
expect a warning on push, not a rejection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgmoutlier._env import check as _envcheck   # noqa: E402
_envcheck()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("--verify", action="store_true",
                    help="reload the pack and compare against the source arrays")
    a = ap.parse_args()

    d = Path(a.cohort)
    npy, out = d / "windows.npy", d / "cohort.npz"
    if not npy.exists():
        print(f"no {npy} -- already packed?", file=sys.stderr)
        return 1

    X = np.load(npy)
    s = np.load(d / "subject_ids.npy", allow_pickle=True)
    man = json.loads((d / "manifest.json").read_text())
    assert X.dtype == np.float32, X.dtype

    # Sidecars used to be dropped: the pack held windows and subject_ids only. That is
    # enough to TRAIN and not enough to rebuild or check anything -- a one-day cohort
    # keys its calendar on days.npy and a multi-day one on day_start.npy/seams.npy, and
    # build_cohort_matched.py --match-blocks, the mechanism that makes the four cells
    # comparable, aligns on exactly those. A clone without them can retrain the models
    # and cannot verify the cohorts they were trained on.
    side = {}
    for f in sorted(d.glob("*.npy")):
        if f.name in ("windows.npy", "subject_ids.npy"):
            continue
        arr = np.load(f, allow_pickle=True)
        if arr.shape and arr.shape[0] == X.shape[0]:
            side[f.stem] = arr
        else:
            print(f"  {f.name} is not per-window ({arr.shape}); NOT packed")
    print(f"{X.shape} float32, {X.nbytes / 1e6:.0f} MB"
          + (f" + sidecars {sorted(side)}" if side else " (no sidecars)")
          + " -> compressing ...")
    np.savez_compressed(out, windows=X, subject_ids=s, **side)
    mb = out.stat().st_size / 1e6
    print(f"  {out}  {mb:.1f} MB  ({mb / (X.nbytes / 1e6):.1%} of source)")
    if mb > 100:
        print("  ERROR: over GitHub's 100 MB hard limit.", file=sys.stderr)
        return 1
    if mb > 50:
        print("  note: over GitHub's 50 MB advisory threshold; push warns but works.")

    z = np.load(out, allow_pickle=True)
    bad = [k for k, v in [("windows", X), ("subject_ids", s)] + list(side.items())
           if not np.array_equal(z[k], v)]
    if bad:
        print(f"  ERROR: round-trip differs for {bad}.", file=sys.stderr)
        return 1
    print(f"  round-trip identical (bit for bit), {2 + len(side)} arrays")

    man["pack"] = dict(file=out.name, format="npz/deflate", dtype="float32",
                       lossless=True, size_mb=round(mb, 1),
                       arrays=["windows", "subject_ids"] + sorted(side),
                       sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (d / "manifest.json").write_text(json.dumps(man, indent=2))

    if a.verify:
        from cgmoutlier.data.cohort import load
        npy.rename(npy.with_suffix(".npy.hidden"))
        try:
            X2, s2, _ = load(d)
            assert np.array_equal(np.asarray(X2), X) and np.array_equal(s2, s)
            print("  load() reads the pack and returns identical arrays")
        finally:
            npy.with_suffix(".npy.hidden").rename(npy)

    print(f"\n{npy.name} stays on disk and is gitignored; {out.name} is committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
