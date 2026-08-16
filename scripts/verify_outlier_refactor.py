#!/usr/bin/env python
"""Did making the outlier stage multichannel change the single-channel answer?

    python scripts/verify_outlier_refactor.py

The stage now takes a `channels` argument. Four things were touched, and every one of
them is on the path the published 875-subject result came down:

    cohort.channel_raw      replaces to_mgdl in clinical.per_subject_metrics
    shape._ce               sums the squared differences over (time, channel)
    shape.cid_dtw           keeps the channel axis instead of reshaping to (T, 1)
    outliers.run            builds `flat` by flattening channels, not by taking [..., 0]

Each is meant to be an identity when C = 1. "Meant to be" is not a measurement, and
`tests/test_regression.py` compares a STORED consensus against a STORED run — it never
recomputes, so it cannot see a refactor at all. These are the equalities that have to
hold before a three-channel number is worth reading, checked on the real cohort rather
than on a fixture.
"""
from __future__ import annotations

import sys

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()

from cgmoutlier.data.cohort import channel_raw, load, to_mgdl
from cgmoutlier.outliers import shape as SH

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def main():
    print("=== channel_raw is to_mgdl on a single-channel manifest ===")
    X, sids, man = load("data/cohort/metabonet875")
    X = np.asarray(X[:5000])
    a = to_mgdl(X[:, :, 0], man)
    b = channel_raw(X, man, "CGM")
    check("exact equality", bool(np.array_equal(a, b)),
          f"max |diff| = {np.abs(a - b).max():.3e}")

    print("\n=== channel_raw picks the right channel on a multichannel manifest ===")
    X3, s3, m3 = load("data/cohort/metabonet_sid_c3")
    X3 = np.asarray(X3[:5000])
    cgm = channel_raw(X3, m3, "CGM")
    check("CGM in a plausible mg/dL range",
          bool(40 < np.median(cgm) < 300), f"median {np.median(cgm):.1f}")
    bas = channel_raw(X3, m3, "basal")
    check("basal is not CGM", bool(np.median(bas) < 1.0), f"median {np.median(bas):.4f}")
    try:
        to_mgdl(X3[:, :, 0], m3)
        check("to_mgdl refuses a multichannel manifest", False, "it did not raise")
    except KeyError:
        check("to_mgdl refuses a multichannel manifest", True, "KeyError as intended")

    print("\n=== the DTW kernel is unchanged at C = 1 ===")
    rng = np.random.default_rng(0)
    A2 = rng.standard_normal((24, 288)).astype(np.float64)
    B2 = rng.standard_normal((6, 288)).astype(np.float64)
    D2 = SH.cid_dtw(A2, B2)
    D3 = SH.cid_dtw(A2[:, :, None], B2[:, :, None])
    check("2-D and (n, T, 1) agree", bool(np.array_equal(D2, D3)),
          f"max |diff| = {np.abs(D2 - D3).max():.3e}")
    ce2, ce3 = SH._ce(A2), SH._ce(A2[:, :, None])
    check("_ce agrees", bool(np.array_equal(ce2, ce3)))

    print("\n=== the DTW kernel actually uses the extra channels ===")
    A3 = np.stack([A2, rng.standard_normal((24, 288))], axis=-1)
    B3 = np.stack([B2, rng.standard_normal((6, 288))], axis=-1)
    Dm = SH.cid_dtw(A3, B3)
    check("multichannel distances differ from CGM-only",
          bool(not np.allclose(Dm, D2)), f"median ratio {np.median(Dm / D2):.3f}")
    check("multichannel distances are larger",
          bool((Dm >= D2 * 0.999).mean() > 0.95),
          f"{(Dm >= D2).mean():.1%} of pairs")

    print("\n=== flattening reproduces the old `flat` at C = 1 ===")
    cur = np.ascontiguousarray(X[..., [0]], np.float32)
    check("reshape == [..., 0]",
          bool(np.array_equal(cur.reshape(cur.shape[0], -1), X[:, :, 0])))

    print(f"\n{len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
