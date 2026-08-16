"""Learned-representation phi dims: emb_min, emb_avg, emb_ball, embcos_ball.

Why
---
Every other phi column is a hand-written distance on raw windows. RQ1 showed that
swapping one hand-written distance for another (Euclidean -> DTW -> Frechet) does not
help, which leaves open the question the hand-written family cannot answer: would a
*learned* representation see membership that fixed formulas miss? Modern MIA work is
built on learned embeddings, so an evaluation that only tries fixed distances is one
a reviewer will not accept as an upper bound on attack strength.

The four dims mirror the cheap four exactly -- min, mean, epsilon-ball, cosine-ball --
so any difference is attributable to the space, not to a different statistic.

Encoder -- and why it is NOT the Context-FID encoder
----------------------------------------------------
TS2Vec, the same architecture the Context-FID quality metric uses, but a SEPARATE fit
with separate weights. The two must not be shared, for three reasons:

  * Context-FID has to stay faithful to the published implementation it came from
    (DiM-TS Utils/context_fid.py), including its refit-per-call behaviour, or it
    stops being comparable to numbers in other papers. The attack encoder is part of
    our adversary and we are free to change it. Coupling them means editing the
    adversary silently edits the yardstick.
  * Their requirements are opposite. The attack needs ONE frozen space shared across
    all G folds -- a per-fold fit would make the feature space depend on which
    subjects were held out, the fold-dependent confound that the fixed r_euc radius
    exists to prevent. Context-FID needs the refit because the refit is part of how
    that metric is defined.
  * A shared encoder would make "good quality" and "resists attack" two projections
    of one space, so the privacy-utility plot would show a relationship created by
    construction rather than measured.

Concretely: Context-FID fits on that call's 2000-window subsample and discards the
weights; this module fits once on 5000 windows drawn from the full training set with
seed 2026 and caches the weights to disk.

RESIDUAL COUPLING, disclosed rather than engineered away: same architecture, same
source data. Different weights, but similar representations. If TS2Vec fails to
encode some distinction in CGM, the metric and the attack lose it together.

SCOPE OF ANY NEGATIVE RESULT: TS2Vec is here because it is present in the environment
and is a recognised time-series representation baseline -- not because it is the best
an adversary could do. If the learned dims do not beat the hand-written ones, the
supported claim is "the TS2Vec representation does not help", not "learned
representations do not help".

LEAKAGE ARGUMENT -- read before trusting any number this produces
-----------------------------------------------------------------
1. The encoder never sees membership labels. It is fit with TS2Vec's contrastive
   objective on windows only.
2. It never sees synthetic data. Fitting on generated samples would let it learn
   generator-specific artifacts and inflate every score.
3. It is fit once and shared across all G folds. Fitting per fold would make each
   fold's feature space depend on which subjects were held out -- a fold-dependent
   confound of exactly the kind the fixed r_euc radius exists to avoid.
4. It IS transductive: the windows it is fit on are the same windows later scored.
   That is standard for a frozen extractor and it does not create label leakage, but
   it is not the same as an encoder trained on disjoint data, and it is recorded in
   the output metadata rather than left implicit.
5. The permutation null remains valid: permuting the subject->group map changes no
   input to the encoder, so the null distribution is computed in the same frozen
   space as the observed statistic.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

LEARNED_NAMES: List[str] = ["emb_min", "emb_avg", "emb_ball", "embcos_ball"]

# TS2Vec is vendored under vendor/ts2vec (see its PROVENANCE.md). An earlier version
# reached it through an absolute path on one machine, which is exactly the kind of
# thing that makes a repository un-cloneable.
_REPO = Path(__file__).resolve().parents[3]
_VENDOR = _REPO / "vendor"
_CACHE = _REPO / "results" / "_encoder_cache"


def _fingerprint(arr: np.ndarray, n_train: int, seed: int, dims: int) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(arr[:64]).tobytes())      # cheap content probe
    h.update(f"{n_train}|{seed}|{dims}".encode())
    return h.hexdigest()[:16]


def get_encoder(real: np.ndarray, *, n_train: int = 5000, seed: int = 2026,
                dims: int = 320, device: str = "cuda"):
    """Fit (or load) a frozen TS2Vec encoder on real windows only.

    Cached by a fingerprint of (real shape + head bytes, n_train, seed, dims) so a
    rerun does not silently pick up a differently-trained encoder -- which would make
    two RQ levels incomparable without any error being raised.
    """
    sys.path.insert(0, str(_VENDOR))
    from ts2vec.ts2vec import TS2Vec  # noqa: E402
    import torch  # noqa: E402

    real = np.asarray(real, dtype=np.float32)
    fp = _fingerprint(real, n_train, seed, dims)
    _CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = _CACHE / f"ts2vec_{fp}.pt"

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = TS2Vec(input_dims=real.shape[-1], device=dev, batch_size=8, lr=0.001,
                   output_dims=dims, max_train_length=3000)
    if ckpt.exists():
        model.net.load_state_dict(torch.load(ckpt, map_location=dev))
        model.net.eval()
        return model, dict(fitted=False, cache=str(ckpt), fingerprint=fp,
                           n_train=n_train, dims=dims, seed=seed)

    rng = np.random.default_rng(seed)
    idx = rng.choice(real.shape[0], size=min(n_train, real.shape[0]), replace=False)
    model.fit(real[np.sort(idx)], verbose=False)
    tmp = ckpt.with_suffix(".tmp")
    import torch as _t
    _t.save(model.net.state_dict(), tmp)
    tmp.rename(ckpt)                                        # atomic: no half-written ckpt
    return model, dict(fitted=True, cache=str(ckpt), fingerprint=fp,
                       n_train=n_train, dims=dims, seed=seed)


def encode_real_cached(model, real: np.ndarray, fingerprint: str, batch: int = 1024) -> np.ndarray:
    """Embeddings of the real training windows, computed once and reused.

    These do not depend on alpha, on the fold, or on the generator -- injection only
    ever rewrites synthetic samples. Recomputing them per alpha was both nine times
    the GPU work and the reason eight concurrent workers exhausted the device: each
    was pushing all 73k windows through TS2Vec at the same moment. Keyed by the
    encoder fingerprint so a different encoder cannot silently reuse this array.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    p = _CACHE / f"real_emb_{fingerprint}_{real.shape[0]}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    E = encode(model, real, batch=batch)
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, E)
    tmp.rename(p)                                   # atomic: no half-written cache
    return E


def encode(model, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    """(N,T,C) -> (N,dims) full-series representation, float32."""
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(model.encode(np.asarray(x[i:i + batch], dtype=np.float32),
                                encoding_window="full_series"))
    return np.concatenate(out, axis=0).astype(np.float32)


def compute_learned_phi(real_emb: np.ndarray, syn_emb: np.ndarray, *,
                        r_emb: float | None = None, r_emb_q: float = 0.05,
                        cos_c0: float = 0.95) -> Tuple[np.ndarray, List[str], dict]:
    """Same four statistics as the cheap dims, computed on embeddings.

    `r_emb` must come from a fixed calibration for the same reason `r_euc` does: a
    per-call quantile would give member and non-member rows different thresholds.
    """
    R = np.asarray(real_emb, dtype=np.float32)
    S = np.asarray(syn_emb, dtype=np.float32)
    scale = np.sqrt(np.float32(R.shape[1]))
    r2 = (R ** 2).sum(1, keepdims=True)
    s2 = (S ** 2).sum(1, keepdims=True).T
    D = np.sqrt(np.clip(r2 + s2 - 2.0 * (R @ S.T), 0.0, None)) / scale

    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    Dcos = Rn @ Sn.T

    r = float(r_emb) if r_emb is not None else float(np.quantile(D, r_emb_q))
    phi = np.stack([D.min(1), D.mean(1),
                    (D < r).mean(1), (Dcos > cos_c0).mean(1)], axis=1).astype(np.float32)
    return phi, list(LEARNED_NAMES), dict(r_emb=r, dims=int(R.shape[1]), cos_c0=cos_c0)


def calibrate_r_emb(real_emb, syn_emb, r_emb_q: float = 0.05) -> float:
    R = np.asarray(real_emb, dtype=np.float32)
    S = np.asarray(syn_emb, dtype=np.float32)
    scale = np.sqrt(np.float32(R.shape[1]))
    r2 = (R ** 2).sum(1, keepdims=True)
    s2 = (S ** 2).sum(1, keepdims=True).T
    D = np.sqrt(np.clip(r2 + s2 - 2.0 * (R @ S.T), 0.0, None)) / scale
    return float(np.quantile(D, r_emb_q))
