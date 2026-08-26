"""Generation quality: Context-FID, discriminative score, predictive score.

    from cgmoutlier.quality import evaluate
    evaluate(real_windows, synthetic_windows, device="cuda")

WHY THIS EXISTS AT ALL, AND WHY IT SHOULD HAVE EXISTED SOONER
--------------------------------------------------------------
Stage 4 released 123 models and stage 5 measured membership leakage on them without
anyone measuring whether the samples were any good. That is a real hole: if a generator
emits noise then d_IN ~ d_OUT for every subject, the attack reads ~0.5, and "no leakage"
and "no model" are indistinguishable. (The stage-5 result is not consistent with pure
noise -- it found AUC 0.68 and one subject at 0.64, which noise cannot produce -- but
"not noise" is a long way from "good enough that the leakage number transfers to a
generator anyone would deploy".)

THE VENDORED IMPLEMENTATIONS DO NOT RUN HERE, FOR THREE SEPARATE REASONS
------------------------------------------------------------------------
`vendor/DiM-TS/Utils/` ships all three metrics and not one of them is usable as-is:

1. `discriminative_metric.py` and `predictive_metric.py` are TensorFlow 1
   (`tf1.reset_default_graph()`). Neither environment has TensorFlow, and installing it
   is how this project previously replaced torch 2.7.0+cu126 with a cu130 build and
   silently lost CUDA. They are reimplemented here in torch.
2. `context_fid.py` imports `from Models.ts2vec.ts2vec import TS2Vec`; the vendored
   TS2Vec is at `vendor/DiM-TS/ts2vec/ts2vec.py`. Nothing in the vendored tree calls
   Context_FID, so that stale path was never exercised. Fixed by importing the real
   location -- a path repair, not a change of method.
3. **Both TimeGAN-lineage metrics are undefined at C=1**, the same assumption that
   made `mmd_alpha` crash on the first real run (docs/DIMTS.md):

       predictive_metric.py:57    X = placeholder(..., dim-1)  -> 0 input features
       predictive_metric.py:103   data[i][:-1, :(dim-1)]       -> empty array
       discriminative_metric.py:74  hidden_dim = int(dim/2)    -> 0 hidden units

   `discriminative`'s zero width is an artefact of integer division, not a choice, so
   `hidden` is a named argument here. `predictive`'s cross-channel formulation has no
   univariate meaning at all, so it is REDEFINED as the natural univariate analogue --
   one-step-ahead x[t+1] from x[1..t], trained on synthetic and scored on real, which
   is what train-on-synthetic-test-on-real means when there is one channel. That is a
   declared change of definition, not a port.

WHAT THESE NUMBERS ARE AND ARE NOT COMPARABLE TO
-------------------------------------------------
They compare OUR configurations to each other on one fixed implementation. They are NOT
comparable to the numbers in DiM-TS's or TimeGAN's papers: different framework,
different width, and a redefined predictive score. Anyone quoting them beside a
published table is comparing two different measurements.

THE ENCODER HERE IS NOT THE ATTACK ENCODER, ON PURPOSE
-------------------------------------------------------
Context-FID refits TS2Vec per call on that call's subsample, as the published
implementation does. `attack/learned_features.py` fits once and freezes. They must stay
separate: sharing one encoder would make "good quality" and "resists attack" two
projections of the same space, and any privacy-utility relationship would then be a
construction rather than a measurement. That module's docstring says so; this one
obeys it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "DiM-TS"


def _vendored_fid():
    """-> (TS2Vec, calculate_fid) from the vendored tree, whose layout moved.

    `vendor/DiM-TS/ts2vec/` sits at the top level, but every file inside it -- and
    `Utils/context_fid.py` -- imports it as `Models.ts2vec.*`, the location it had in
    the project this was vendored from. Nothing in the tree calls Context_FID, so the
    stale paths were never exercised and the break only surfaces now.

    The alias is registered in sys.modules rather than by moving or editing files: the
    vendored source stays byte-identical to what PROVENANCE.md records, and the repair
    lives here where it is visible. `Models.ts2vec` is bound to the real package
    object, so `Models.ts2vec.models.encoder` then resolves through its __path__.
    """
    import importlib

    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    if "Models.ts2vec" not in sys.modules:
        sys.modules.setdefault("Models", importlib.import_module("Models"))
        sys.modules["Models.ts2vec"] = importlib.import_module("ts2vec")
    ts2vec = importlib.import_module("ts2vec.ts2vec")
    context_fid_mod = importlib.import_module("Utils.context_fid")
    return ts2vec.TS2Vec, context_fid_mod.calculate_fid


def _sub(x: np.ndarray, n: int, rng) -> np.ndarray:
    """n windows drawn without replacement, or all of them if there are fewer."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= n:
        return x
    return x[np.sort(rng.choice(len(x), size=n, replace=False))]


def _trunc(x, k, rng):
    """Cut to k WITHOUT taking a prefix.

    `_sub` returns its subsample in ascending index order and windows.npy is ordered by
    (id, day), so `x[:k]` keeps the alphabetically-first subjects rather than a sample of
    the cohort. Latent while both sides hit the n=3000 cap -- every shipped
    results/quality/*.json has n_real 176,445 and n_synth 12,000 -- and live the first
    time a generator emits fewer than n samples, which is silent: n_eval records the
    reduced k and nothing records WHICH windows.
    """
    if len(x) <= k:
        return x
    return x[np.sort(rng.choice(len(x), k, replace=False))]


def context_fid(real, synth, *, n=2000, seed=2026, device=0, init_seed=None):
    """Frechet distance between TS2Vec representations of real and synthetic windows.

    TS2Vec is refit on this call's real subsample and discarded, which is how the
    published metric is defined; the refit is part of the definition, not an oversight.

    SEEDING. This had exactly the defect PITFALLS section 16 records for
    `discriminative_score`, in the metric next door, and it survived that fix.
    `np.random.default_rng(seed)` covered only the subsample: TS2Vec builds its encoder
    from torch's GLOBAL rng, shuffles through a DataLoader from the same, and -- in
    vendor/DiM-TS/ts2vec/ts2vec.py -- takes its training crops from numpy's LEGACY global
    rng, which `default_rng` does not touch. So the reading depended on how many models
    had been scored earlier in the same process, and `eval_quality.py --runs A B C` scores
    them in one process by design. Every global stream is now pinned.
    """
    import torch
    TS2Vec, calculate_fid = _vendored_fid()

    isd = seed if init_seed is None else init_seed
    torch.manual_seed(isd)
    np.random.seed(isd % (2 ** 32))          # ts2vec crops come from the LEGACY global rng
    rng = np.random.default_rng(seed)
    r, s = _sub(real, n, rng), _sub(synth, n, rng)
    k = min(len(r), len(s))
    r, s = _trunc(r, k, rng), _trunc(s, k, rng)

    m = TS2Vec(input_dims=r.shape[-1], device=device, batch_size=8, lr=0.001,
               output_dims=320, max_train_length=3000)
    m.fit(r, verbose=False)
    er = m.encode(r, encoding_window="full_series")
    es = m.encode(s, encoding_window="full_series")
    return float(calculate_fid(er, es))


def _gru(inp, hidden, out):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(inp, hidden, batch_first=True)
            self.head = nn.Linear(hidden, out)

        def forward(self, x, last=False):
            h, _ = self.rnn(x)
            return self.head(h[:, -1] if last else h)

    return Net()


def discriminative_score(real, synth, *, n=3000, seed=2026, device="cuda",
                         hidden=32, iters=2000, batch=128, lr=1e-3, init_seed=None):
    """|classification accuracy - 0.5| for a post-hoc GRU telling real from synthetic.

    0 means indistinguishable (the best a generator can do); 0.5 means trivially
    separable. `hidden` is named rather than derived as dim/2, which is 0 at C=1.

    ⚠ THIS IS A LOWER BOUND ON SEPARABILITY, AND ONE DRAW OF IT.
    Until 2026-08-17 the GRU's weights were initialised from torch's GLOBAL rng, which
    nothing here seeded, so the result depended on how many models had been scored
    earlier in the same process. Measured cost of that: the seven-day h96 base scored
    0.9475 when it was the first run in its process and 0.5908 when it was the second,
    on a byte-identical samples.npy. `init_seed` now controls it and defaults to `seed`,
    so the default path is reproducible.

    The instability is not uniform, and the shape of it is the useful part. Sixteen
    scorings of the one-day single-channel base span 0.482-0.543: where the generator is
    genuinely indistinguishable the classifier converges to chance every time. Where
    there IS a separable feature the classifier either finds it or does not, and the
    draws are bimodal. So a single low reading is weak evidence of quality, while a
    single high reading is strong evidence of separability -- sweep `init_seed` and
    report the MAXIMUM, not one draw.
    """
    import torch

    torch.manual_seed(seed if init_seed is None else init_seed)
    rng = np.random.default_rng(seed)
    r, s = _sub(real, n, rng), _sub(synth, n, rng)
    k = min(len(r), len(s))
    r, s = _trunc(r, k, rng), _trunc(s, k, rng)

    cut = int(0.8 * k)
    perm_r, perm_s = rng.permutation(k), rng.permutation(k)
    r, s = r[perm_r], s[perm_s]
    dev = torch.device(device)
    tr = torch.as_tensor(np.concatenate([r[:cut], s[:cut]]), device=dev)
    ytr = torch.cat([torch.ones(cut), torch.zeros(cut)]).to(dev)
    te = torch.as_tensor(np.concatenate([r[cut:], s[cut:]]), device=dev)
    yte = torch.cat([torch.ones(k - cut), torch.zeros(k - cut)]).to(dev)

    net = _gru(r.shape[-1], hidden, 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    net.train()
    for _ in range(iters):
        idx = torch.randint(0, len(tr), (batch,), generator=g).to(dev)
        opt.zero_grad()
        loss = lossf(net(tr[idx], last=True).squeeze(-1), ytr[idx])
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred = (net(te, last=True).squeeze(-1) > 0).float()
    acc = float((pred == yte).float().mean())
    return abs(acc - 0.5), acc


def predictive_score(real, synth, *, n=3000, seed=2026, device="cuda",
                     hidden=32, iters=2000, batch=128, lr=1e-3):
    """Train-on-synthetic, test-on-real one-step-ahead MAE. Lower is better.

    REDEFINED for C=1. The published version predicts the last channel from the others
    and has no meaning with one channel (see the module docstring). Here the model sees
    x[1..t] and predicts x[t+1] within the single channel, which is what TSTR means
    univariately. Both the training set (synthetic) and the scoring set (real) use that
    same definition, so the number is internally consistent -- it is simply not the
    published quantity.
    """
    import torch

    # `discriminative_score` seeds torch globally and is called immediately before this
    # by `evaluate`, so the net here HAPPENED to be reproducible -- by coincidence, not
    # construction. `eval_quality.py --skip discriminative` removes the reseed and the
    # coincidence with it, and the weights then come from whatever state TS2Vec's
    # training left behind. Seed it here rather than rely on a neighbour.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r, s = _sub(real, n, rng), _sub(synth, n, rng)
    dev = torch.device(device)
    S = torch.as_tensor(s, device=dev)
    R = torch.as_tensor(r, device=dev)

    net = _gru(r.shape[-1], hidden, r.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.L1Loss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    net.train()
    for _ in range(iters):
        idx = torch.randint(0, len(S), (batch,), generator=g).to(dev)
        x = S[idx]
        opt.zero_grad()
        loss = lossf(net(x[:, :-1]), x[:, 1:])          # next step from the prefix
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        mae = float(torch.abs(net(R[:, :-1]) - R[:, 1:]).mean())
    return mae


def evaluate(real, synth, *, n=3000, n_fid=2000, seed=2026, device="cuda",
             skip=()) -> dict:
    """All three, as a dict. `skip` names metrics to leave out (e.g. a broken TS2Vec)."""
    out = {"n_real": int(len(real)), "n_synth": int(len(synth)),
           "n_eval": int(min(n, len(real), len(synth))), "seed": seed}
    if "context_fid" not in skip:
        out["context_fid"] = context_fid(real, synth, n=n_fid, seed=seed,
                                         device=0 if device == "cuda" else device)
    if "discriminative" not in skip:
        d, acc = discriminative_score(real, synth, n=n, seed=seed, device=device)
        out["discriminative_score"] = d
        out["discriminative_accuracy"] = acc
    if "predictive" not in skip:
        out["predictive_score_tstr"] = predictive_score(real, synth, n=n, seed=seed,
                                                        device=device)
        out["predictive_note"] = ("redefined univariate one-step-ahead; NOT the "
                                  "published cross-channel score, which is undefined "
                                  "at C=1")
    return out
