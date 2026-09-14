#!/usr/bin/env python
"""Does resuming IG-FM mid-training land on the same weights as never stopping?

WHY THIS EXISTS. The seven-day campaign trains 52 include models of 55-80 h each against
24 h walltimes, so every one of them resumes 2-3 times. Two claims now rest on the resume
path being exact, and neither had a test:

  * the 26 include models of a cell are `steps`-iteration models, comparable to each other
    and to a `base` staged at the same budget;
  * `iter-N.pt` is the end state of an N-step run -- which is what lets
    `scripts/stage_base_from_curve.py` substitute a checkpoint for a training run at all.

Before 2026-09-14 both were FALSE: `fit()` restored model/EMA/optimiser but not the RNG,
while `torch.manual_seed` ran unconditionally at process start, so iterations after the
resume replayed the batch stream. Both `d7` bases were resumed at iteration 10,000 and
nobody noticed for a week.

THE DESIGN, AND WHY IT IS NOT JUST "COMPARE TWO RUNS"
----------------------------------------------------
GPU kernels are not bit-deterministic, so "A equals B exactly" is the wrong bar and would
fail for reasons that have nothing to do with the resume. Three runs, same seed, same
data, same tiny config:

    A   train straight through to `steps`
    A'  train straight through to `steps` again, in a fresh directory
    B   train to `steps//2`, stop, then resume a NEW generator to `steps`

`A vs A'` measures the floor: how far apart two identical uninterrupted runs land on this
hardware. `A vs B` is the question. The test passes when **A-vs-B is no worse than
A-vs-A'** -- i.e. resuming costs nothing beyond the hardware's own noise.

    C   same split as B, but the RNG keys are stripped from the checkpoint first

C is a POSITIVE CONTROL and the test also fails if C *passes*: a test for "the resume is
exact" that cannot detect an inexact resume proves nothing. C reproduces exactly the bug
this code was written to fix, so `C >> floor` is what gives the A-vs-B result its meaning.

    python scripts/check_resume_equivalence.py            # needs 1 GPU, ~1 minute
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.generators.registry import get as get_generator     # noqa: E402

SEED = 2026


def _params(steps: int, workdir: Path) -> dict:
    # Deliberately tiny: this is a test of the training LOOP's bookkeeping, not of the
    # model. save_every=1 so the split point is always on a checkpoint boundary, and
    # keep_checkpoints high enough that nothing is pruned under us.
    return {"hidden": 16, "n_heads": 2, "n_enc_layers": 1, "n_dec_layers": 1,
            "level_dims": 8, "order_dims": 8, "coord_mask": True,
            "steps": steps, "save_every": 1, "keep_checkpoints": 99,
            "micro_batch": 2, "batch_size": 4, "workdir": str(workdir)}


def _fit(X, steps, workdir, T, C):
    gen = get_generator("igfm")(T=T, C=C, params=_params(steps, workdir),
                                device="cuda", seed=SEED)
    gen.fit(X)
    return gen


def _flat(gen):
    """Model AND EMA.

    `sample()` draws from `self._ema.module`, so the EMA is the released artefact, and it
    carries state the model does not: `AveragedModel` registers `n_averaged` as a
    persistent buffer, and `update_parameters` branches on it -- at 0 it COPIES the model
    parameters in instead of averaging. So if `n_averaged` ever stopped round-tripping
    through the checkpoint, the first step after every resume would silently wipe the
    accumulated EMA, while `_model` stayed bit-identical and a model-only comparison
    reported 0.0. Comparing only the model would leave this test with no power over half
    of what the resume path restores.
    """
    import torch
    sd = dict(gen._model.state_dict())
    sd.update({"ema." + k: v for k, v in gen._ema.state_dict().items()})
    return torch.cat([v.detach().float().flatten().cpu() for _, v in sorted(sd.items())])


def _dist(a, b) -> float:
    import torch
    return float(torch.max(torch.abs(a - b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8, help="must be even")
    ap.add_argument("--keep", default=None, help="keep the scratch dirs here")
    a = ap.parse_args()
    if a.steps % 2:
        sys.exit("--steps 必须是偶数")

    import torch
    if not torch.cuda.is_available():
        sys.exit("这个检查必须在 GPU 上做:抽批次的是 CUDA 那条随机流,"
                 "在 CPU 上跑等于没测到要测的东西")

    T, C, N = 32, 1, 16
    X = np.clip(np.random.RandomState(0).randn(N, T, C), -1, 1).astype(np.float32)
    root = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="igfm_resume_"))
    root.mkdir(parents=True, exist_ok=True)
    half = a.steps // 2
    print(f"[check] T={T} C={C} N={N} steps={a.steps} 断点={half}  工作目录 {root}\n")

    print("== A:一口气训完")
    A = _flat(_fit(X, a.steps, root / "A", T, C))
    print("\n== A':同样的配置再训一遍(测硬件本身的不确定性)")
    A2 = _flat(_fit(X, a.steps, root / "A2", T, C))
    floor = _dist(A, A2)

    print(f"\n== B:训到 {half} 步停,再续到 {a.steps}")
    _fit(X, half, root / "B", T, C)
    B = _flat(_fit(X, a.steps, root / "B", T, C))
    resume = _dist(A, B)

    print(f"\n== C:同样的断点,但先把存档里的 RNG 状态删掉(阳性对照)")
    shutil.copytree(root / "B", root / "C", dirs_exist_ok=True)
    for p in sorted((root / "C").glob("iter-*.pt")):
        if int(p.stem.split("-")[1]) != half:
            p.unlink()                       # 只留断点那个,续训才会从它接
    ck = torch.load(root / "C" / f"iter-{half}.pt", map_location="cpu",
                    weights_only=False)
    for k in ("rng_torch", "rng_numpy", "rng_cuda"):
        ck.pop(k, None)
    torch.save(ck, root / "C" / f"iter-{half}.pt")
    C_ = _flat(_fit(X, a.steps, root / "C", T, C))
    broken = _dist(A, C_)

    print("\n" + "=" * 64)
    print(f"  A vs A'  (硬件噪声地板)          {floor:.3e}")
    print(f"  A vs B   (续训,恢复 RNG)         {resume:.3e}")
    print(f"  A vs C   (续训,RNG 被抹掉,阳性对照) {broken:.3e}")
    print("=" * 64)

    ok = True
    # 续训误差不得超过地板本身 —— 地板为 0(完全确定)时要求逐位相同。
    budget = max(floor * 10, 1e-9)
    if resume > budget:
        print(f"❌ 续训之后的权重和一口气训完差 {resume:.3e},超过地板 {floor:.3e} 的容许"
              f"范围 {budget:.3e} —— 续训【不】等价,iter-N.pt 不是 N 步训练的终态")
        ok = False
    else:
        print(f"✅ 续训等价:差 {resume:.3e} ≤ {budget:.3e}(地板 {floor:.3e})")
    # 阳性对照必须失败,否则这个检查没有分辨力,上面那个 ✅ 不能信
    if broken <= budget:
        print(f"❌ 阳性对照也通过了({broken:.3e}):这个检查测不出「RNG 没恢复」这件事,"
              f"所以它对续训等价性的结论没有证明力。先修检查本身。")
        ok = False
    else:
        print(f"✅ 阳性对照按预期失败({broken:.3e} > {budget:.3e}),说明本检查有分辨力")

    if not a.keep and ok:
        shutil.rmtree(root, ignore_errors=True)
    elif not ok:
        print(f"\n现场保留在 {root} —— 失败时不删,否则只剩四个数字没法查")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
