#!/usr/bin/env python
"""Stage a budget-curve milestone as the paired `base` of an include campaign.

WHY THIS EXISTS
---------------
The seven-day cells are trained at a budget the curve chose, not at the 25,000 the base
was trained to: `d7_c1` at 12,000 (Context-FID 0.1265 against 0.1035 at the end, both far
inside DiM-TS's 0.3256) and `d7_c2` at 18,000, which is that cell's MINIMUM -- 25,000
reads 0.4205 and fails the gate. The 26 include models are trained with `steps` set to the
milestone.

A membership pair is only a membership pair if the two models differ by ONE SUBJECT. So
the `base` those 26 are compared against has to be the base AT THE SAME BUDGET. Retraining
one costs 50-75 GPU-hours per cell; `iter-<m>.pt` is already on disk.

THE EQUIVALENCE, AND ITS ONE REAL EXCEPTION
-------------------------------------------
The claim "iter-<m>.pt is the end state of an m-step run" rests on four properties of
`generators/igfm.py`. Three are checked here:

  * no LR scheduler -- one AdamW at a fixed `lr`, never touched, so nothing about the
    trajectory knows the horizon (checked by reading the adapter source, below);
  * the only horizon-aware term is `core.lambda_schedule(it, schedule)`, and its first
    boundary must exceed the source run's own `steps` so lam was constant throughout;
  * `_save` fires at `(it+1) % save_every == 0`, so `save_every` must divide m for
    `iter-<m>.pt` to be the state after exactly m iterations.

The fourth -- batches come from `torch.randint` after ONE `torch.manual_seed(job_seed)`,
so iteration i draws the same indices whatever the horizon says -- **holds only for an
uninterrupted run**, and it is the one that has actually been violated. Before
2026-09-14 `fit()` restored model/ema/opt on resume but not the RNG, while
`torch.manual_seed(self.seed)` ran unconditionally at process start. Both `d7` bases
resumed at iteration 10,000, so their iterations 10,000+ drew the indices a fresh run
draws from iteration 0: a REPLAYED stretch of the batch stream.

That is detected here, from the checkpoint mtimes, and it is **not fatal but not
ignorable**. Not fatal: the weights are still an m-iteration model on the same data with
the same hyperparameters, and every job already has its own `job_seed` and therefore its
own batch stream, so a replayed stretch adds no systematic difference between base and
include. Not ignorable: the model is not reproducible from its command line and it is not
the model an m-step run would produce, so staging it requires `--accept-replayed-stream`
and the fact is written into the staged `meta.json` rather than left to be rediscovered.

WHAT IS NOT IDENTICAL EITHER, AND WHY IT DOES NOT BIAS THE RESULT
-----------------------------------------------------------------
The DRAW. `igfm_budget_curve.py` calls `torch.manual_seed(job_seed)` immediately before
`sample()`; an uninterrupted run samples with the RNG wherever training left it. Same
weights, same distribution, a different draw -- the stamp `loo/train.py:resample()` writes
for exactly this case.

It does not bias the study's statistic, but NOT because it "cancels": `attack.statistic.
gap_for_pair` computes `d_out` from the target's own real windows against the base
release, so a different base draw perturbs each target by a target-specific amount rather
than shifting both arms equally. The reason it is unbiased is simpler -- a seeded draw and
a training-RNG draw are both unbiased draws from the SAME model distribution, so neither
the arm AUC nor anything else acquires a systematic tilt. What it does add is variance,
and `subject_auc.py` and `leak_locus.py` report per-subject ABSOLUTE levels and threshold
counts, where that variance does not divide out the way it does in a between-arm
comparison. `d1` has the identical single-shared-base structure, so this is not a `d7`
asymmetry -- but do not claim cancellation.

WHAT THIS SCRIPT REFUSES TO DO
------------------------------
Everything that would let a sample file be attributed to the wrong model. The milestone
directory records `job`, `cohort`, `subjects_sha1` and `from_checkpoint`; all four are
re-derived here from the design and the cohort rather than trusted, the checkpoint is
opened and its own `iter` read back, and the copy is verified by digest. `run_loo.py` will
skip this directory forever after on the strength of its `subjects_sha1`, so this is the
last place anything can be checked.

    python scripts/stage_base_from_curve.py \
        --from results/matrix/curve/d7_c1/m12000 \
        --to   results/runs/loo_igfm_priv_d7_c1/base \
        --cohort data/cohort/matrix_d7_c1 --milestone 12000 \
        --accept-replayed-stream
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

from cgmoutlier._env import check as _envcheck                      # noqa: E402
_envcheck()
from cgmoutlier.data.cohort import load as load_cohort              # noqa: E402
from cgmoutlier.loo.train import training_set, _job_seed            # noqa: E402
import cgmoutlier.generators.igfm as _igfm_mod                      # noqa: E402


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _assert_no_lr_scheduler() -> None:
    """The adapter must not schedule the learning rate.

    This is the premise that lets a checkpoint stand in for a shorter training run, and
    until now it was only asserted in a comment. Checked the same way the lambda default
    is: by reading the adapter source, so editing `igfm.py` cannot leave a stale CHECKED
    in a staged meta.json.
    """
    src = Path(_igfm_mod.__file__).read_text()
    hits = [ln.strip() for ln in src.splitlines()
            if re.search(r"lr_scheduler|LRScheduler|OneCycle|CosineAnneal|"
                         r"get_last_lr|param_group\[.lr.\]", ln)]
    if hits:
        raise SystemExit(
            "generators/igfm.py 里出现了学习率调度的痕迹,而「存档 = 短训练的终态」"
            "这条等价性正是建立在【没有调度】上的。先把它重新论证一遍:\n  "
            + "\n  ".join(hits))


def _adapter_lambda_default() -> str:
    """The adapter's `lambda_schedule` default, read out of its source.

    Not a copy of the literal: if `igfm.py` is edited, a copy would keep the premise
    check passing against a stale value, and that premise is the whole licence for
    substituting a checkpoint for a training run.
    """
    src = Path(_igfm_mod.__file__).read_text()
    m = re.search(r'cfg\.get\(\s*"lambda_schedule"\s*,\s*"([^"]+)"\s*\)', src)
    if not m:
        raise SystemExit(
            "在 generators/igfm.py 里找不到 lambda_schedule 的默认值 —— 这个脚本的"
            "「lam 全程是常数」那条前提就没法验了,拒绝继续")
    return m.group(1)


def _detect_resume(run_dir: Path) -> dict:
    """Did the run that wrote these checkpoints get killed and resumed?

    Read off the checkpoint mtimes: training writes one every `save_every` iterations at
    a near-constant rate, so a resume shows up as one interval whose seconds-per-iteration
    is far above the median. This is derived from the run directory itself rather than
    from a log file, because a run directory carries no pointer to its log.

    A heuristic, and it says so: it reports what it measured (the intervals and the
    median) alongside the verdict, so a reader can disagree with the 3x threshold. It can
    only be fooled in the safe direction by a resume that happened to take about as long
    as normal training -- and the reverse, a slow node flagged as a resume, costs nothing
    but an explicit --accept-replayed-stream.
    """
    cks = sorted(((int(p.stem.split("-")[1]), p) for p in run_dir.glob("iter-*.pt")),
                 key=lambda t: t[0])
    if len(cks) < 3:
        return {"checked": False, "reason": f"只有 {len(cks)} 个存档,看不出节奏"}
    # 一条区间一个记录,保持 (起, 止, 每步秒数) 在一起 —— 之前把速率单独存一个列表、
    # 再和 zip(cks, cks[1:]) 并排遍历,任何一条区间被跳过两边就会错位。
    ivals = []
    for (i0, p0), (i1, p1) in zip(cks, cks[1:]):
        di = i1 - i0
        if di > 0:
            ivals.append((i0, i1, (p1.stat().st_mtime - p0.stat().st_mtime) / di))
    if len(ivals) < 2:
        return {"checked": False, "reason": "可用区间不足 2 个,看不出节奏"}
    flat = sorted(v[2] for v in ivals)
    n = len(flat)
    med = flat[n // 2] if n % 2 else 0.5 * (flat[n // 2 - 1] + flat[n // 2])
    gaps = [{"after_iter": i0, "before_iter": i1, "sec_per_iter": round(r, 1)}
            for i0, i1, r in ivals if med > 0 and r > 3 * med]
    return {"checked": True, "resumed": bool(gaps), "gaps": gaps,
            "median_sec_per_iter": round(med, 3),
            "n_checkpoints": len(cks), "n_intervals": len(ivals),
            "threshold": "sec_per_iter > 3x median"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="a milestone dir written by scripts/igfm_budget_curve.py")
    ap.add_argument("--to", dest="dst", required=True,
                    help="the include tree's base/ dir, e.g. "
                         "results/runs/loo_igfm_priv_d7_c1/base")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--design", default="results/matrix/design/rep1")
    ap.add_argument("--job", default="base")
    ap.add_argument("--milestone", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--assert-resumed", action="store_true",
                    help="fail if the mtime scan does NOT find a resume. Use it when you "
                         "already know from the run log that there was one: it turns the "
                         "heuristic from a guess into a check, and catches the case where "
                         "mtimes were flattened by a copy.")
    ap.add_argument("--accept-replayed-stream", action="store_true",
                    help="stage even though the source run was resumed without restoring "
                         "its RNG, so part of its batch stream is a replay. Required in "
                         "that case; the acknowledgement is recorded in the staged meta.")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    s_samples, s_meta = src / "samples.npy", src / "meta.json"
    for p in (s_samples, s_meta):
        if not p.exists():
            raise SystemExit(f"{p} 不存在 —— 这不是一个画完的里程碑目录")
    meta = json.loads(s_meta.read_text())

    # ---- 1. the milestone dir must be the milestone it is being used as -------------
    if int(meta.get("milestone", -1)) != a.milestone:
        raise SystemExit(f"{s_meta} 记的是 milestone={meta.get('milestone')!r},"
                         f"不是 {a.milestone} —— 拒绝当成这个预算的 base")
    if meta.get("generator") != "igfm":
        raise SystemExit(f"generator 是 {meta.get('generator')!r},不是 igfm")

    # ---- 2. it must be THIS design's THIS job on THIS cohort, re-derived ------------
    job_file = Path(a.design) / "jobs" / f"{a.job}.json"
    job = json.loads(job_file.read_text())
    fp = hashlib.sha1("\n".join(map(str, job["subjects"])).encode()).hexdigest()[:16]
    for key, want in (("job", job["name"]), ("cohort", str(a.cohort)),
                      ("subjects_sha1", fp)):
        if meta.get(key) != want:
            raise SystemExit(f"{s_meta} 的 {key} 是 {meta.get(key)!r},不是 {want!r} —— "
                             f"这份样本不是本设计本格子的 base")

    # ---- 3. the three checkable premises -------------------------------------------
    params = dict(meta.get("params") or {})
    src_steps = int(params.get("steps", params.get("total_iters", 0)))
    if src_steps < a.milestone:
        raise SystemExit(f"源 run 只训了 {src_steps} 步,取不到 {a.milestone} 的存档")
    _assert_no_lr_scheduler()
    sched = str(params.get("lambda_schedule") or _adapter_lambda_default())
    try:
        first_boundary = min(int(piece.split(":")[0]) for piece in sched.split(","))
    except ValueError as e:                                  # 拼错了就停,不要猜
        raise SystemExit(f"lambda_schedule {sched!r} 解析不了: {e}")
    if first_boundary <= src_steps:
        raise SystemExit(
            f"lambda_schedule 的第一个边界是 {first_boundary},不大于源 run 的 "
            f"{src_steps} 步 —— lam 在轨迹中途变过,那么「25000 步跑到第 "
            f"{a.milestone} 步」和「配置成 {a.milestone} 步的一次训练」不是同一条轨迹,"
            f"这份存档不能当成同预算的 base")
    save_every = int(params.get("save_every", 0))
    if save_every <= 0 or a.milestone % save_every:
        raise SystemExit(
            f"save_every={save_every} 除不尽 {a.milestone} —— iter-{a.milestone}.pt "
            f"不会是「恰好 {a.milestone} 步之后」的那个状态")

    # ---- 4. the checkpoint the samples came from must still say so -----------------
    ck_path = str(meta.get("from_checkpoint") or "")
    if not ck_path:
        raise SystemExit(f"{s_meta} 里没有 from_checkpoint,无法复核这份样本的来源")
    ckpt = Path(ck_path)
    if not ckpt.is_file():
        raise SystemExit(f"meta 里的存档 {ckpt} 不在盘上(或不是文件),无法复核")
    import torch
    ck_iter = int(torch.load(ckpt, map_location="cpu", weights_only=False)["iter"])
    if ck_iter != a.milestone:
        raise SystemExit(f"{ckpt} 里记的是 iter={ck_iter},不是 {a.milestone}")

    # ---- 5. the premise that is NOT safe to assume: was the source run resumed? -----
    resume = _detect_resume(ckpt.parent)
    replayed = bool(resume.get("resumed"))
    if replayed:
        first = resume["gaps"][0]
        print(f"[stage] ⚠️ 源 run 在第 {first['after_iter']} 步之后续过训"
              f"(该区间 {first['sec_per_iter']} 秒/步,中位数 "
              f"{resume['median_sec_per_iter']} 秒/步)")
        print(f"[stage]    2026-09-14 之前的 fit() 续训时不恢复 RNG,所以第 "
              f"{first['after_iter']} 步之后的批次是一次干净训练前段的【重放】。")
        print(f"[stage]    iter-{a.milestone}.pt 因此不是「配置成 {a.milestone} 步跑一次」"
              f"的终态。统计上无害(每个 job 本来就有各自的 job_seed),但不可复现。")
        if not a.accept_replayed_stream:
            raise SystemExit(
                "拒绝在没有明确认可的情况下把它摆成 base。确认要用就加 "
                "--accept-replayed-stream(这个认可会写进 staged meta.json)")
        print("[stage]    --accept-replayed-stream 已给,继续,并记进 meta。")
    elif not resume.get("checked"):
        print(f"[stage] ⚠️ 续训检查没做成:{resume.get('reason')}")
    if a.assert_resumed and not replayed:
        raise SystemExit(
            f"--assert-resumed 要求检出续训,但 mtime 扫描没检出:{resume}\n"
            f"已知 {ckpt.parent} 的训练日志里有 resuming 行,所以要么 mtime 被拷贝/恢复\n"
            f"抹平了(那么这套检查对这个目录失效),要么阈值不合适。先弄清楚,"
            f"不要让 meta.json 记下一个「未见续训」的假证据。")

    # ---- 6. K must be the job's own training-set size, re-derived from the cohort ---
    # Not read off the meta: K is what the attack matches on, and `loo/train.py` sets it
    # to the training-set size precisely so a released set is what a custodian would
    # release. A milestone drawn at some other K is not this campaign's base.
    X, sids, man = load_cohort(a.cohort)
    Xtr, _ = training_set(X, sids, job["subjects"])
    N, T, C = Xtr.shape
    S = np.load(s_samples, mmap_mode="r")
    if S.shape != (N, T, C):
        raise SystemExit(f"样本形状 {S.shape},而本 job 的训练集是 {(N, T, C)} —— "
                         f"K 或窗口长度对不上")
    if int(meta.get("K", N)) != N:
        raise SystemExit(f"meta 的 K={meta.get('K')},训练集是 {N}")
    if bool(np.isnan(np.asarray(S)).any()):
        raise SystemExit("样本里有 NaN,拒绝把它当数据摆出去")
    del S

    # ---- 7. idempotence: a second run must not silently swap the base out ----------
    d_samples, d_meta = dst / "samples.npy", dst / "meta.json"
    src_digest = _sha256(s_samples)
    if d_samples.exists() and not a.overwrite:
        if _sha256(d_samples) != src_digest:
            raise SystemExit(
                f"{d_samples} 已存在且和 {s_samples} 不同。26 个 include 可能已经在和它"
                f"配对 —— 换掉它就是悄悄换掉全部 26 个对照。确认要换再加 --overwrite")
        if d_meta.exists():
            print(f"[stage] {d_samples} 已存在且与源逐字节相同,meta 也在,不动它")
            return 0
        # 样本在、meta 不在 = 上一次在 rename 和写 meta 之间被杀。补 meta,别报成功了事。
        print(f"[stage] {d_samples} 与源相同但缺 {d_meta},补写 meta")
    else:
        dst.mkdir(parents=True, exist_ok=True)
        tmp = dst / "samples.tmp.npy"
        shutil.copyfile(s_samples, tmp)
        if _sha256(tmp) != src_digest:
            tmp.unlink(missing_ok=True)
            raise SystemExit("拷贝后的摘要和源不一致,已删除临时文件")
        tmp.rename(d_samples)                                # atomic: 没有半份样本

    # `steps` is rewritten to the milestone ON PURPOSE: the 26 include models are trained
    # with steps=m and `igfm_priv_loo_cell.body.sh` compares the two key by key before it
    # starts any of them. (`run_loo.py` and `run_attack.py` do NOT check it -- run_loo's
    # skip path compares only `subjects_sha1`, and run_attack never opens a meta.json. So
    # this field is guarded by the campaign body script and by nothing else.) Everything
    # about where the weights really came from is recorded alongside, not instead.
    out_params = dict(params)
    out_params["steps"] = int(a.milestone)
    out_params["workdir"] = str(dst)
    staged = dict(
        job=job["name"], role=job["role"], target=job.get("target"),
        group=job.get("group"), job_file=str(job_file), subjects_sha1=fp,
        n_subjects=job["n_subjects"], n_train_windows=int(N), K=int(N),
        T=int(T), C=int(C), generator="igfm", params=out_params,
        base_seed=int(a.seed), job_seed=_job_seed(a.seed, job["name"]),
        cohort=str(a.cohort), cohort_windows=int(man["n_windows"]),
        sample_seconds=meta.get("sample_seconds"),
        sample_range=meta.get("sample_range"),
        budget_iters=int(a.milestone),
        staged_from=str(src), source_checkpoint=str(ckpt),
        source_run_steps=src_steps, samples_sha256=src_digest,
        source_run_resume_scan=resume,
        source_stream_replayed=replayed,
        bit_reproducible=False,
        fit_seconds=None,
        fit_seconds_note=(
            "No training happened here; the weights are iter-%d.pt of the %d-step run at "
            "%s. Do NOT size a campaign from this directory."
            % (a.milestone, src_steps, ckpt.parent)),
        equivalence_note=(
            ("CHECKED against the adapter source on this run: no LR scheduler, "
             "lambda_schedule's first boundary (%d) is past the whole %d-step "
             "trajectory, and save_every (%d) divides the milestone. "
             % (first_boundary, src_steps, save_every))
            + ("NOT met: the source run was resumed (see source_run_resume_scan) by a "
               "fit() that did not restore the RNG, so iterations after the resume point "
               "replayed the batch stream. These weights are an %d-iteration model on "
               "this data, but NOT the model an uninterrupted %d-step run would produce, "
               "and they are not reproducible from a command line. Accepted deliberately "
               "via --accept-replayed-stream: every job already carries its own job_seed "
               "and therefore its own batch stream, so a replayed stretch adds no "
               "systematic difference between this base and the include models."
               % (a.milestone, a.milestone)
               if replayed else
               "The mtime scan detected NO resume (it flags an inter-checkpoint interval "
               "above 3x the median, so a requeue faster than about twice the "
               "checkpoint interval would be missed, and copied mtimes would hide "
               "everything). On that evidence iter-%d.pt is the end state an %d-step "
               "run would reach." % (a.milestone, a.milestone))),
        stage_note=(
            "Released set drawn by scripts/igfm_budget_curve.py, which seeds immediately "
            "before sample(); an uninterrupted run samples with the RNG left by training. "
            "Same weights, same distribution, a different draw. Both are unbiased draws "
            "from the same model, so the arm AUC acquires no tilt -- but it does NOT "
            "'cancel' (attack.statistic.gap_for_pair computes d_out from each target's "
            "own windows), and subject_auc.py / leak_locus.py report absolute per-subject "
            "levels where the added variance does not divide out."),
    )
    d_meta.write_text(json.dumps(staged, indent=2))
    print(f"[stage] {src} -> {dst}")
    print(f"        milestone={a.milestone} K={N:,} T={T} C={C} "
          f"subjects_sha1={fp} sha256={src_digest[:16]}…")
    print(f"        源 run 续训扫描: {'检出重放' if replayed else '未见续训'}")
    print(f"        run_loo 之后会凭 subjects_sha1 跳过这个目录,不会重训 base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
