"""Turn an outlier list into the exact set of training runs the experiment needs.

The design is stated in docs/DESIGN.md. For a target subject the comparison is between
two generators whose training sets differ by that subject and nothing else.

SYMMETRIC (`symmetric=True`, the default and what the study runs)
----------------------------------------------------------------
The controls are drawn OUT of the normal pool before the base model is built, so every
target -- outlier or control -- is compared the same way:

    B             the background: normals minus the drawn controls
    B + t         background plus target t, for each outlier AND each control

Every pair is (B+t, B). One base model is the non-member side for all 40 pairs.

ASYMMETRIC (`symmetric=False`, kept only so the copy_paste validation design in
`results/design/` stays reproducible)
-------------------------------------------------------------------------------
    M0            all N normals                     the base model
    M0 + s        all N normals, plus outlier s     s is a member
    M0 - c        all N normals, minus control c    c is not a member

An outlier's pair is (M0+s, M0); a control's pair is (M0, M0-c).

WHY THE ASYMMETRIC FORM IS NOT USED FOR THE RESULT
--------------------------------------------------
It looks like a pure saving -- one base serving both arms halves the runs -- but the
base enters the two arms with OPPOSITE SIGN. It is the non-member side for outliers and
the member side for controls, so if that single training run happens to release samples
a little further from everything (call the offset d), then

    outlier gap  = d(s, S_base) - d(s, S_M0+s)          shifts by  +d
    control gap  = d(c, S_M0-c) - d(c, S_base)          shifts by  -d

The two arms move apart by 2d for a reason that is a property of one training run and
not of membership -- and d is shared within each arm, so averaging over 20 targets does
not remove it. That is the same quantity the study is trying to measure. Under
`symmetric=True` the base sits in d_OUT for every pair, so d shifts both arms together
and cancels in the arm comparison.

Two smaller asymmetries the symmetric form also removes: the arms compared N+1 vs N and
N vs N-1 (now both |B|+1 vs |B|), and each `M0-c` had its own distinct background of
N-1 subjects (now all 41 runs share one background B).

WHAT NEITHER FORM REMOVES
-------------------------
Every target still shares one base model, so measurements within a replicate are
correlated and a p-value computed as if the 20 were independent is overstated. The way
out is replicates: an independent control draw gives an independent background, base and
arm difference, and those replicate-level differences are what a p-value can rest on.
`match_controls` is seeded for exactly that, and `exclude` keeps the draws disjoint.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def safe(sid: str) -> str:
    """A subject id as a single path component.

    Subject ids became `(source_file, id)` pairs once it turned out that `id` is only
    unique within a study (docs/PITFALLS.md 13), and they are spelled "DCLP3/138". A
    job's `name` is written as `<name>.json` and later as a run DIRECTORY, so an id
    containing a slash silently asks for a subdirectory that does not exist:

        FileNotFoundError: .../jobs/include_DCLP3/138.json

    Only the NAME is escaped. `target` keeps the true id, because that is what indexes
    the cohort. The two must not be conflated: reversing this mapping is not needed
    anywhere, and a study whose own name contains "__" would make it ambiguous.

    The separator itself is deliberately not changed. `np.unique(sids)` sorts, the
    global window order follows that sort, and the C-group methods sample the reference
    pool by global index -- so renaming "DCLP3/138" to "DCLP3-138" would resample the
    reference pool and produce a different outlier list.
    """
    return str(sid).replace("/", "__")


@dataclass
class Job:
    """One generator training run."""
    name: str
    role: str                     # "base" | "include" | "exclude"
    target: str | None            # the subject this run is about; None for the base
    subjects: list[str]           # the exact training set
    group: str | None = None      # "outlier" | "control"; None for the base. Under
                                  # symmetric=True both arms are `include` runs and
                                  # this is the only thing that tells them apart.
    n_subjects: int = 0

    def __post_init__(self):
        self.n_subjects = len(self.subjects)


@dataclass
class Design:
    jobs: list[Job]
    outliers: list[str]
    controls: list[str]
    normals: list[str]            # every non-outlier in the cohort
    background: list[str] = field(default_factory=list)   # what the base trains on
    symmetric: bool = True
    pairs: list[dict] = field(default_factory=list)
    matching: dict = field(default_factory=dict)
    borderline: dict = field(default_factory=dict)


def match_controls(days: dict[str, int], outliers, pool, seed=2026, tol=0.15,
                   n_candidates=1, exclude=()):
    """Pick one control per outlier, matched on day count.

    Day count is not a nuisance detail here. The 20 stable outliers have a median of
    142 complete days against the pool's 173, and day count drives nearest-neighbour
    distance on its own -- more days means more chances for some day to sit close to a
    released sample, whether or not the subject was a member. Random controls would put
    that difference between the two arms and it would be indistinguishable from the
    effect being measured. This is why the draw is matched and NOT a plain shuffle.

    Greedy nearest-neighbour without replacement, outliers taken in order of how rare
    their day count is, so the hardest to match choose first.

    `n_candidates` is how the draw is randomised WITHOUT giving up the matching: the
    outlier's k nearest still-available candidates by day count are collected and one is
    picked uniformly with `seed`. k=1 reproduces the deterministic greedy draw. Anything
    larger trades a little matching quality for a genuinely different draw per replicate
    -- read `median_rel_gap` in the report to see what it cost, and note that the
    candidates are the k nearest AVAILABLE, so a rare day count still gets its closest
    match rather than being widened to a fixed tolerance band.

    `exclude` removes subjects already used as controls by another replicate, which is
    what makes replicate draws disjoint.

    NOTE `seed` was accepted and never used before 2026-08-07. Every call returned the
    same greedy draw, so replicates that varied only the seed would have shared one
    control set and measured far less than they appeared to.

    Returns (controls, report); `report` carries the realised difference, which should
    be read rather than assumed to be small -- if the outliers occupy the extreme of the
    day distribution there may be no close match available at all.
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    blocked = set(outliers) | set(exclude)
    pool = [s for s in pool if s not in blocked]
    if len(pool) < len(outliers):
        raise ValueError(f"{len(pool)} candidates for {len(outliers)} controls")

    rng = np.random.default_rng(seed)
    dpool = np.array([days[s] for s in pool], float)
    # rarity = distance to the pool's density at that day count; scarce first
    order = sorted(outliers, key=lambda s: -np.min(np.abs(dpool - days[s])))

    taken, chosen, rows = set(), {}, []
    for s in order:
        d = days[s]
        avail = [(abs(dpool[j] - d), c) for j, c in enumerate(pool) if c not in taken]
        # sort by gap, then by id so ties do not depend on dict/glob ordering
        avail.sort(key=lambda t: (t[0], t[1]))
        k = min(n_candidates, len(avail))
        bestgap, best = avail[int(rng.integers(k))]
        taken.add(best)
        chosen[s] = best
        rows.append(dict(outlier=s, control=best, outlier_days=int(d),
                         control_days=int(days[best]),
                         rel_gap=round(bestgap / max(d, 1), 4)))

    gaps = np.array([r["rel_gap"] for r in rows])
    report = dict(
        pairs=rows, tol=tol, seed=seed, n_candidates=n_candidates,
        n_excluded=len(set(exclude)),
        max_rel_gap=float(gaps.max()), median_rel_gap=float(np.median(gaps)),
        n_over_tol=int((gaps > tol).sum()),
        outlier_days=dict(median=float(np.median([days[s] for s in outliers])),
                          min=int(min(days[s] for s in outliers)),
                          max=int(max(days[s] for s in outliers))),
        pool_days=dict(median=float(np.median(dpool)),
                       min=int(dpool.min()), max=int(dpool.max())),
    )
    if report["n_over_tol"]:
        report["warning"] = (
            f"{report['n_over_tol']} of {len(rows)} controls are further than {tol:.0%} "
            f"in day count from their outlier. Day count then differs between the arms "
            f"and any gap in the attack statistic is partly that.")
    # controls returned in the order of `outliers`, not of `order`
    return [chosen[s] for s in outliers], report


def build(all_subjects, outliers, days, seed=2026, tol=0.15, n_controls=None,
          symmetric=True, n_candidates=1, exclude=(), borderline=None):
    """-> Design. Pure bookkeeping: no data is read and no model is trained.

    `symmetric=True` draws the controls out of the pool before the base is built, so
    every pair is (background + target, background). See the module docstring for why
    the alternative puts the base model's own offset into the two arms with opposite
    sign. `symmetric=False` reproduces that older form and is kept only so the
    copy_paste validation design stays rebuildable.

    `borderline` is an optional {subject: n_seeds_flagged} map for subjects that some
    outlier-detection seed flagged but not all. They are NOT removed from the control
    pool: removing them would leave the controls a hand-picked "most normal" subset and
    tighten the empirical null, which makes separation easier to find rather than
    harder. They are recorded instead, so whether the drawn ones behave differently is
    something the analysis can check rather than assume.
    """
    all_subjects = [str(s) for s in all_subjects]
    outliers = [str(s) for s in outliers]
    missing = set(outliers) - set(all_subjects)
    if missing:
        raise ValueError(f"outliers not in the cohort: {sorted(missing)}")

    normals = [s for s in all_subjects if s not in set(outliers)]
    controls, report = match_controls(days, outliers, normals, seed=seed, tol=tol,
                                      n_candidates=n_candidates, exclude=exclude)
    if n_controls is not None:
        controls = controls[:n_controls]

    if symmetric:
        # the controls leave the background, so all 1 + 2n runs share one training set
        # apart from their own target
        drawn = set(controls)
        background = [s for s in normals if s not in drawn]
        jobs = [Job(name="base", role="base", target=None, subjects=background)]
        jobs += [Job(name=f"include_{safe(s)}", role="include", target=s,
                     group="outlier", subjects=background + [s]) for s in outliers]
        jobs += [Job(name=f"include_{safe(c)}", role="include", target=c,
                     group="control", subjects=background + [c]) for c in controls]
        pairs = [dict(target=t, group=g, member="include_" + safe(t), nonmember="base")
                 for t, g in ([(s, "outlier") for s in outliers] +
                              [(c, "control") for c in controls])]
    else:
        background = normals
        jobs = [Job(name="base", role="base", target=None, subjects=normals)]
        for s in outliers:
            jobs.append(Job(name=f"include_{safe(s)}", role="include", target=s,
                            group="outlier", subjects=normals + [s]))
        for c in controls:
            jobs.append(Job(name=f"exclude_{safe(c)}", role="exclude", target=c,
                            group="control",
                            subjects=[x for x in normals if x != c]))

        pairs = ([dict(target=s, group="outlier", member="include_" + safe(s),
                       nonmember="base") for s in outliers] +
                 [dict(target=c, group="control", member="base",
                       nonmember="exclude_" + safe(c)) for c in controls])

    drawn_borderline = {c: int(n) for c, n in (borderline or {}).items()
                        if c in set(controls)}
    if drawn_borderline:
        report["borderline_controls"] = (
            f"{len(drawn_borderline)} of {len(controls)} controls were flagged as "
            f"outliers by some but not all detection seeds: "
            f"{' '.join(sorted(drawn_borderline))}. They are kept on purpose (see "
            f"build's docstring); check them separately in the analysis.")

    return Design(jobs=jobs, outliers=outliers, controls=controls, normals=normals,
                  background=background, symmetric=symmetric, pairs=pairs,
                  matching=report, borderline=drawn_borderline)


def save(design: Design, out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps(dict(
        n_jobs=len(design.jobs), n_outliers=len(design.outliers),
        n_controls=len(design.controls), n_normals=len(design.normals),
        n_background=len(design.background), symmetric=design.symmetric,
        outliers=design.outliers, controls=design.controls,
        borderline_controls=design.borderline,
        pairs=design.pairs, matching=design.matching,
        jobs=[{k: v for k, v in asdict(j).items() if k != "subjects"}
              for j in design.jobs],
    ), indent=2))
    js = out / "jobs"
    js.mkdir(exist_ok=True)
    for j in design.jobs:
        (js / f"{j.name}.json").write_text(json.dumps(asdict(j), indent=2))
    return out
