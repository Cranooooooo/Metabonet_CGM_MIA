#!/usr/bin/env python
"""Pick the N targets a design will use, from a seed-stability run.

`always` is the list that survives every seed, which is the list worth carrying. Within
it the ranking is by consensus vote count in the primary seed, ties broken by id so the
choice is reproducible.

`--force` puts a subject at the front regardless of its vote count. Subject 1142 is
forced into every design in this project: he is the one who has survived a cohort
rebuild, a subject-key correction, a channel change and a change of window length, and
a cell that did not contain him could not be compared with the others.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability", required=True)
    ap.add_argument("--consensus", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--all", action="store_true",
                    help="take every subject in `always`. The count is then a property "
                         "of the cohort rather than a number someone picked, and a "
                         "stable outlier left out is power thrown away for nothing -- "
                         "the base is shared, so each extra target costs one include "
                         "model and one matched control, and buys a wider arm.")
    ap.add_argument("--force", default=None)
    a = ap.parse_args()

    always = list(json.loads(Path(a.stability).read_text())["always"])
    votes = json.loads(Path(a.consensus).read_text()).get("votes", {})
    ranked = sorted(always, key=lambda s: (-votes.get(s, 0), s))

    n = len(ranked) if a.all else a.n
    pick = []
    if a.force and a.force in ranked:
        pick.append(a.force)
    elif a.force:
        print(f"[targets] WARNING: {a.force} is not in `always` and cannot be forced",
              file=sys.stderr)
    pick += [s for s in ranked if s not in pick][:n - len(pick)]

    if len(pick) < n:
        print(f"[targets] only {len(pick)} stable outliers available, asked for {n}",
              file=sys.stderr)
    print(f"[targets] {len(always)} stable; taking {len(pick)}", file=sys.stderr)
    for s in pick:
        print(f"  {s:20} votes {votes.get(s, 0)}", file=sys.stderr)
    print(",".join(pick))
    return 0


if __name__ == "__main__":
    sys.exit(main())
