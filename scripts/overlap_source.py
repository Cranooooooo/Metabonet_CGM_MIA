#!/usr/bin/env python
"""How much of the CGM / CGM+basal+bolus outlier overlap is forced by design?

Group A -- A1 to A4 -- is glucose-only in both runs because its metrics are the
Battelino consensus and a basal rate has no time in range. Four of the thirteen votes
are therefore identical across the two runs, and a subject those four carry appears in
both lists whatever the extra channels do. The reported overlap is a floor on the
effect of adding channels; this says how much of it is the floor.
"""
import json
from pathlib import Path

R = Path("results/outliers_sid_c3")
A = {"A1", "A2", "A3", "A4"}
cmp_ = json.loads((R / "comparison.json").read_text())


def flagged(side):
    c = json.loads((R / side / "seed2026" / "consensus.json").read_text())
    return c["flagged_by"]


f1, f3 = flagged("stability_1ch"), flagged("stability_3ch")

print(f"{'subject':>14}{'1ch votes':>11}{'of which A':>12}"
      f"{'3ch votes':>11}{'of which A':>12}  verdict")
forced = carried = 0
for s in cmp_["both"]:
    v1, v3 = f1.get(s, []), f3.get(s, [])
    a1, a3 = len(set(v1) & A), len(set(v3) & A)
    non_a3 = len(v3) - a3
    if non_a3 >= 7:
        carried += 1
        v = "survives without group A"
    else:
        forced += 1
        v = f"needs group A ({non_a3} non-A votes)"
    print(f"{s:>14}{len(v1):>11}{a1:>12}{len(v3):>11}{a3:>12}  {v}")

print(f"\nof the {len(cmp_['both'])} shared subjects:")
print(f"  {carried} clear 7 votes in the three-channel run WITHOUT any group A vote")
print(f"  {forced} need group A, so that much of the overlap is forced")

print(f"\nfound only by the three channels ({len(cmp_['only_b'])}):")
for s in cmp_["only_b"]:
    v3 = sorted(f3.get(s, []))
    print(f"  {s:>14} {len(v3)} votes  {v3}")
