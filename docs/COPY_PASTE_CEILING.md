# Why the copy-paste positive control scores 0.816, not 1.0

*A note on reading the membership-inference sanity check.*

## The puzzle

`copy_paste` is the positive control for our membership-inference attack. It is not a
generative model at all: it "generates" by replaying real training windows verbatim. The
attack should therefore find every member trivially, and the per-subject AUC should be
**1.0**. It is not. It reads **0.816**, and it does so reproducibly.

The natural reading of that gap is that something is wrong with the attack, the labels,
or the aggregation — which is exactly what a sanity check is supposed to rule out. It
turns out nothing is wrong. The ceiling really is 0.816, for a reason that has nothing
to do with membership inference, and knowing why is necessary to read the number.

## The intuition that says it should be 1.0

Suppose 100 subjects with 10 windows each, so 1,000 training windows.

- `include_t` trains on all 1,000 windows, including subject *t*'s 10.
- `base` trains on the other 99 subjects, 990 windows, without *t*.

For each of *t*'s windows, measure the distance to the nearest released sample:

| | `d_in` (vs `include_t`) | `d_out` (vs `base`) |
|---|---|---|
| every window of *t* | **0** — its own copy is in there | **> 0** — *t* was never seen |

A classifier that says "nearest distance is zero ⇒ member" separates the two perfectly.
AUC = 1.0.

**This reasoning is correct**, and the classifier is the right one — it is exactly the
`min` reduction the frozen statistic uses. The reasoning has one false premise.

## The false premise: the released set is not the training set

The generator does not release the training set. It releases *n* draws **with
replacement** from it (`src/cgmoutlier/generators/copy_paste.py:26`):

```python
idx = rng.integers(0, self._X.shape[0], size=int(n))
return self._X[idx].copy()
```

With `n = N`, the released set is a *multiset*, not a permutation. Some windows appear
twice or three times; others never appear at all. Drawing 10 indices from 10 windows:

```
draw 1   counts per window [0,0,0,2,1,0,1,3,1,2]   never drawn: w0 w1 w2 w5   (4/10)
draw 2                     [2,0,0,0,1,0,2,3,1,1]   never drawn: w1 w2 w3 w5   (4/10)
draw 3                     [2,0,1,1,0,2,0,1,3,0]   never drawn: w1 w4 w6 w9   (4/10)
draw 4                     [1,0,0,3,1,0,1,1,1,2]   never drawn: w1 w2 w5      (3/10)
```

The `3`s and `2`s are the point. Ten draws must total ten; every window that is taken
more than once forces another to be taken not at all.

A given window is missed with probability `(1 - 1/N)^n`, which for `n = N` converges to
`1/e = 0.3679`. So **63.2% of training windows appear at least once, and 36.8% do not.**

## Verifying it on the artefacts

If the released set of `n = N` windows really is a bootstrap resample, it contains
`N x 0.632` *distinct* windows. That needs only `samples.npy` to count:

| run | N | n released | distinct | fraction |
|---|---|---|---|---|
| `base` | 179,084 | 179,084 | 113,344 | **0.6329** |
| `exclude_1013` | 178,910 | 178,910 | 113,143 | 0.6324 |
| `exclude_1026` | 178,974 | 178,974 | 113,005 | 0.6314 |
| `exclude_1059` | 179,017 | 179,017 | 113,216 | 0.6324 |
| `exclude_1109` | 178,931 | 178,931 | 112,812 | 0.6305 |

Against a prediction of `1 - 1/e = 0.6321`.

## Where 0.816 comes from

For a subject with *m* windows, about `0.632 m` of them have a verbatim copy in the
`include_t` release and about `0.368 m` do not.

- **Copied windows.** `d_in = 0` while `d_out > 0`, so the comparison is decided
  correctly every time. Contribution **1.0**.
- **Missed windows.** The nearest thing in the `include_t` release is somebody else's
  window, and so is the nearest thing in `base`. The two distances are drawn from
  effectively the same distribution. Contribution **0.5**.

```
AUC  =  0.632 x 1.0  +  0.368 x 0.5  =  0.816
```

Equivalently: **63.2% of a subject's windows can be caught red-handed; the remaining
36.8% can only be guessed at.**

## A worked example on real data

Subject `973`, 67 windows, scored against `include_973` and `base`:

```
d_in sorted:  0 0 0 0 0 0 0 0 0 0 0 0 ...  then jumps above 0.11
break point:  43 / 67  =  0.642            (predicted 0.632)
```

| window | `d_in` | `d_out` | |
|---|---|---|---|
| 0 | 0.0000 | 0.1224 | caught |
| 1 | 0.0000 | 0.1061 | caught |
| 2 | 0.1641 | 0.1764 | guess |
| 3 | 0.1527 | 0.1696 | guess |
| 4 | 0.1121 | 0.1136 | guess |
| 5 | 0.0000 | 0.1572 | caught |

- 43 caught windows: `d_in = 0` against `d_out ~ 0.13`
- 24 guessed windows: `d_in ~ 0.154` against `d_out ~ 0.148`

```
predicted   0.642 x 1.0 + 0.358 x 0.5  =  0.821
measured                                  0.811
```

The 0.01 shortfall is because the guessed windows land slightly *below* chance: their
median `d_out` (0.1478) is a little smaller than their median `d_in` (0.1537).

## A trap for anyone reproducing this

Counting copies with `d_in < 1e-9` gives **25/67 = 0.373**, and the arithmetic does not
close. `window_distances` computes `sqrt(|r|^2 + |s|^2 - 2 r.s)` in float32; for an exact
copy those three terms cancel only to float32 precision, so a verbatim match reads around
`1e-4`, not `0`. The distribution is sharply bimodal — copies below `1e-3`, nearest
non-copies above `0.11`, two orders of magnitude apart — so threshold in the gap
(`d_in < 0.01`), not at machine zero.

## Three things this ceiling is not

**It is not a ceiling on membership inference.** It is entirely an artefact of the
bootstrap interface. A generator that released its training set without replacement
would score 1.0. Changing `sample()` to `rng.permutation(N)[:n]` would recover the clean
1.0 that a positive control ought to give.

**It is not a ceiling on the arm-level AUC.** The arm AUC asks a different question —
whether *outliers* leak more than *matched normals* — and `copy_paste` scores **0.543**
there, which is correct rather than a failure. Copy-paste memorises everyone
indiscriminately, so both arms leak equally and neither separates. The positive control
validates the per-subject detection pipeline, not the group contrast.

**It is not stable at small sample sizes.** The number of copied windows is
`Binomial(m, 0.632)`, so the spread depends strongly on how many windows a subject has:

| windows *m* | expected copied | s.d. | typical swing in the fraction |
|---|---|---|---|
| 4 | 2.5 | 0.97 | **+/- 24 points** |
| 10 | 6.3 | 1.5 | +/- 15 |
| 24 | 15.2 | 2.4 | +/- 10 |
| 67 | 42.4 | 3.9 | +/- 6 |

0.816 is the large-*m* expectation. For a subject with 4 windows the per-subject ceiling
is anywhere from 0.5 (none copied) to 1.0 (all four copied). The published 0.8187 was
measured on a cohort averaging ~180 days per subject, where *m* is large and the value
is stable.

## Summary

| quantity | value |
|---|---|
| fraction of training windows released at least once | 0.632 (measured 0.6305–0.6329) |
| per-subject AUC ceiling for `copy_paste`, large *m* | **0.816** |
| measured per-subject AUC, published cohort | 0.8187 |
| measured per-subject AUC, subject 973 | 0.811 |
| arm-level AUC for `copy_paste` (frozen `min x mean`) | 0.543 — correct, not a failure |

Reproduce with `scripts/report/copypaste_ceiling.py` and
`scripts/report/copypaste_example.py`.
