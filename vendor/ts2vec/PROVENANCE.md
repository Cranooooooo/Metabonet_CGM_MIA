# TS2Vec

Vendored from the `tsgen_metrics` package's `_vendor/ts2vec`, which itself vendors the
official TS2Vec implementation (Yue et al., AAAI 2022).

**Vendored rather than imported** because the previous version of this code reached it
through a hard-coded absolute path (`/home/ling/workspace/tsgen_metrics`). That works
on one machine and nowhere else, which defeats the point of a cloneable repository.

**This encoder is not the one any quality metric uses.** Context-FID also uses TS2Vec,
but a separate fit with separate weights, and the two must stay separate: the quality
metric has to stay faithful to its published implementation (including refitting per
call), while the attack needs one frozen space shared across every fold. Sharing one
fit would make "good quality" and "resists attack" two projections of the same space,
so a privacy-utility plot would show a relationship created by construction.

Residual coupling, disclosed rather than engineered away: same architecture, same
source data, different weights. If TS2Vec fails to encode some distinction in CGM,
the metric and the attack lose it together.
