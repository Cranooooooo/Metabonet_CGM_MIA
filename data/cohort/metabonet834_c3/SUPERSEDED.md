# Do not use this cohort

Built 2026-08-14 and superseded the same day by `../metabonet_sid_c3`. Two defects,
either one disqualifying:

1. **Wrong subject key.** It keys on `id`, which is unique only within a source study
   (`docs/PITFALLS.md` §13), so 27% of its subjects are composites of 2–9 people.
2. **Duplicated windows.** Its builder assumed the file was in `(id, date)` order and
   carried only the last group across each batch boundary. The file is ordered by
   source instead, so **5,919 (subject, day) pairs were emitted twice** —
   `scripts/verify_cohort_multi.py` returns `verdict: REBUILD`.

The corruption is visible in the manifest without any tooling: this cohort's CGM
constants are **mean 96.52, sd 71.10 mg/dL**, against **145.94 / 57.65** for the
corrected build and **144.78 / 56.94** for the shipped `metabonet875`. A mean glucose
of 96 mg/dL across a type 1 diabetes cohort is not plausible.

Kept only so the numbers above stay checkable. Delete when that is no longer useful.
