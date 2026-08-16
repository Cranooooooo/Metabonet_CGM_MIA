#!/usr/bin/env python
"""Put the raw MetaboNet parquet where `build_cohort.py` expects it.

    python scripts/fetch_data.py --check                  # what do I already have?
    python scripts/fetch_data.py --from /path/to/file.parquet
    python scripts/fetch_data.py --url https://.../metabonet_public.parquet

WHICH RELEASE THIS IS. `metabonet_public.parquet` is the immediately-available portion
of MetaboNet -- the studies whose contributors released them with no data use
agreement. The DUA-gated studies are not in it and are not needed here. That is why the
derived cohort under `data/cohort/` can be committed to this repository at all.

Nothing is downloaded without an explicit `--url` or `--from`. There is no default
mirror: a raw CGM file is not something a script should acquire on your behalf because
a Makefile target ran.

After this, build the cohort (windows, per-cohort normalisation, day floor):

    python scripts/build_cohort.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "metabonet_public.parquet"

# Recorded from the file the checked-in results were built from, so a different
# release announces itself instead of quietly shifting every number downstream.
KNOWN = {
    "n_rows": None,        # filled on first successful --check against the real file
    "sha256": None,
}


def sha256(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def describe(p: Path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    md = pq.ParquetFile(p).metadata
    print(f"  {p}")
    print(f"  {p.stat().st_size / 1e9:.2f} GB | {md.num_rows:,} rows | "
          f"{md.num_columns} columns")
    print(f"  columns: {list(pq.ParquetFile(p).schema.names)[:12]}")
    digest = sha256(p)
    print(f"  sha256: {digest}")
    if KNOWN["sha256"] and digest != KNOWN["sha256"]:
        print("\n  WARNING: this is not the file the checked-in results came from.\n"
              "  Rebuild the cohort and rerun the methods; do not mix the two.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from", dest="src", help="copy or symlink an existing parquet")
    g.add_argument("--url", help="download over https to data/raw/")
    ap.add_argument("--symlink", action="store_true",
                    help="with --from, link instead of copying 1.3 GB")
    ap.add_argument("--check", action="store_true", help="describe what is present")
    a = ap.parse_args()

    DEST.parent.mkdir(parents=True, exist_ok=True)

    if a.check or (not a.src and not a.url):
        if DEST.exists() or DEST.is_symlink():
            print("raw parquet present:")
            describe(DEST.resolve())
        else:
            print(f"no raw parquet at {DEST}\n"
                  f"  supply one with --from <path> or --url <https://...>")
        cohorts = sorted((ROOT / "data" / "cohort").glob("*/manifest.json"))
        print(f"\nderived cohorts ({len(cohorts)}):")
        for m in cohorts:
            import json
            d = json.loads(m.read_text())
            tag = "  [SYNTHETIC]" if d.get("fake") else ""
            print(f"  {m.parent.name}: {d['n_subjects']} subjects, "
                  f"{d['n_windows']:,} windows{tag}")
        print("\nThe derived cohorts are committed. You only need the raw parquet to\n"
              "build a DIFFERENT cohort -- a different day floor, a different draw.")
        return 0

    if DEST.exists() or DEST.is_symlink():
        print(f"{DEST} already exists; remove it first.", file=sys.stderr)
        return 1

    if a.src:
        src = Path(a.src).expanduser().resolve()
        if not src.exists():
            print(f"no such file: {src}", file=sys.stderr)
            return 1
        if a.symlink:
            DEST.symlink_to(src)
            print(f"linked {DEST} -> {src}")
        else:
            print(f"copying {src.stat().st_size / 1e9:.2f} GB ...")
            shutil.copy2(src, DEST)
    else:
        import urllib.request
        print(f"downloading {a.url} ...")
        tmp = DEST.with_suffix(".partial")
        urllib.request.urlretrieve(a.url, tmp)
        tmp.rename(DEST)

    describe(DEST.resolve())
    print("\nnext:  python scripts/build_cohort.py --config configs/data.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
