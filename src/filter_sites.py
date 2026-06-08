#!/usr/bin/env python3
"""Filter a sites CSV to selected LIDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_lids(text: str) -> list[str]:
    return [x.strip().upper() for x in text.replace(";", ",").split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter sites CSV by comma-separated LIDs.")
    parser.add_argument("--sites", required=True)
    parser.add_argument("--lids", default="", help="Comma-separated LIDs. Blank copies all rows.")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N limit after LID filtering.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.sites, dtype=str).fillna("")
    if "lid" not in df.columns:
        raise SystemExit("sites file must contain a lid column")

    lids = parse_lids(args.lids)
    if lids:
        wanted = set(lids)
        df["_lid_upper"] = df["lid"].astype(str).str.upper().str.strip()
        missing = [lid for lid in lids if lid not in set(df["_lid_upper"])]
        if missing:
            raise SystemExit(f"Requested LIDs not found in sites file: {', '.join(missing)}")
        df = df[df["_lid_upper"].isin(wanted)].drop(columns=["_lid_upper"])
        order = {lid: i for i, lid in enumerate(lids)}
        df = df.assign(_sort=df["lid"].astype(str).str.upper().map(order)).sort_values("_sort").drop(columns=["_sort"])

    if args.limit:
        df = df.head(args.limit)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    if not df.empty:
        print("LIDs:", ", ".join(df["lid"].astype(str).tolist()))


if __name__ == "__main__":
    main()
