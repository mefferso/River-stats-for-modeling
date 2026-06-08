#!/usr/bin/env python3
"""Report raw-data, model availability, and profile-training status by river forecast point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import nwps_multigage_model as base


def summarize_models(lid: str) -> dict[str, object]:
    old = list(base.MODEL_DIR.glob(f"{lid}_ridge_model.joblib"))
    event_models = list(base.MODEL_DIR.glob(f"{lid}_*_ridge_model.joblib"))
    all_models = sorted(set(old + event_models))
    labels = []
    for path in all_models:
        name = path.name
        if name == f"{lid}_ridge_model.joblib":
            labels.append("legacy")
        else:
            label = name.removeprefix(f"{lid}_").removesuffix("_ridge_model.joblib")
            labels.append(label)
    return {
        "model_count": len(all_models),
        "model_labels": ";".join(sorted(labels)),
    }


def read_profile_summary(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {}
    if "lid" not in df.columns:
        return {}
    df["lid"] = df["lid"].astype(str).str.upper().str.strip()
    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        lid = str(row.get("lid", "")).upper().strip()
        if not lid:
            continue
        out[lid] = {
            "profile_run_label": row.get("run_label", ""),
            "profile_status": row.get("status", ""),
            "profile_event_count": row.get("event_count", ""),
            "profile_holdout_events": row.get("holdout_events", ""),
            "profile_mae_ft": row.get("mae_ft", ""),
            "profile_bias_ft": row.get("bias_ft", ""),
            "profile_r2": row.get("r2", ""),
        }
    return out


def command_status(args: argparse.Namespace) -> None:
    base.ensure_dirs()
    sites = base.read_sites(args.sites)
    profile = read_profile_summary(Path(args.profile_summary))
    rows = []
    for _, row in sites.iterrows():
        lid = str(row["lid"]).upper()
        raw = base.DATA_RAW / f"{lid}_usgs_stage.csv"
        raw_rows: int | str = ""
        raw_start = ""
        raw_end = ""
        latest_stage = ""
        max_stage = ""
        if raw.exists():
            try:
                df = pd.read_csv(raw, usecols=["datetime", "stage_ft"])
                raw_rows = len(df)
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                df["stage_ft"] = pd.to_numeric(df["stage_ft"], errors="coerce")
                df = df.dropna(subset=["datetime", "stage_ft"]).sort_values("datetime")
                if not df.empty:
                    raw_start = df["datetime"].min().date().isoformat()
                    raw_end = df["datetime"].max().date().isoformat()
                    latest_stage = round(float(df.iloc[-1]["stage_ft"]), 2)
                    max_stage = round(float(df["stage_ft"].max()), 2)
            except Exception as exc:  # noqa: BLE001
                raw_rows = f"error: {exc}"

        model_info = summarize_models(lid)
        profile_info = profile.get(
            lid,
            {
                "profile_run_label": "",
                "profile_status": "",
                "profile_event_count": "",
                "profile_holdout_events": "",
                "profile_mae_ft": "",
                "profile_bias_ft": "",
                "profile_r2": "",
            },
        )
        rows.append(
            {
                "lid": lid,
                "name": row.get("name", ""),
                "usgs_site": row.get("usgs_site", ""),
                "action_stage_ft": row.get("action_stage_ft", ""),
                "minor_stage_ft": row.get("minor_stage_ft", ""),
                "moderate_stage_ft": row.get("moderate_stage_ft", ""),
                "major_stage_ft": row.get("major_stage_ft", ""),
                "has_raw": raw.exists(),
                "raw_rows": raw_rows,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "latest_stage_ft": latest_stage,
                "max_stage_ft": max_stage,
                **model_info,
                **profile_info,
            }
        )
    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report river model/data status.")
    parser.add_argument("--sites", default=str(base.ROOT / "config" / "sites_with_usgs.csv"))
    parser.add_argument("--output", default=str(base.REPORT_DIR / "repo_status.csv"))
    parser.add_argument("--profile-summary", default=str(base.REPORT_DIR / "model_profile_summary.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command_status(args)


if __name__ == "__main__":
    main()
