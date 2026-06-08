#!/usr/bin/env python3
"""Generate latest crest forecasts from trained profile models."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import crest_eventset_train as event_train
import nwps_multigage_model as base
import train_model_profiles

FORECAST_DIR = base.OUT / "forecasts"
DOCS_DIR = base.ROOT / "docs"
OVERRIDE_COLUMNS = [
    "lid",
    "min_forecast_stage_ft",
    "early_window_ft",
    "min_stage_rise_ft",
    "min_r3_rise_rate",
    "max_remaining_rise_ft",
    "notes",
]


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:  # noqa: BLE001
        return None
    return x if math.isfinite(x) else None


def clean(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 3)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def clean_raw_stage(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["stage_ft"] = pd.to_numeric(df.get("stage_ft"), errors="coerce")
    df = df[(df["stage_ft"] > -1000) & (df["stage_ft"] < 200)]
    return df


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "lid" not in df.columns:
        return {}
    for col in OVERRIDE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["lid"] = df["lid"].astype(str).str.upper().str.strip()
    return {str(row["lid"]): row.to_dict() for _, row in df.iterrows()}


def override_float(overrides: dict[str, Any], key: str, default: float | None) -> float | None:
    value = finite_float(overrides.get(key, ""))
    return value if value is not None else default


def nearest_stage_at_or_before(df: pd.DataFrame, when: pd.Timestamp) -> float | None:
    older = df[df["datetime"] <= when]
    if older.empty:
        return None
    return finite_float(older.iloc[-1]["stage_ft"])


def rate(df: pd.DataFrame, current_time: pd.Timestamp, current_stage: float, hours: float) -> float:
    prior = nearest_stage_at_or_before(df, current_time - pd.Timedelta(hours=hours))
    if prior is None:
        return 0.0
    return (current_stage - prior) / hours


def find_recent_h0(df: pd.DataFrame, current_time: pd.Timestamp, lookback_hours: float) -> tuple[pd.Timestamp, float]:
    start = current_time - pd.Timedelta(hours=lookback_hours)
    window = df[df["datetime"] >= start]
    if window.empty:
        window = df.tail(1)
    idx = window["stage_ft"].idxmin()
    h0_time = pd.Timestamp(df.loc[idx, "datetime"])
    h0_stage = float(df.loc[idx, "stage_ft"])
    return h0_time, h0_stage


def load_profile_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "lid" not in df.columns:
        return {}
    df["lid"] = df["lid"].astype(str).str.upper().str.strip()
    return {str(row["lid"]): row.to_dict() for _, row in df.iterrows()}


def confidence_bucket(status: str, profile: dict[str, Any]) -> str:
    if status != "ok":
        return "none"
    event_count = finite_float(profile.get("event_count")) or 0
    holdout_events = finite_float(profile.get("holdout_events")) or 0
    mae = finite_float(profile.get("mae_ft"))
    bias = finite_float(profile.get("bias_ft"))
    r2 = finite_float(profile.get("r2"))

    if event_count < 5:
        return "none"
    if event_count < 8 or holdout_events < 2:
        return "low"
    if mae is None:
        return "low"
    if mae <= 1.0 and (bias is None or abs(bias) <= 0.75) and (r2 is None or r2 >= 0.5):
        return "high"
    if mae <= 1.75 and (bias is None or abs(bias) <= 1.25):
        return "medium"
    return "low"


def profile_label(profile: pd.Series | dict[str, Any]) -> tuple[str, float | None, str]:
    event_set = str(profile.get("recommended_event_set", "")).lower().strip()
    min_crest = event_train.as_float(profile.get("recommended_min_crest_stage_ft", ""))
    settings = event_train.EventSetSettings(event_set=event_set, min_crest_stage=min_crest)
    return event_set, min_crest, settings.label


def active_event_flag(
    site: pd.Series,
    df: pd.DataFrame,
    current_stage: float,
    h0_stage: float,
    r3: float,
    overrides: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, float, str]:
    start_threshold = event_train.event_start_threshold(df["stage_ft"], site)
    min_forecast_stage = override_float(overrides, "min_forecast_stage_ft", start_threshold)
    early_window = override_float(overrides, "early_window_ft", args.early_window_ft) or 0.0
    min_stage_rise = override_float(overrides, "min_stage_rise_ft", args.min_stage_rise_ft) or 0.0
    min_r3 = override_float(overrides, "min_r3_rise_rate", args.min_r3_rise_rate) or 0.0
    stage_rise = current_stage - h0_stage

    if min_forecast_stage is None:
        min_forecast_stage = start_threshold

    if current_stage >= min_forecast_stage:
        return True, float(min_forecast_stage), "current stage at/above forecast threshold"

    near_enough = current_stage >= min_forecast_stage - early_window
    rising_enough = stage_rise >= min_stage_rise and r3 >= min_r3
    if near_enough and rising_enough:
        return True, float(min_forecast_stage), "early rise: near threshold with enough stage/rate rise"

    note = f"inactive: below forecast threshold {min_forecast_stage:.2f} ft"
    if not near_enough:
        note += f"; outside early window {early_window:.2f} ft"
    elif not rising_enough:
        note += f"; rise/rate below trigger ({stage_rise:.2f} ft, {r3:.2f} ft/hr)"
    return False, float(min_forecast_stage), note


def base_row(site: pd.Series, profile: pd.Series) -> dict[str, Any]:
    lid = str(site["lid"]).upper()
    event_set, _min_crest, label = profile_label(profile)
    reason = str(profile.get("reason", ""))
    return {
        "lid": lid,
        "name": site.get("name", ""),
        "usgs_site": site.get("usgs_site", ""),
        "profile_event_set": event_set,
        "run_label": label if event_set != "skip" else "skip",
        "profile_reason": reason,
        "forecast_status": "",
        "forecast_note": "",
    }


def forecast_one(
    site: pd.Series,
    profile: pd.Series,
    profile_skill: dict[str, Any],
    overrides: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lid = str(site["lid"]).upper()
    event_set, _min_crest, label = profile_label(profile)
    reason = str(profile.get("reason", ""))
    row = base_row(site, profile)

    if event_set == "skip":
        row.update({"forecast_status": "skipped", "forecast_note": reason, "confidence": "none"})
        return row

    raw = base.DATA_RAW / f"{lid}_usgs_stage.csv"
    if not raw.exists():
        row.update({"forecast_status": "missing_raw", "forecast_note": f"missing {raw}", "confidence": "none"})
        return row

    model_file = base.MODEL_DIR / f"{lid}_{label}_ridge_model.joblib"
    if not model_file.exists():
        row.update({"forecast_status": "missing_model", "forecast_note": f"missing {model_file}", "confidence": "none"})
        return row

    df = base.prep_stage(clean_raw_stage(pd.read_csv(raw)))
    if df.empty:
        row.update({"forecast_status": "empty_raw", "forecast_note": "no usable stage rows", "confidence": "none"})
        return row

    current_time = pd.Timestamp(df.iloc[-1]["datetime"])
    current_stage = float(df.iloc[-1]["stage_ft"])
    h0_time, h0_stage = find_recent_h0(df, current_time, args.h0_lookback_hours)
    elapsed = max(0.0, (current_time - h0_time).total_seconds() / 3600)
    r1 = rate(df, current_time, current_stage, 1)
    r3 = rate(df, current_time, current_stage, 3)
    r6 = rate(df, current_time, current_stage, 6)
    r12 = rate(df, current_time, current_stage, 12)
    active, threshold_used, active_note = active_event_flag(site, df, current_stage, h0_stage, r3, overrides, args)

    shared = {
        "forecast_time_utc": current_time.isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "current_stage_ft": current_stage,
        "h0_stage_ft": h0_stage,
        "elapsed_hr_since_rise_start": elapsed,
        "r1_ft_per_hr": r1,
        "r3_ft_per_hr": r3,
        "r6_ft_per_hr": r6,
        "r12_ft_per_hr": r12,
        "forecast_threshold_used_ft": threshold_used,
        "model_mae_ft": profile_skill.get("mae_ft", ""),
        "model_bias_ft": profile_skill.get("bias_ft", ""),
        "model_event_count": profile_skill.get("event_count", ""),
        "model_holdout_events": profile_skill.get("holdout_events", ""),
        "model_r2": profile_skill.get("r2", ""),
        "model_file": str(model_file),
        "override_notes": overrides.get("notes", ""),
    }
    row.update(shared)

    if not active:
        row.update({"forecast_status": "inactive", "forecast_note": active_note, "confidence": "none"})
        return row

    features = pd.DataFrame([
        {
            "stage_ft": current_stage,
            "h0_stage_ft": h0_stage,
            "elapsed_hr_since_rise_start": elapsed,
            "r1_ft_per_hr": r1,
            "r3_ft_per_hr": r3,
            "r6_ft_per_hr": r6,
            "r12_ft_per_hr": r12,
            "momentum_r1_minus_r3": r1 - r3,
            "momentum_r3_minus_r6": r3 - r6,
            "stage_above_h0_ft": current_stage - h0_stage,
        }
    ])

    bundle = joblib.load(model_file)
    model = bundle["model"]
    meta = bundle.get("metadata", {})
    remaining = max(0.0, float(model.predict(features[base.FEATURES])[0]))
    cap = override_float(overrides, "max_remaining_rise_ft", None)
    if cap is not None and remaining > cap:
        active_note += f"; remaining rise capped at {cap:.2f} ft by override"
        remaining = cap
    likely = current_stage + remaining

    mae = finite_float(profile_skill.get("mae_ft"))
    bias = finite_float(profile_skill.get("bias_ft")) or 0.0
    under_p75 = finite_float(meta.get("skill", {}).get("underforecast_p75_ft")) or (mae or 0.0)
    under_p90 = finite_float(meta.get("skill", {}).get("underforecast_p90_ft")) or ((mae or 0.0) * 1.5)

    bias_corrected = likely - bias
    conservative = max(likely, bias_corrected + max(0.0, under_p75))
    high_end = max(conservative, bias_corrected + max(0.0, under_p90))
    confidence = confidence_bucket("ok", profile_skill)

    row.update(
        {
            "forecast_status": "ok",
            "forecast_note": active_note,
            "pred_remaining_rise_ft": remaining,
            "pred_crest_likely_ft": likely,
            "pred_crest_conservative_ft": conservative,
            "pred_crest_high_end_ft": high_end,
            "confidence": confidence,
        }
    )
    return row


def manual_model_payload(
    sites: pd.DataFrame,
    profiles: pd.DataFrame,
    skill_by_lid: dict[str, dict[str, Any]],
    overrides_by_lid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    site_by_lid = {str(row["lid"]).upper(): row for _, row in sites.iterrows()}
    models: dict[str, Any] = {}
    for _, profile in profiles.iterrows():
        lid = str(profile["lid"]).upper()
        event_set, _min_crest, label = profile_label(profile)
        skill = skill_by_lid.get(lid, {})
        overrides = overrides_by_lid.get(lid, {})
        site = site_by_lid.get(lid, {})
        model_file = base.MODEL_DIR / f"{lid}_{label}_ridge_model.joblib"
        item: dict[str, Any] = {
            "lid": lid,
            "name": site.get("name", "") if hasattr(site, "get") else "",
            "run_label": label if event_set != "skip" else "skip",
            "event_set": event_set,
            "status": skill.get("status", "skipped" if event_set == "skip" else "missing_model"),
            "profile_reason": profile.get("reason", ""),
            "skill": {k: clean(v) for k, v in skill.items()},
            "overrides": {k: clean(v) for k, v in overrides.items()},
        }
        if event_set != "skip" and model_file.exists():
            bundle = joblib.load(model_file)
            meta = bundle.get("metadata", {})
            item["status"] = skill.get("status", "ok")
            item["equation"] = clean(meta.get("equation", {}))
            item["model_skill"] = clean(meta.get("skill", {}))
        models[lid] = item
    return models


def build_outputs(
    rows: list[dict[str, Any]],
    manual_models: dict[str, Any],
    output_csv: Path,
    output_json: Path,
    docs_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    docs_json.parent.mkdir(parents=True, exist_ok=True)

    clean_rows = [{k: clean(v) for k, v in row.items()} for row in rows]
    pd.DataFrame(clean_rows).to_csv(output_csv, index=False)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(clean_rows),
        "forecasts": clean_rows,
        "manual_models": clean(manual_models),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    docs_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    print(f"Wrote {docs_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latest crest forecasts from profile models.")
    parser.add_argument("--sites", default=str(base.ROOT / "config" / "sites_with_usgs.csv"))
    parser.add_argument("--profiles", default=str(base.ROOT / "config" / "model_profiles.csv"))
    parser.add_argument("--overrides", default=str(base.ROOT / "config" / "forecast_overrides.csv"))
    parser.add_argument("--profile-summary", default=str(base.REPORT_DIR / "model_profile_summary.csv"))
    parser.add_argument("--lids", default="", help="Optional comma-separated LIDs")
    parser.add_argument("--h0-lookback-hours", type=float, default=48.0)
    parser.add_argument("--early-window-ft", type=float, default=2.0)
    parser.add_argument("--min-stage-rise-ft", type=float, default=0.75)
    parser.add_argument("--min-r3-rise-rate", type=float, default=0.05)
    parser.add_argument("--output-csv", default=str(FORECAST_DIR / "latest_forecasts.csv"))
    parser.add_argument("--output-json", default=str(FORECAST_DIR / "latest_forecasts.json"))
    parser.add_argument("--docs-json", default=str(DOCS_DIR / "latest_forecasts.json"))
    args = parser.parse_args()

    base.ensure_dirs()
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    sites = event_train.read_sites(args.sites)
    profiles = train_model_profiles.read_profiles(args.profiles)
    selected = train_model_profiles.parse_lids(args.lids)
    if selected:
        sites = sites[sites["lid"].isin(selected)].copy()
        profiles = profiles[profiles["lid"].isin(selected)].copy()

    profile_by_lid = {str(row["lid"]).upper(): row for _, row in profiles.iterrows()}
    skill_by_lid = load_profile_summary(Path(args.profile_summary))
    overrides_by_lid = load_overrides(Path(args.overrides))
    rows: list[dict[str, Any]] = []

    for _, site in sites.iterrows():
        lid = str(site["lid"]).upper()
        profile = profile_by_lid.get(lid)
        if profile is None:
            rows.append({"lid": lid, "name": site.get("name", ""), "forecast_status": "missing_profile", "confidence": "none"})
            continue
        row = forecast_one(site, profile, skill_by_lid.get(lid, {}), overrides_by_lid.get(lid, {}), args)
        rows.append(row)
        print(f"{lid}: {row.get('forecast_status')} {row.get('run_label')} {row.get('pred_crest_likely_ft', '')}")

    models = manual_model_payload(sites, profiles, skill_by_lid, overrides_by_lid)
    build_outputs(rows, models, Path(args.output_csv), Path(args.output_json), Path(args.docs_json))


if __name__ == "__main__":
    main()
