#!/usr/bin/env python3
"""
LIX multi-gage crest model builder.

Builds station-specific remaining-rise / final-crest models from historical river
stage time series. The intent is an auditable statistical aid: current stage +
rate-of-rise behavior + basin memory -> estimated remaining rise.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import model_core as core
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"
MODEL_DIR = OUT / "models"
REPORT_DIR = OUT / "reports"

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
NWPS_GAUGE_URLS = [
    "https://api.water.noaa.gov/nwps/v1/gauges/{lid}",
    "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow",
]

FEATURES = core.FEATURES


SITE_COLUMNS = [
    "lid",
    "name",
    "usgs_site",
    "event_threshold_ft",
    "action_stage_ft",
    "flood_stage_ft",
    "notes",
]


@dataclass
class EventSettings:
    min_total_rise_ft: float = 1.0
    below_threshold_hours_to_end: float = 24.0
    pre_crest_lookback_hours: float = 48.0
    include_pre_event_hours: float = 24.0
    sample_interval: str = "1h"


def ensure_dirs() -> None:
    for p in [DATA_RAW, DATA_PROCESSED, MODEL_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


coerce_float = core.coerce_float


def read_sites(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"lid", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sites file missing required columns: {sorted(missing)}")
    for col in SITE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[SITE_COLUMNS + [c for c in df.columns if c not in SITE_COLUMNS]]
    df["lid"] = df["lid"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["usgs_site"] = df["usgs_site"].astype(str).str.strip()
    return df


def write_sites(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 45,
    tries: int = 3,
    sleep_seconds: float = 0.5,
) -> Any | None:
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "lix-river-crest-model/0.2"},
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 - command-line tool should keep moving
            last_err = exc
            if attempt < tries:
                time.sleep(sleep_seconds * attempt)
    print(f"WARN: failed {url}: {last_err}", file=sys.stderr)
    return None


def recursive_find_usgs(obj: Any) -> list[str]:
    """Find plausible 7-15 digit USGS IDs anywhere in NWPS metadata."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if "usgs" in key and isinstance(v, (str, int, float)):
                found += re.findall(r"\b\d{7,15}\b", str(v))
            found += recursive_find_usgs(v)
    elif isinstance(obj, list):
        for item in obj:
            found += recursive_find_usgs(item)
    elif isinstance(obj, str) and "usgs" in obj.lower():
        found += re.findall(r"\b\d{7,15}\b", obj)
    return sorted(set(found))


def recursive_find_latlon(obj: Any) -> tuple[float | None, float | None]:
    """Best-effort extraction of latitude/longitude from NWPS-ish JSON."""
    lat_keys = {"lat", "latitude", "y"}
    lon_keys = {"lon", "lng", "longitude", "x"}
    if isinstance(obj, dict):
        lower = {str(k).lower(): v for k, v in obj.items()}
        lat = next((coerce_float(lower[k]) for k in lat_keys if k in lower), None)
        lon = next((coerce_float(lower[k]) for k in lon_keys if k in lower), None)
        if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
        for v in obj.values():
            lat, lon = recursive_find_latlon(v)
            if lat is not None and lon is not None:
                return lat, lon
    elif isinstance(obj, list):
        for item in obj:
            lat, lon = recursive_find_latlon(item)
            if lat is not None and lon is not None:
                return lat, lon
    return None, None


def command_discover_sites(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    rows: list[dict[str, Any]] = []

    for _, row in sites.iterrows():
        out_row = row.to_dict()
        lid = out_row["lid"]
        existing_usgs = str(out_row.get("usgs_site", "")).strip()
        candidates: list[str] = []
        meta = None

        if not existing_usgs:
            for tmpl in NWPS_GAUGE_URLS:
                for candidate_lid in [lid.lower(), lid.upper()]:
                    meta = fetch_json(tmpl.format(lid=candidate_lid), timeout=60)
                    if meta:
                        candidates = recursive_find_usgs(meta)
                        if candidates:
                            break
                if candidates:
                    break

        if existing_usgs:
            status = "already_set"
            selected = existing_usgs
        elif candidates:
            status = "auto_from_nwps"
            selected = candidates[0]
        else:
            status = "not_found"
            selected = ""

        out_row["usgs_site"] = selected
        out_row["discovery_status"] = status
        out_row["candidate_usgs_sites"] = ";".join(candidates)

        if meta:
            lat, lon = recursive_find_latlon(meta)
            out_row["nwps_latitude"] = lat if lat is not None else ""
            out_row["nwps_longitude"] = lon if lon is not None else ""

        rows.append(out_row)
        print(f"{lid:6s} {selected or 'NO_USGS_ID_FOUND':>15s}  {status}")

    out = Path(args.output)
    write_sites(pd.DataFrame(rows), out)
    print(f"\nWrote {out}")
    print("Review blanks in usgs_site. A bad LID-to-USGS mapping will poison the model, because rivers enjoy being assholes.")


def parse_usgs_iv_json(data: Any) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    try:
        ts_list = data["value"]["timeSeries"]
    except Exception:
        return pd.DataFrame(columns=["datetime", "stage_ft"])

    for ts in ts_list:
        for group in ts.get("values", []):
            for v in group.get("value", []):
                stage = coerce_float(v.get("value"))
                if stage is None:
                    continue
                recs.append(
                    {
                        "datetime": v.get("dateTime"),
                        "stage_ft": stage,
                        "qualifiers": ";".join(v.get("qualifiers", [])),
                    }
                )
    if not recs:
        return pd.DataFrame(columns=["datetime", "stage_ft", "qualifiers"])
    df = pd.DataFrame(recs)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = clean_stage_data(df)
    return (
        df.dropna(subset=["datetime", "stage_ft"])
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def download_usgs_stage(
    site: str,
    start: datetime,
    end: datetime,
    chunk_months: int = 12,
    sleep_seconds: float = 0.1,
) -> pd.DataFrame:
    """Download USGS instantaneous gage height (00065), chunked to avoid huge requests."""
    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + relativedelta(months=chunk_months), end)
        params = {
            "format": "json",
            "sites": site,
            "parameterCd": "00065",
            "startDT": cursor.strftime("%Y-%m-%d"),
            "endDT": chunk_end.strftime("%Y-%m-%d"),
            "siteStatus": "all",
        }
        data = fetch_json(USGS_IV_URL, params=params, timeout=120, tries=3)
        if data:
            chunk = parse_usgs_iv_json(data)
            if not chunk.empty:
                chunks.append(chunk)
        cursor = chunk_end + timedelta(days=1)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not chunks:
        return pd.DataFrame(columns=["datetime", "stage_ft", "qualifiers"])

    df = clean_stage_data(pd.concat(chunks, ignore_index=True))
    return (
        df.dropna(subset=["datetime", "stage_ft"])
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def command_download(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    end = pd.Timestamp(args.end, tz="UTC").to_pydatetime() if args.end else datetime.now(timezone.utc)
    start = pd.Timestamp(args.start, tz="UTC").to_pydatetime() if args.start else end - relativedelta(years=int(args.years))

    if args.limit:
        sites = sites.head(args.limit)

    for _, row in sites.iterrows():
        lid = row["lid"]
        site = str(row.get("usgs_site", "")).strip()
        if not site:
            print(f"SKIP {lid}: no usgs_site")
            continue

        out = DATA_RAW / f"{lid}_usgs_stage.csv"
        if out.exists() and args.skip_existing and not args.force:
            print(f"SKIP {lid}: {out} already exists")
            continue

        print(f"Downloading {lid} / USGS {site} from {start.date()} to {end.date()}")
        df = download_usgs_stage(
            site,
            start=start,
            end=end,
            chunk_months=args.chunk_months,
            sleep_seconds=args.sleep_seconds,
        )
        df.to_csv(out, index=False)
        print(f"  {len(df):,} rows -> {out}")


clean_stage_data = core.clean_stage_data
prep_stage = core.prep_stage
choose_threshold = core.choose_threshold
choose_event_threshold = core.choose_event_threshold
detect_events = core.detect_events
nearest_stage_at = core.nearest_stage_at
feature_rows = core.feature_rows
original_scale_equation = core.original_scale_equation
safe_r2 = core.safe_r2
residual_stats = core.residual_stats


def train_one_lid(lid: str, stage_df: pd.DataFrame, site_row: pd.Series | dict[str, Any], settings: EventSettings) -> dict[str, Any]:
    stage_df = clean_stage_data(stage_df)
    events = detect_events(stage_df, site_row=site_row, settings=settings)
    event_out = DATA_PROCESSED / f"{lid}_events.csv"
    train_out = DATA_PROCESSED / f"{lid}_training_rows.csv"
    scored_out = DATA_PROCESSED / f"{lid}_training_rows_scored.csv"

    if events.empty or len(events) < 5:
        events.to_csv(event_out, index=False)
        return {"lid": lid, "status": "too_few_events", "event_count": int(len(events))}

    events.to_csv(event_out, index=False)
    train = feature_rows(stage_df, events, lid, sample_interval=settings.sample_interval)
    train.to_csv(train_out, index=False)

    if len(train) < 40 or train["event_id"].nunique() < 5:
        return {
            "lid": lid,
            "status": "too_few_training_rows",
            "event_count": int(len(events)),
            "training_rows": int(len(train)),
        }

    X = train[FEATURES]
    y = train["remaining_rise_ft"].clip(lower=0)
    groups = train["event_id"]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=[0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])),
        ]
    )

    if train["event_id"].nunique() >= 7:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        test_pred_remaining = np.maximum(0, model.predict(X.iloc[test_idx]))
        test_obs_remaining = y.iloc[test_idx].to_numpy()
        holdout_rows = len(test_idx)
        holdout_events = int(train.iloc[test_idx]["event_id"].nunique())
        mae = float(mean_absolute_error(test_obs_remaining, test_pred_remaining))
        rmse = float(math.sqrt(mean_squared_error(test_obs_remaining, test_pred_remaining)))
        bias = float(np.mean(test_pred_remaining - test_obs_remaining))
        r2 = safe_r2(test_obs_remaining, test_pred_remaining)
    else:
        model.fit(X, y)
        pred_remaining = np.maximum(0, model.predict(X))
        obs_remaining = y.to_numpy()
        holdout_rows = 0
        holdout_events = 0
        mae = float(mean_absolute_error(obs_remaining, pred_remaining))
        rmse = float(math.sqrt(mean_squared_error(obs_remaining, pred_remaining)))
        bias = float(np.mean(pred_remaining - obs_remaining))
        r2 = safe_r2(obs_remaining, pred_remaining)

    model.fit(X, y)
    full_pred_remaining = np.maximum(0, model.predict(X))
    train["pred_remaining_rise_ft"] = full_pred_remaining
    train["pred_crest_stage_ft"] = train["stage_ft"] + train["pred_remaining_rise_ft"]
    train["error_ft"] = train["pred_crest_stage_ft"] - train["observed_crest_stage_ft"]
    train.to_csv(scored_out, index=False)

    eq = original_scale_equation(model)
    resid = residual_stats(train["error_ft"])
    metadata = {
        "lid": lid,
        "name": site_row.get("name", "") if isinstance(site_row, dict) else site_row.get("name", ""),
        "features": FEATURES,
        "event_settings": settings.__dict__,
        "training_rows": int(len(train)),
        "event_count": int(events["event_id"].nunique()),
        "skill": {
            "mae_ft": mae,
            "rmse_ft": rmse,
            "bias_ft": bias,
            "r2": r2,
            "holdout_rows": holdout_rows,
            "holdout_events": holdout_events,
            **resid,
        },
        "equation": eq,
    }

    joblib.dump({"lid": lid, "features": FEATURES, "model": model, "metadata": metadata}, MODEL_DIR / f"{lid}_ridge_model.joblib")
    with open(REPORT_DIR / f"{lid}_equation.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "lid": lid,
        "status": "ok",
        "event_count": int(events["event_id"].nunique()),
        "training_rows": int(len(train)),
        "holdout_rows": holdout_rows,
        "holdout_events": holdout_events,
        "mae_ft": mae,
        "rmse_ft": rmse,
        "bias_ft": bias,
        "r2": r2,
        **resid,
        "equation_file": str(REPORT_DIR / f"{lid}_equation.json"),
        "model_file": str(MODEL_DIR / f"{lid}_ridge_model.joblib"),
    }


def build_event_settings(args: argparse.Namespace) -> EventSettings:
    return EventSettings(
        min_total_rise_ft=args.min_total_rise,
        below_threshold_hours_to_end=args.below_hours,
        pre_crest_lookback_hours=args.h0_lookback_hours,
        include_pre_event_hours=args.pre_event_hours,
        sample_interval=args.sample_interval,
    )


def command_train(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    if args.limit:
        sites = sites.head(args.limit)
    settings = build_event_settings(args)
    summaries: list[dict[str, Any]] = []

    for _, row in sites.iterrows():
        lid = row["lid"]
        raw = DATA_RAW / f"{lid}_usgs_stage.csv"
        if not raw.exists():
            print(f"SKIP {lid}: missing {raw}")
            continue
        print(f"Training {lid}...")
        stage_df = pd.read_csv(raw)
        result = train_one_lid(lid, stage_df, row, settings)
        summaries.append(result)
        print(f"  {result}")

    summary = pd.DataFrame(summaries)
    summary_out = REPORT_DIR / "model_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"\nWrote {summary_out}")


def forecast_from_features(lid: str, row: dict[str, float]) -> dict[str, Any]:
    model_path = MODEL_DIR / f"{lid}_ridge_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for {lid}: {model_path}")
    bundle = joblib.load(model_path)
    X = pd.DataFrame([row])[FEATURES]
    remaining = max(0.0, float(bundle["model"].predict(X)[0]))
    likely_crest = row["stage_ft"] + remaining
    metadata = bundle.get("metadata", {})
    skill = metadata.get("skill", {})
    mae = float(skill.get("mae_ft", 0.0) or 0.0)
    under75 = float(skill.get("underforecast_p75_ft", mae) or mae)
    under90 = float(skill.get("underforecast_p90_ft", max(mae, under75)) or max(mae, under75))

    return {
        "lid": lid,
        "current_stage_ft": round(row["stage_ft"], 2),
        "pred_remaining_rise_ft": round(remaining, 2),
        "pred_crest_stage_ft": round(likely_crest, 2),
        "low_reasonable_crest_ft": round(max(row["stage_ft"], likely_crest - mae), 2),
        "conservative_crest_ft": round(likely_crest + max(mae, under75), 2),
        "high_conservative_crest_ft": round(likely_crest + max(mae, under90), 2),
        "skill_used": {k: skill.get(k) for k in ["mae_ft", "rmse_ft", "bias_ft", "r2", "underforecast_p75_ft", "underforecast_p90_ft"]},
        "inputs": row,
    }


def command_forecast(args: argparse.Namespace) -> None:
    lid = args.lid.upper()
    row = {
        "stage_ft": args.stage,
        "h0_stage_ft": args.h0,
        "elapsed_hr_since_rise_start": args.elapsed,
        "r1_ft_per_hr": args.r1,
        "r3_ft_per_hr": args.r3,
        "r6_ft_per_hr": args.r6,
        "r12_ft_per_hr": args.r12,
        "momentum_r1_minus_r3": args.r1 - args.r3,
        "momentum_r3_minus_r6": args.r3 - args.r6,
        "stage_above_h0_ft": args.stage - args.h0,
    }
    print(json.dumps(forecast_from_features(lid, row), indent=2))


def infer_latest_features(stage_df: pd.DataFrame, h0_lookback_hours: float = 72.0) -> dict[str, float]:
    df = prep_stage(stage_df).set_index("datetime")
    if df.empty:
        raise ValueError("No usable stage data")
    now = df.index.max()
    stage = nearest_stage_at(df, now)
    h0_window = df[df.index >= now - pd.Timedelta(hours=h0_lookback_hours)]
    if h0_window.empty:
        h0_window = df
    h0_idx = h0_window["stage_ft"].idxmin()
    h0 = float(h0_window.loc[h0_idx, "stage_ft"])
    elapsed = max(0.0, (now - h0_idx).total_seconds() / 3600)

    def rate(hours: int) -> float:
        past = nearest_stage_at(df, now - pd.Timedelta(hours=hours), tolerance=pd.Timedelta(minutes=45))
        return np.nan if pd.isna(past) else (stage - past) / hours

    r1, r3, r6, r12 = rate(1), rate(3), rate(6), rate(12)
    if any(pd.isna(x) for x in [r1, r3, r6, r12]):
        raise ValueError("Could not compute all rates from latest data; need at least 12 hours of recent data")

    return {
        "stage_ft": float(stage),
        "h0_stage_ft": h0,
        "elapsed_hr_since_rise_start": float(elapsed),
        "r1_ft_per_hr": float(r1),
        "r3_ft_per_hr": float(r3),
        "r6_ft_per_hr": float(r6),
        "r12_ft_per_hr": float(r12),
        "momentum_r1_minus_r3": float(r1 - r3),
        "momentum_r3_minus_r6": float(r3 - r6),
        "stage_above_h0_ft": float(stage - h0),
    }


def command_forecast_latest(args: argparse.Namespace) -> None:
    lid = args.lid.upper()
    raw = DATA_RAW / f"{lid}_usgs_stage.csv"
    if not raw.exists():
        raise FileNotFoundError(f"Missing {raw}. Run download first.")
    stage_df = pd.read_csv(raw)
    features = infer_latest_features(stage_df, h0_lookback_hours=args.h0_lookback_hours)
    print(json.dumps(forecast_from_features(lid, features), indent=2))


def command_status(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    rows: list[dict[str, Any]] = []
    for _, row in sites.iterrows():
        lid = row["lid"]
        raw = DATA_RAW / f"{lid}_usgs_stage.csv"
        model = MODEL_DIR / f"{lid}_ridge_model.joblib"
        raw_rows: int | str = ""
        start = ""
        end = ""
        if raw.exists():
            try:
                df = pd.read_csv(raw, usecols=["datetime", "stage_ft"])
                raw_rows = len(df)
                dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dropna()
                if not dt.empty:
                    start = dt.min().date().isoformat()
                    end = dt.max().date().isoformat()
            except Exception as exc:  # noqa: BLE001
                raw_rows = f"error: {exc}"
        rows.append(
            {
                "lid": lid,
                "name": row["name"],
                "usgs_site": row.get("usgs_site", ""),
                "has_raw": raw.exists(),
                "raw_rows": raw_rows,
                "raw_start": start,
                "raw_end": end,
                "has_model": model.exists(),
            }
        )
    out = pd.DataFrame(rows)
    status_out = REPORT_DIR / "repo_status.csv"
    out.to_csv(status_out, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {status_out}")


def command_run_all(args: argparse.Namespace) -> None:
    discover_args = argparse.Namespace(sites=args.sites, output=args.discovered_sites)
    command_discover_sites(discover_args)

    download_args = argparse.Namespace(
        sites=args.discovered_sites,
        years=args.years,
        start=args.start,
        end=args.end,
        limit=args.limit,
        chunk_months=args.chunk_months,
        sleep_seconds=args.sleep_seconds,
        skip_existing=args.skip_existing,
        force=args.force,
    )
    command_download(download_args)

    train_args = argparse.Namespace(
        sites=args.discovered_sites,
        limit=args.limit,
        min_total_rise=args.min_total_rise,
        below_hours=args.below_hours,
        h0_lookback_hours=args.h0_lookback_hours,
        pre_event_hours=args.pre_event_hours,
        sample_interval=args.sample_interval,
    )
    command_train(train_args)


def add_event_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-total-rise", type=float, default=1.0, help="Minimum rise from H0 to crest to count as an event")
    parser.add_argument("--below-hours", type=float, default=24.0, help="Hours below threshold required to end an event")
    parser.add_argument("--h0-lookback-hours", type=float, default=48.0, help="Lookback from crest to search for rise-start/H0")
    parser.add_argument("--pre-event-hours", type=float, default=24.0, help="Hours before threshold crossing to include in event")
    parser.add_argument("--sample-interval", default="1h", help="Training sample interval inside each event")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi-gage river crest models from historical stage data.")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("discover-sites", help="Try to map NWS LIDs to USGS site IDs using NWPS metadata")
    ps.add_argument("--sites", default=str(ROOT / "config" / "sites.csv"))
    ps.add_argument("--output", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    ps.set_defaults(func=command_discover_sites)

    pdn = sub.add_parser("download", help="Download historical USGS gage-height data")
    pdn.add_argument("--sites", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    pdn.add_argument("--years", type=int, default=15)
    pdn.add_argument("--start", default="", help="Optional YYYY-MM-DD UTC start date")
    pdn.add_argument("--end", default="", help="Optional YYYY-MM-DD UTC end date")
    pdn.add_argument("--limit", type=int, default=0, help="Only process first N sites; useful for smoke tests")
    pdn.add_argument("--chunk-months", type=int, default=12)
    pdn.add_argument("--sleep-seconds", type=float, default=0.1)
    pdn.add_argument("--skip-existing", action="store_true", help="Do not re-download files that already exist")
    pdn.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    pdn.set_defaults(func=command_download)

    pt = sub.add_parser("train", help="Detect events and train station-specific models")
    pt.add_argument("--sites", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    pt.add_argument("--limit", type=int, default=0, help="Only process first N sites")
    add_event_args(pt)
    pt.set_defaults(func=command_train)

    pf = sub.add_parser("forecast", help="Forecast crest from manually supplied current hydrologic features")
    pf.add_argument("--lid", required=True)
    pf.add_argument("--stage", type=float, required=True)
    pf.add_argument("--h0", type=float, required=True)
    pf.add_argument("--elapsed", type=float, required=True)
    pf.add_argument("--r1", type=float, required=True)
    pf.add_argument("--r3", type=float, required=True)
    pf.add_argument("--r6", type=float, required=True)
    pf.add_argument("--r12", type=float, required=True)
    pf.set_defaults(func=command_forecast)

    pfl = sub.add_parser("forecast-latest", help="Infer latest features from downloaded data and forecast")
    pfl.add_argument("--lid", required=True)
    pfl.add_argument("--h0-lookback-hours", type=float, default=72.0)
    pfl.set_defaults(func=command_forecast_latest)

    pst = sub.add_parser("status", help="Show mapping/data/model status by LID")
    pst.add_argument("--sites", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    pst.set_defaults(func=command_status)

    pra = sub.add_parser("run-all", help="Discover sites, download data, and train models")
    pra.add_argument("--sites", default=str(ROOT / "config" / "sites.csv"))
    pra.add_argument("--discovered-sites", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    pra.add_argument("--years", type=int, default=15)
    pra.add_argument("--start", default="")
    pra.add_argument("--end", default="")
    pra.add_argument("--limit", type=int, default=0)
    pra.add_argument("--chunk-months", type=int, default=12)
    pra.add_argument("--sleep-seconds", type=float, default=0.1)
    pra.add_argument("--skip-existing", action="store_true")
    pra.add_argument("--force", action="store_true")
    add_event_args(pra)
    pra.set_defaults(func=command_run_all)

    return p


def main() -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
