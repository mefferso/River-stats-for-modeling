#!/usr/bin/env python3
"""
LIX multi-gage crest model builder.

What it does:
1. Reads NWS LIDs from config/sites.csv
2. Tries to discover USGS site IDs from NWPS metadata
3. Downloads historical USGS gage height/stage data
4. Detects flood/rise events
5. Builds station-specific remaining-rise models
6. Outputs model equations, skill scores, and forecast CLI

This is intentionally conservative and auditable. It favors a plain Ridge
regression formula over a black-box model, because operationally you need to
know when the model is full of crap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
NWPS_GAUGE_URLS = [
    "https://api.water.noaa.gov/nwps/v1/gauges/{lid}",
    "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow",
]

FEATURES = [
    "stage_ft",
    "h0_stage_ft",
    "elapsed_hr_since_rise_start",
    "r1_ft_per_hr",
    "r3_ft_per_hr",
    "r6_ft_per_hr",
    "r12_ft_per_hr",
    "momentum_r1_minus_r3",
    "momentum_r3_minus_r6",
    "stage_above_h0_ft",
]


def ensure_dirs() -> None:
    for p in [DATA_RAW, DATA_PROCESSED, MODEL_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_sites(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"lid", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sites file missing required columns: {missing}")
    if "usgs_site" not in df.columns:
        df["usgs_site"] = ""
    df["lid"] = df["lid"].str.upper().str.strip()
    df["usgs_site"] = df["usgs_site"].astype(str).str.strip()
    return df


def recursive_find_usgs(obj: Any) -> list[str]:
    """Find plausible 8-15 digit USGS IDs anywhere in NWPS metadata."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if "usgs" in key and isinstance(v, (str, int, float)):
                s = str(v)
                found += re.findall(r"\b\d{7,15}\b", s)
            found += recursive_find_usgs(v)
    elif isinstance(obj, list):
        for item in obj:
            found += recursive_find_usgs(item)
    elif isinstance(obj, str):
        if "usgs" in obj.lower():
            found += re.findall(r"\b\d{7,15}\b", obj)
    return sorted(set(found))


def fetch_json(url: str, params: dict[str, Any] | None = None, timeout: int = 45) -> Any | None:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "lix-river-crest-model/0.1"})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"WARN: failed {url}: {e}", file=sys.stderr)
        return None


def discover_sites(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    rows = []
    for _, row in sites.iterrows():
        lid = row["lid"]
        usgs = row.get("usgs_site", "").strip()
        meta = None
        if not usgs:
            for tmpl in NWPS_GAUGE_URLS:
                meta = fetch_json(tmpl.format(lid=lid.lower())) or fetch_json(tmpl.format(lid=lid.upper()))
                if meta:
                    ids = recursive_find_usgs(meta)
                    if ids:
                        usgs = ids[0]
                        break
        rows.append({"lid": lid, "name": row["name"], "usgs_site": usgs})
        print(f"{lid:6s} {usgs or 'NO_USGS_ID_FOUND'}")
    out = Path(args.output or ROOT / "config" / "sites_with_usgs.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


def download_usgs_stage(site: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download USGS instantaneous gage height (00065), chunked to avoid huge requests."""
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + relativedelta(months=6), end)
        params = {
            "format": "json",
            "sites": site,
            "parameterCd": "00065",
            "startDT": cursor.strftime("%Y-%m-%d"),
            "endDT": chunk_end.strftime("%Y-%m-%d"),
            "siteStatus": "all",
        }
        data = fetch_json(USGS_IV_URL, params=params, timeout=90)
        if data:
            try:
                ts_list = data["value"]["timeSeries"]
                for ts in ts_list:
                    vals = ts["values"][0]["value"]
                    recs = []
                    for v in vals:
                        try:
                            stage = float(v["value"])
                        except Exception:
                            continue
                        if math.isfinite(stage):
                            recs.append({"datetime": v["dateTime"], "stage_ft": stage})
                    if recs:
                        chunks.append(pd.DataFrame(recs))
            except Exception as e:
                print(f"WARN: could not parse USGS response for {site}: {e}", file=sys.stderr)
        cursor = chunk_end + timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["datetime", "stage_ft"])
    df = pd.concat(chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "stage_ft"]).drop_duplicates("datetime").sort_values("datetime")
    return df


def command_download(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    end = datetime.now(timezone.utc)
    start = end - relativedelta(years=int(args.years))
    for _, row in sites.iterrows():
        lid = row["lid"]
        site = row["usgs_site"].strip()
        if not site:
            print(f"SKIP {lid}: no usgs_site")
            continue
        print(f"Downloading {lid} / USGS {site} from {start.date()} to {end.date()}")
        df = download_usgs_stage(site, start, end)
        out = DATA_RAW / f"{lid}_usgs_stage.csv"
        df.to_csv(out, index=False)
        print(f"  {len(df):,} rows -> {out}")


def prep_stage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "stage_ft"]).sort_values("datetime")
    df = df.set_index("datetime")
    # Standardize to 15-min. Interpolate small holes only; do not bridge giant outages.
    s = df["stage_ft"].resample("15min").median().interpolate(limit=4)
    out = s.to_frame()
    out["stage_ft"] = out["stage_ft"].rolling(3, center=True, min_periods=1).median()
    return out.reset_index()


def auto_event_threshold(stage: pd.Series) -> float:
    """Fallback event threshold. Operationally, replace with action stage if desired."""
    q90 = float(stage.quantile(0.90))
    q75 = float(stage.quantile(0.75))
    # Avoid letting low-flow noise make events everywhere.
    return max(q90, q75 + 1.0)


def detect_events(stage_df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    df = prep_stage(stage_df)
    if df.empty:
        return pd.DataFrame()
    threshold = float(threshold) if threshold is not None else auto_event_threshold(df["stage_ft"])
    df["above"] = df["stage_ft"] >= threshold
    events = []
    in_event = False
    start_idx = None
    below_count = 0
    min_gap_steps = 24 * 4  # 24 hours below threshold ends event

    for i, above in enumerate(df["above"].to_numpy()):
        if above and not in_event:
            in_event = True
            start_idx = max(0, i - 24 * 4)  # include prior day for rise start
            below_count = 0
        elif in_event:
            if above:
                below_count = 0
            else:
                below_count += 1
                if below_count >= min_gap_steps:
                    end_idx = i
                    events.append((start_idx, end_idx))
                    in_event = False
                    start_idx = None
                    below_count = 0
    if in_event and start_idx is not None:
        events.append((start_idx, len(df) - 1))

    rows = []
    for event_num, (a, b) in enumerate(events, start=1):
        ev = df.iloc[a:b + 1].copy()
        if len(ev) < 12:
            continue
        peak_idx = ev["stage_ft"].idxmax()
        peak_time = ev.loc[peak_idx, "datetime"]
        peak_stage = float(ev.loc[peak_idx, "stage_ft"])
        # Rise start = lowest stage in the 48h before peak within event window.
        pre_peak = ev[ev["datetime"] <= peak_time].copy()
        if pre_peak.empty:
            continue
        lookback_start = peak_time - pd.Timedelta(hours=48)
        rise_window = pre_peak[pre_peak["datetime"] >= lookback_start]
        if rise_window.empty:
            rise_window = pre_peak
        h0_idx = rise_window["stage_ft"].idxmin()
        h0_time = ev.loc[h0_idx, "datetime"]
        h0_stage = float(ev.loc[h0_idx, "stage_ft"])
        if peak_stage - h0_stage < 1.0:
            continue
        rows.append({
            "event_id": f"E{event_num:04d}_{peak_time.strftime('%Y%m%d%H%M')}",
            "start_time": ev.iloc[0]["datetime"],
            "rise_start_time": h0_time,
            "crest_time": peak_time,
            "end_time": ev.iloc[-1]["datetime"],
            "h0_stage_ft": h0_stage,
            "crest_stage_ft": peak_stage,
            "total_rise_ft": peak_stage - h0_stage,
            "threshold_used_ft": threshold,
        })
    return pd.DataFrame(rows)


def feature_rows(stage_df: pd.DataFrame, events: pd.DataFrame, lid: str) -> pd.DataFrame:
    df = prep_stage(stage_df).set_index("datetime")
    rows = []
    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        crest_time = pd.to_datetime(ev["crest_time"], utc=True)
        rise_start = pd.to_datetime(ev["rise_start_time"], utc=True)
        h0 = float(ev["h0_stage_ft"])
        crest = float(ev["crest_stage_ft"])
        window = df[(df.index >= rise_start) & (df.index < crest_time)].copy()
        if window.empty:
            continue
        # hourly samples make training less dominated by 15-min autocorrelation
        sample_times = window.resample("1h").nearest().dropna().index
        for t in sample_times:
            stage = float(df.loc[t, "stage_ft"])
            def rate(hours: int) -> float:
                t0 = t - pd.Timedelta(hours=hours)
                if t0 not in df.index:
                    # nearest within 30 min
                    idx = df.index.get_indexer([t0], method="nearest", tolerance=pd.Timedelta(minutes=30))
                    if idx[0] == -1:
                        return np.nan
                    past = float(df.iloc[idx[0]]["stage_ft"])
                else:
                    past = float(df.loc[t0, "stage_ft"])
                return (stage - past) / hours
            r1, r3, r6, r12 = rate(1), rate(3), rate(6), rate(12)
            remaining = crest - stage
            if remaining < -0.05:
                continue
            rows.append({
                "lid": lid,
                "event_id": event_id,
                "datetime": t,
                "stage_ft": stage,
                "h0_stage_ft": h0,
                "elapsed_hr_since_rise_start": (t - rise_start).total_seconds() / 3600,
                "r1_ft_per_hr": r1,
                "r3_ft_per_hr": r3,
                "r6_ft_per_hr": r6,
                "r12_ft_per_hr": r12,
                "momentum_r1_minus_r3": r1 - r3 if pd.notna(r1) and pd.notna(r3) else np.nan,
                "momentum_r3_minus_r6": r3 - r6 if pd.notna(r3) and pd.notna(r6) else np.nan,
                "stage_above_h0_ft": stage - h0,
                "remaining_rise_ft": remaining,
                "observed_crest_stage_ft": crest,
            })
    out = pd.DataFrame(rows)
    return out.dropna(subset=FEATURES + ["remaining_rise_ft"])


def train_one_lid(lid: str, stage_df: pd.DataFrame) -> dict[str, Any] | None:
    events = detect_events(stage_df)
    if events.empty or len(events) < 5:
        return {"lid": lid, "status": "too_few_events", "event_count": len(events)}
    events.to_csv(DATA_PROCESSED / f"{lid}_events.csv", index=False)
    train = feature_rows(stage_df, events, lid)
    train.to_csv(DATA_PROCESSED / f"{lid}_training_rows.csv", index=False)
    if len(train) < 40 or train["event_id"].nunique() < 5:
        return {"lid": lid, "status": "too_few_training_rows", "event_count": len(events), "rows": len(train)}

    X = train[FEATURES]
    y = train["remaining_rise_ft"].clip(lower=0)
    groups = train["event_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])),
    ])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = np.maximum(0, model.predict(X.iloc[test_idx]))
    obs = y.iloc[test_idx].to_numpy()

    full_pred = np.maximum(0, model.predict(X))
    train["pred_remaining_rise_ft"] = full_pred
    train["pred_crest_stage_ft"] = train["stage_ft"] + train["pred_remaining_rise_ft"]
    train["error_ft"] = train["pred_crest_stage_ft"] - train["observed_crest_stage_ft"]
    train.to_csv(DATA_PROCESSED / f"{lid}_training_rows_scored.csv", index=False)

    joblib.dump({"lid": lid, "features": FEATURES, "model": model}, MODEL_DIR / f"{lid}_ridge_model.joblib")

    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["scale"]
    # Convert standardized coefficients back to original feature units.
    coefs_orig = ridge.coef_ / scaler.scale_
    intercept_orig = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
    equation = {
        "intercept": float(intercept_orig),
        "coefficients": {feat: float(coef) for feat, coef in zip(FEATURES, coefs_orig)},
        "formula": "remaining_rise_ft = max(0, intercept + SUM(coef_i * feature_i)); crest = current_stage + remaining_rise",
    }
    with open(REPORT_DIR / f"{lid}_equation.json", "w", encoding="utf-8") as f:
        json.dump(equation, f, indent=2)

    return {
        "lid": lid,
        "status": "ok",
        "event_count": int(events["event_id"].nunique()),
        "training_rows": int(len(train)),
        "test_rows": int(len(test_idx)),
        "mae_ft": float(mean_absolute_error(obs, pred)),
        "rmse_ft": float(math.sqrt(mean_squared_error(obs, pred))),
        "bias_ft": float(np.mean(pred - obs)),
        "r2": float(r2_score(obs, pred)) if len(set(obs)) > 1 else np.nan,
        "equation_file": str(REPORT_DIR / f"{lid}_equation.json"),
        "model_file": str(MODEL_DIR / f"{lid}_ridge_model.joblib"),
    }


def command_train(args: argparse.Namespace) -> None:
    ensure_dirs()
    sites = read_sites(args.sites)
    summaries = []
    for _, row in sites.iterrows():
        lid = row["lid"]
        raw = DATA_RAW / f"{lid}_usgs_stage.csv"
        if not raw.exists():
            print(f"SKIP {lid}: missing {raw}")
            continue
        print(f"Training {lid}...")
        stage_df = pd.read_csv(raw)
        result = train_one_lid(lid, stage_df)
        if result:
            summaries.append(result)
            print(f"  {result}")
    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORT_DIR / "model_summary.csv", index=False)
    print(f"\nWrote {REPORT_DIR / 'model_summary.csv'}")


def command_forecast(args: argparse.Namespace) -> None:
    lid = args.lid.upper()
    model_path = MODEL_DIR / f"{lid}_ridge_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for {lid}: {model_path}")
    bundle = joblib.load(model_path)
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
    X = pd.DataFrame([row])[FEATURES]
    remaining = max(0.0, float(bundle["model"].predict(X)[0]))
    crest = args.stage + remaining
    print(json.dumps({
        "lid": lid,
        "current_stage_ft": args.stage,
        "pred_remaining_rise_ft": round(remaining, 2),
        "pred_crest_stage_ft": round(crest, 2),
        "inputs": row,
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi-gage river crest models from USGS historical stage data.")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("discover-sites", help="Try to map NWS LIDs to USGS site IDs using NWPS metadata")
    ps.add_argument("--sites", default=str(ROOT / "config" / "sites.csv"))
    ps.add_argument("--output", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    ps.set_defaults(func=discover_sites)

    pdn = sub.add_parser("download", help="Download historical USGS gage height data")
    pdn.add_argument("--sites", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    pdn.add_argument("--years", type=int, default=15)
    pdn.set_defaults(func=command_download)

    pt = sub.add_parser("train", help="Detect events and train station-specific models")
    pt.add_argument("--sites", default=str(ROOT / "config" / "sites_with_usgs.csv"))
    pt.add_argument("--years", type=int, default=15)  # kept for future compatibility
    pt.set_defaults(func=command_train)

    pf = sub.add_parser("forecast", help="Forecast crest from current hydrologic features")
    pf.add_argument("--lid", required=True)
    pf.add_argument("--stage", type=float, required=True)
    pf.add_argument("--h0", type=float, required=True)
    pf.add_argument("--elapsed", type=float, required=True)
    pf.add_argument("--r1", type=float, required=True)
    pf.add_argument("--r3", type=float, required=True)
    pf.add_argument("--r6", type=float, required=True)
    pf.add_argument("--r12", type=float, required=True)
    pf.set_defaults(func=command_forecast)
    return p


def main() -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
