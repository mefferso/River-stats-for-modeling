#!/usr/bin/env python3
"""Train crest models using crest-category event filters.

This is a thin add-on around nwps_multigage_model.py. It separates:
1) event-window detection threshold, and
2) minimum crest stage required for training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nwps_multigage_model as base

EVENT_SETS = {"all", "flood", "moderate", "major", "custom"}
EXTRA_SITE_COLUMNS = [
    "event_start_threshold_ft",
    "min_crest_stage_ft",
    "minor_stage_ft",
    "moderate_stage_ft",
    "major_stage_ft",
]


@dataclass
class EventSetSettings:
    event_set: str = "flood"
    min_crest_stage: float | None = None
    min_total_rise: float = 1.0
    below_hours: float = 24.0
    h0_lookback_hours: float = 48.0
    pre_event_hours: float = 24.0
    sample_interval: str = "1h"

    @property
    def label(self) -> str:
        if self.event_set == "custom" and self.min_crest_stage is not None:
            return safe_label(f"custom_{self.min_crest_stage:g}ft_plus")
        if self.event_set == "all":
            return "all_events"
        return safe_label(f"{self.event_set}_plus")


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip()).strip("_") or "model"


def as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except TypeError:
        if value is None:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def read_sites(path: str | Path) -> pd.DataFrame:
    df = base.read_sites(path)
    for col in EXTRA_SITE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def fetch_json(url: str, tries: int = 3) -> Any | None:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=45, headers={"User-Agent": "lix-river-eventset-model/0.1"})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    print(f"WARN: failed {url}: {last}", file=sys.stderr)
    return None


def recursive_thresholds(obj: Any) -> dict[str, float]:
    """Best-effort extraction of action/minor/moderate/major stage thresholds."""
    out: dict[str, float] = {}
    mapping = {
        "action": "action_stage_ft",
        "minor": "minor_stage_ft",
        "flood": "flood_stage_ft",
        "moderate": "moderate_stage_ft",
        "major": "major_stage_ft",
    }

    def keep(key: str, value: Any) -> None:
        lk = key.lower()
        value_f = as_float(value)
        if value_f is None or value_f <= 0 or value_f > 200:
            return
        if any(bad in lk for bad in ["flow", "cfs", "latitude", "longitude", "lat", "lon"]):
            return
        for word, col in mapping.items():
            if word in lk and col not in out:
                out[col] = value_f

    if isinstance(obj, dict):
        lower_keys = {str(k).lower(): k for k in obj.keys()}
        for k, v in obj.items():
            if isinstance(v, (str, int, float)):
                keep(str(k), v)
            elif isinstance(v, dict):
                lk = str(k).lower()
                for stage_key in ["stage", "stage_ft", "value", "threshold"]:
                    if stage_key in lower_keys:
                        keep(lk, obj[lower_keys[stage_key]])
                nested_lower = {str(kk).lower(): kk for kk in v.keys()}
                for stage_key in ["stage", "stage_ft", "value", "threshold"]:
                    if stage_key in nested_lower:
                        keep(lk, v[nested_lower[stage_key]])
                for kk, vv in recursive_thresholds(v).items():
                    out.setdefault(kk, vv)
            elif isinstance(v, list):
                for item in v:
                    for kk, vv in recursive_thresholds(item).items():
                        out.setdefault(kk, vv)
    elif isinstance(obj, list):
        for item in obj:
            for kk, vv in recursive_thresholds(item).items():
                out.setdefault(kk, vv)
    return out


def command_discover_thresholds(args: argparse.Namespace) -> None:
    sites = read_sites(args.sites)
    rows: list[dict[str, Any]] = []
    urls = [
        "https://api.water.noaa.gov/nwps/v1/gauges/{lid}",
        "https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow",
    ]
    for _, row in sites.iterrows():
        out = row.to_dict()
        lid = str(row["lid"]).upper()
        found: dict[str, float] = {}
        for tmpl in urls:
            meta = fetch_json(tmpl.format(lid=lid.lower())) or fetch_json(tmpl.format(lid=lid.upper()))
            if meta:
                found.update(recursive_thresholds(meta))
        for col, val in found.items():
            if not str(out.get(col, "")).strip() or args.overwrite:
                out[col] = val
        rows.append(out)
        if found:
            print(f"{lid:6s} " + ", ".join(f"{k}={v}" for k, v in found.items()))
        else:
            print(f"{lid:6s} no thresholds found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output}")


def event_start_threshold(stage: pd.Series, site: pd.Series | dict[str, Any]) -> float:
    for col in ["event_start_threshold_ft", "event_threshold_ft", "action_stage_ft", "minor_stage_ft", "flood_stage_ft"]:
        value = as_float(site.get(col, ""))
        if value is not None:
            return value
    q90 = float(stage.quantile(0.90))
    q75 = float(stage.quantile(0.75))
    return max(q90, q75 + 1.0)


def min_crest_stage(site: pd.Series | dict[str, Any], settings: EventSetSettings) -> tuple[float | None, str]:
    if settings.min_crest_stage is not None:
        return settings.min_crest_stage, "explicit"
    override = as_float(site.get("min_crest_stage_ft", ""))
    if override is not None:
        return override, "site_min_crest_stage_ft"
    if settings.event_set == "all":
        return None, "all_events"
    if settings.event_set == "custom":
        return None, "custom_no_min_crest"

    if settings.event_set == "major":
        order = ["major_stage_ft"]
    elif settings.event_set == "moderate":
        order = ["moderate_stage_ft"]
    else:
        order = ["minor_stage_ft", "flood_stage_ft"]

    for col in order:
        value = as_float(site.get(col, ""))
        if value is not None:
            return value, col
    return None, f"{settings.event_set}_threshold_missing"


def detect_events(stage_df: pd.DataFrame, site: pd.Series | dict[str, Any], settings: EventSetSettings) -> pd.DataFrame:
    df = base.prep_stage(stage_df)
    if df.empty:
        return pd.DataFrame()

    start_thresh = event_start_threshold(df["stage_ft"], site)
    min_crest, min_source = min_crest_stage(site, settings)
    df["above"] = df["stage_ft"] >= start_thresh

    events: list[tuple[int, int]] = []
    in_event = False
    start_idx: int | None = None
    below_count = 0
    min_gap_steps = max(1, int(settings.below_hours * 4))
    pre_steps = max(0, int(settings.pre_event_hours * 4))

    for i, above in enumerate(df["above"].to_numpy()):
        if above and not in_event:
            in_event = True
            start_idx = max(0, i - pre_steps)
            below_count = 0
        elif in_event:
            if above:
                below_count = 0
            else:
                below_count += 1
                if below_count >= min_gap_steps:
                    events.append((start_idx or 0, i))
                    in_event = False
                    start_idx = None
                    below_count = 0
    if in_event and start_idx is not None:
        events.append((start_idx, len(df) - 1))

    rows: list[dict[str, Any]] = []
    for event_num, (a, b) in enumerate(events, start=1):
        ev = df.iloc[a : b + 1].copy()
        if len(ev) < 12:
            continue
        peak_idx = ev["stage_ft"].idxmax()
        peak_time = ev.loc[peak_idx, "datetime"]
        peak_stage = float(ev.loc[peak_idx, "stage_ft"])
        if min_crest is not None and peak_stage < min_crest:
            continue

        pre_peak = ev[ev["datetime"] <= peak_time]
        lookback_start = peak_time - pd.Timedelta(hours=settings.h0_lookback_hours)
        rise_window = pre_peak[pre_peak["datetime"] >= lookback_start]
        if rise_window.empty:
            rise_window = pre_peak
        if rise_window.empty:
            continue
        h0_idx = rise_window["stage_ft"].idxmin()
        h0_time = ev.loc[h0_idx, "datetime"]
        h0_stage = float(ev.loc[h0_idx, "stage_ft"])
        total_rise = peak_stage - h0_stage
        if total_rise < settings.min_total_rise:
            continue

        rows.append(
            {
                "event_id": f"E{event_num:04d}_{peak_time.strftime('%Y%m%d%H%M')}",
                "event_set": settings.event_set,
                "run_label": settings.label,
                "start_time": ev.iloc[0]["datetime"],
                "rise_start_time": h0_time,
                "crest_time": peak_time,
                "end_time": ev.iloc[-1]["datetime"],
                "h0_stage_ft": h0_stage,
                "crest_stage_ft": peak_stage,
                "total_rise_ft": total_rise,
                "event_start_threshold_used_ft": start_thresh,
                "min_crest_stage_used_ft": min_crest if min_crest is not None else "",
                "min_crest_stage_source": min_source,
                "duration_hr": (ev.iloc[-1]["datetime"] - ev.iloc[0]["datetime"]).total_seconds() / 3600,
                "rise_duration_hr": (peak_time - h0_time).total_seconds() / 3600,
            }
        )
    return pd.DataFrame(rows)


def train_one(lid: str, stage_df: pd.DataFrame, site: pd.Series | dict[str, Any], settings: EventSetSettings) -> dict[str, Any]:
    label = settings.label
    events = detect_events(stage_df, site, settings)
    event_file = base.DATA_PROCESSED / f"{lid}_{label}_events.csv"
    train_file = base.DATA_PROCESSED / f"{lid}_{label}_training_rows.csv"
    scored_file = base.DATA_PROCESSED / f"{lid}_{label}_training_rows_scored.csv"

    events.to_csv(event_file, index=False)
    if len(events) < 5:
        return {"lid": lid, "run_label": label, "status": "too_few_events", "event_count": int(len(events))}

    train = base.feature_rows(stage_df, events, lid)
    train.to_csv(train_file, index=False)
    if len(train) < 40 or train["event_id"].nunique() < 5:
        return {"lid": lid, "run_label": label, "status": "too_few_training_rows", "event_count": int(len(events)), "training_rows": int(len(train))}

    X = train[base.FEATURES]
    y = train["remaining_rise_ft"].clip(lower=0)
    groups = train["event_id"]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])),
    ])

    if train["event_id"].nunique() >= 7:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.maximum(0, model.predict(X.iloc[test_idx]))
        obs = y.iloc[test_idx].to_numpy()
        holdout_rows = len(test_idx)
        holdout_events = int(train.iloc[test_idx]["event_id"].nunique())
    else:
        model.fit(X, y)
        pred = np.maximum(0, model.predict(X))
        obs = y.to_numpy()
        holdout_rows = 0
        holdout_events = 0

    mae = float(mean_absolute_error(obs, pred))
    rmse = float(math.sqrt(mean_squared_error(obs, pred)))
    bias = float(np.mean(pred - obs))
    r2 = float(r2_score(obs, pred)) if len(obs) > 1 and len(set(np.round(obs, 6))) > 1 else float("nan")

    model.fit(X, y)
    full_pred = np.maximum(0, model.predict(X))
    train["pred_remaining_rise_ft"] = full_pred
    train["pred_crest_stage_ft"] = train["stage_ft"] + train["pred_remaining_rise_ft"]
    train["error_ft"] = train["pred_crest_stage_ft"] - train["observed_crest_stage_ft"]
    train.to_csv(scored_file, index=False)

    err = pd.to_numeric(train["error_ft"], errors="coerce").dropna()
    under = (-err).clip(lower=0)
    event_start_used = pd.to_numeric(events["event_start_threshold_used_ft"], errors="coerce")
    min_crest_used = pd.to_numeric(events["min_crest_stage_used_ft"], errors="coerce")
    meta = {
        "lid": lid,
        "name": site.get("name", ""),
        "event_settings": settings.__dict__ | {"label": label},
        "event_count": int(events["event_id"].nunique()),
        "training_rows": int(len(train)),
        "event_start_threshold_used_ft": float(event_start_used.median()),
        "min_crest_stage_used_ft": float(min_crest_used.median()) if min_crest_used.notna().any() else None,
        "min_crest_stage_source": str(events["min_crest_stage_source"].mode().iloc[0]),
        "features": base.FEATURES,
        "skill": {
            "mae_ft": mae,
            "rmse_ft": rmse,
            "bias_ft": bias,
            "r2": r2,
            "holdout_rows": holdout_rows,
            "holdout_events": holdout_events,
            "error_p10_ft": float(err.quantile(0.10)),
            "error_p50_ft": float(err.quantile(0.50)),
            "error_p90_ft": float(err.quantile(0.90)),
            "underforecast_p75_ft": float(under.quantile(0.75)),
            "underforecast_p90_ft": float(under.quantile(0.90)),
            "abs_error_p75_ft": float(err.abs().quantile(0.75)),
            "abs_error_p90_ft": float(err.abs().quantile(0.90)),
        },
        "equation": base.original_scale_equation(model),
    }

    model_file = base.MODEL_DIR / f"{lid}_{label}_ridge_model.joblib"
    equation_file = base.REPORT_DIR / f"{lid}_{label}_equation.json"
    joblib.dump({"lid": lid, "features": base.FEATURES, "model": model, "metadata": meta}, model_file)
    with open(equation_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "lid": lid,
        "run_label": label,
        "event_set": settings.event_set,
        "status": "ok",
        "event_count": int(events["event_id"].nunique()),
        "training_rows": int(len(train)),
        "holdout_rows": holdout_rows,
        "holdout_events": holdout_events,
        "event_start_threshold_used_ft": meta["event_start_threshold_used_ft"],
        "min_crest_stage_used_ft": meta["min_crest_stage_used_ft"],
        "min_crest_stage_source": meta["min_crest_stage_source"],
        "mae_ft": mae,
        "rmse_ft": rmse,
        "bias_ft": bias,
        "r2": r2,
        "model_file": str(model_file),
        "equation_file": str(equation_file),
    }


def command_train(args: argparse.Namespace) -> None:
    base.ensure_dirs()
    sites = read_sites(args.sites)
    if args.limit:
        sites = sites.head(args.limit)
    settings = EventSetSettings(
        event_set=args.event_set,
        min_crest_stage=args.min_crest_stage,
        min_total_rise=args.min_total_rise,
        below_hours=args.below_hours,
        h0_lookback_hours=args.h0_lookback_hours,
        pre_event_hours=args.pre_event_hours,
        sample_interval=args.sample_interval,
    )
    rows: list[dict[str, Any]] = []
    for _, site in sites.iterrows():
        lid = str(site["lid"]).upper()
        raw = base.DATA_RAW / f"{lid}_usgs_stage.csv"
        if not raw.exists():
            print(f"SKIP {lid}: missing {raw}")
            continue
        print(f"Training {lid} ({settings.label})")
        result = train_one(lid, pd.read_csv(raw), site, settings)
        rows.append(result)
        print(f"  {result}")
    summary = pd.DataFrame(rows)
    out = base.REPORT_DIR / f"model_summary_{settings.label}.csv"
    summary.to_csv(out, index=False)
    summary.to_csv(base.REPORT_DIR / "model_summary.csv", index=False)
    print(f"\nWrote {out}")
    print(f"Also wrote {base.REPORT_DIR / 'model_summary.csv'}")


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sites", default=str(base.ROOT / "config" / "sites_with_usgs.csv"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--event-set", choices=sorted(EVENT_SETS), default="flood")
    p.add_argument("--min-crest-stage", type=float, default=None)
    p.add_argument("--min-total-rise", type=float, default=1.0)
    p.add_argument("--below-hours", type=float, default=24.0)
    p.add_argument("--h0-lookback-hours", type=float, default=48.0)
    p.add_argument("--pre-event-hours", type=float, default=24.0)
    p.add_argument("--sample-interval", default="1h")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train crest models with crest-category event filters.")
    sub = parser.add_subparsers(dest="command", required=True)
    th = sub.add_parser("discover-thresholds", help="Try to fill action/minor/moderate/major stage thresholds from NWPS metadata")
    th.add_argument("--sites", default=str(base.ROOT / "config" / "sites_with_usgs.csv"))
    th.add_argument("--output", default=str(base.ROOT / "config" / "sites_with_usgs.csv"))
    th.add_argument("--overwrite", action="store_true")
    th.set_defaults(func=command_discover_thresholds)

    tr = sub.add_parser("train", help="Train models using flood/moderate/major event sets")
    add_train_args(tr)
    tr.set_defaults(func=command_train)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
