#!/usr/bin/env python3
"""Shared model preparation, event, feature, and scoring helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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


def coerce_float(value: Any) -> float | None:
    """Convert scalar-ish values to finite floats; ignore containers and blanks."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return None
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def clean_stage_data(df: pd.DataFrame, stage_col: str = "stage_ft") -> pd.DataFrame:
    """Return rows with usable stage values only.

    USGS/NWPS stage files can contain blanks, NaN values, no-data sentinels
    such as -999999, and physically implausible values. Keep the original
    columns, coerce the stage column to numeric, and retain only values in the
    plausible modeling range (-1000, 200).
    """
    columns = list(df.columns) if stage_col in df.columns else [*df.columns, stage_col]
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    if stage_col not in out.columns:
        out[stage_col] = np.nan
    out[stage_col] = pd.to_numeric(out[stage_col], errors="coerce")
    out = out[out[stage_col].notna() & (out[stage_col] > -1000) & (out[stage_col] < 200)].copy()
    return out


def prep_stage(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["datetime", "stage_ft"])
    out = clean_stage_data(df)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime", "stage_ft"]).sort_values("datetime")
    if out.empty:
        return pd.DataFrame(columns=["datetime", "stage_ft"])

    out = out.set_index("datetime")
    s = out["stage_ft"].resample(freq).median().interpolate(limit=4)
    clean = s.to_frame()
    clean["stage_ft"] = clean["stage_ft"].rolling(3, center=True, min_periods=1).median()
    return clean.reset_index().dropna(subset=["stage_ft"])


def choose_threshold(
    stage: pd.Series,
    site_row: pd.Series | dict[str, Any] | None = None,
    threshold_columns: list[str] | None = None,
) -> float:
    if site_row is not None:
        for col in threshold_columns or ["event_threshold_ft", "action_stage_ft", "flood_stage_ft"]:
            x = coerce_float(site_row.get(col, "") if isinstance(site_row, dict) else site_row.get(col, ""))
            if x is not None:
                return x

    q90 = float(stage.quantile(0.90))
    q75 = float(stage.quantile(0.75))
    return max(q90, q75 + 1.0)


def choose_event_threshold(stage: pd.Series, site_row: pd.Series | dict[str, Any] | None = None) -> float:
    return choose_threshold(stage, site_row, ["event_threshold_ft", "action_stage_ft", "flood_stage_ft"])


def detect_events(
    stage_df: pd.DataFrame,
    site_row: pd.Series | dict[str, Any] | None = None,
    settings: Any | None = None,
) -> pd.DataFrame:
    if settings is None:
        settings = type(
            "EventSettingsDefaults",
            (),
            {
                "min_total_rise_ft": 1.0,
                "below_threshold_hours_to_end": 24.0,
                "pre_crest_lookback_hours": 48.0,
                "include_pre_event_hours": 24.0,
                "sample_interval": "1h",
            },
        )()
    df = prep_stage(stage_df)
    if df.empty:
        return pd.DataFrame()

    threshold = choose_event_threshold(df["stage_ft"], site_row)
    df["above"] = df["stage_ft"] >= threshold
    events: list[tuple[int, int]] = []
    in_event = False
    start_idx: int | None = None
    below_count = 0
    min_gap_steps = max(1, int(settings.below_threshold_hours_to_end * 4))
    pre_steps = max(0, int(settings.include_pre_event_hours * 4))

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

        pre_peak = ev[ev["datetime"] <= peak_time].copy()
        if pre_peak.empty:
            continue

        lookback_start = peak_time - pd.Timedelta(hours=settings.pre_crest_lookback_hours)
        rise_window = pre_peak[pre_peak["datetime"] >= lookback_start]
        if rise_window.empty:
            rise_window = pre_peak

        h0_idx = rise_window["stage_ft"].idxmin()
        h0_time = ev.loc[h0_idx, "datetime"]
        h0_stage = float(ev.loc[h0_idx, "stage_ft"])
        total_rise = peak_stage - h0_stage
        if total_rise < settings.min_total_rise_ft:
            continue

        rows.append(
            {
                "event_id": f"E{event_num:04d}_{peak_time.strftime('%Y%m%d%H%M')}",
                "start_time": ev.iloc[0]["datetime"],
                "rise_start_time": h0_time,
                "crest_time": peak_time,
                "end_time": ev.iloc[-1]["datetime"],
                "h0_stage_ft": h0_stage,
                "crest_stage_ft": peak_stage,
                "total_rise_ft": total_rise,
                "threshold_used_ft": threshold,
                "duration_hr": (ev.iloc[-1]["datetime"] - ev.iloc[0]["datetime"]).total_seconds() / 3600,
                "rise_duration_hr": (peak_time - h0_time).total_seconds() / 3600,
            }
        )
    return pd.DataFrame(rows)


def nearest_stage_at(df: pd.DataFrame, t: pd.Timestamp, tolerance: pd.Timedelta = pd.Timedelta(minutes=30)) -> float:
    if t in df.index:
        return float(df.loc[t, "stage_ft"])
    idx = df.index.get_indexer([t], method="nearest", tolerance=tolerance)
    if idx[0] == -1:
        return np.nan
    return float(df.iloc[idx[0]]["stage_ft"])


def feature_rows(stage_df: pd.DataFrame, events: pd.DataFrame, lid: str, sample_interval: str = "1h") -> pd.DataFrame:
    df = prep_stage(stage_df).set_index("datetime")
    rows: list[dict[str, Any]] = []

    for _, ev in events.iterrows():
        event_id = ev["event_id"]
        crest_time = pd.to_datetime(ev["crest_time"], utc=True)
        rise_start = pd.to_datetime(ev["rise_start_time"], utc=True)
        h0 = float(ev["h0_stage_ft"])
        crest = float(ev["crest_stage_ft"])
        window = df[(df.index >= rise_start) & (df.index < crest_time)].copy()
        if window.empty:
            continue

        sample_times = window.resample(sample_interval).nearest().dropna().index
        for t in sample_times:
            stage = nearest_stage_at(df, t)
            if pd.isna(stage):
                continue

            def rate(hours: int) -> float:
                past = nearest_stage_at(df, t - pd.Timedelta(hours=hours))
                return np.nan if pd.isna(past) else (stage - past) / hours

            r1, r3, r6, r12 = rate(1), rate(3), rate(6), rate(12)
            remaining = crest - stage
            if remaining < -0.05:
                continue

            rows.append(
                {
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
                    "hours_to_crest": (crest_time - t).total_seconds() / 3600,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.dropna(subset=FEATURES + ["remaining_rise_ft"]).reset_index(drop=True)


def original_scale_equation(model: Any) -> dict[str, Any]:
    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["scale"]
    coefs_orig = ridge.coef_ / scaler.scale_
    intercept_orig = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
    return {
        "intercept": float(intercept_orig),
        "coefficients": {feat: float(coef) for feat, coef in zip(FEATURES, coefs_orig)},
        "formula": "remaining_rise_ft = max(0, intercept + SUM(coef_i * feature_i)); crest = current_stage + remaining_rise",
    }


def safe_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or len(set(np.round(obs, 6))) < 2:
        return float("nan")
    return float(r2_score(obs, pred))


def residual_stats(error_ft: pd.Series) -> dict[str, float]:
    err = pd.to_numeric(error_ft, errors="coerce").dropna()
    if err.empty:
        return {}
    under = (-err).clip(lower=0)
    return {
        "error_p10_ft": float(err.quantile(0.10)),
        "error_p50_ft": float(err.quantile(0.50)),
        "error_p90_ft": float(err.quantile(0.90)),
        "underforecast_p75_ft": float(under.quantile(0.75)),
        "underforecast_p90_ft": float(under.quantile(0.90)),
        "abs_error_p75_ft": float(err.abs().quantile(0.75)),
        "abs_error_p90_ft": float(err.abs().quantile(0.90)),
    }
