#!/usr/bin/env python3
"""Backtest recommended profile models without replacing operational models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import crest_eventset_train as event_train
import nwps_multigage_model as base
import train_model_profiles

BACKTEST_DIR = base.REPORT_DIR / "backtests"
RIDGE_ALPHAS = [0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
HOURS_TO_CREST_BINS = [0, 3, 6, 12, 24, 48, math.inf]
STAGE_BINS = [-math.inf, 5, 10, 15, 20, 30, math.inf]
RISE_RATE_BINS = [-math.inf, 0, 0.05, 0.10, 0.25, 0.50, math.inf]


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )


def profile_settings(profile: pd.Series, args: argparse.Namespace) -> event_train.EventSetSettings:
    event_set = str(profile["recommended_event_set"]).lower().strip()
    min_crest = event_train.as_float(profile.get("recommended_min_crest_stage_ft", ""))
    return event_train.EventSetSettings(
        event_set=event_set,
        min_crest_stage=min_crest,
        min_total_rise=args.min_total_rise,
        below_hours=args.below_hours,
        h0_lookback_hours=args.h0_lookback_hours,
        pre_event_hours=args.pre_event_hours,
        sample_interval=args.sample_interval,
    )


def split_event_groups(train: pd.DataFrame, max_loo_events: int) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    groups = train["event_id"].astype(str).to_numpy()
    unique_events = np.array(sorted(set(groups)))
    if len(unique_events) <= max_loo_events:
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(train[base.FEATURES], train["remaining_rise_ft"], groups=groups))
        return "leave_one_event_out", splits

    n_splits = min(10, len(unique_events))
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(train[base.FEATURES], train["remaining_rise_ft"], groups=groups))
    return f"group_kfold_{n_splits}", splits


def backtest_training_rows(train: pd.DataFrame, lid: str, run_label: str, max_loo_events: int = 30) -> tuple[pd.DataFrame, str]:
    if train.empty or train["event_id"].nunique() < 2:
        return pd.DataFrame(), "too_few_events"

    split_method, splits = split_event_groups(train, max_loo_events=max_loo_events)
    rows: list[pd.DataFrame] = []
    X = train[base.FEATURES]
    y = train["remaining_rise_ft"].clip(lower=0)

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold = train.iloc[test_idx].copy()
        remaining = np.maximum(0, model.predict(X.iloc[test_idx]))
        fold["split_num"] = split_num
        fold["split_method"] = split_method
        fold["lid"] = lid
        fold["run_label"] = run_label
        fold["pred_remaining_rise_ft"] = remaining
        fold["pred_crest_stage_ft"] = fold["stage_ft"] + fold["pred_remaining_rise_ft"]
        fold["error_ft"] = fold["pred_crest_stage_ft"] - fold["observed_crest_stage_ft"]
        fold["abs_error_ft"] = fold["error_ft"].abs()
        fold["underforecast_ft"] = (-fold["error_ft"]).clip(lower=0)
        fold["is_underforecast"] = fold["error_ft"] < 0
        rows.append(fold)

    if not rows:
        return pd.DataFrame(), "no_splits"
    return pd.concat(rows, ignore_index=True), split_method


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = group_cols + [
        "row_count",
        "event_count",
        "gage_count",
        "bias_ft",
        "mae_ft",
        "rmse_ft",
        "abs_error_p50_ft",
        "abs_error_p90_ft",
        "underforecast_count",
        "underforecast_frequency",
        "mean_underforecast_ft",
        "worst_underforecast_ft",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, observed=False) if group_cols else [((), df)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        error = pd.to_numeric(group["error_ft"], errors="coerce").dropna()
        abs_error = error.abs()
        under = (-error).clip(lower=0)
        out = {col: value for col, value in zip(group_cols, key)}
        out.update(
            {
                "row_count": int(len(error)),
                "event_count": int(group["event_id"].nunique()) if "event_id" in group else 0,
                "gage_count": int(group["lid"].nunique()) if "lid" in group else 0,
                "bias_ft": float(error.mean()) if len(error) else math.nan,
                "mae_ft": float(abs_error.mean()) if len(error) else math.nan,
                "rmse_ft": float(math.sqrt(float((error**2).mean()))) if len(error) else math.nan,
                "abs_error_p50_ft": float(abs_error.quantile(0.50)) if len(error) else math.nan,
                "abs_error_p90_ft": float(abs_error.quantile(0.90)) if len(error) else math.nan,
                "underforecast_count": int((error < 0).sum()) if len(error) else 0,
                "underforecast_frequency": float((error < 0).mean()) if len(error) else math.nan,
                "mean_underforecast_ft": float(under.mean()) if len(error) else math.nan,
                "worst_underforecast_ft": float(under.max()) if len(error) else math.nan,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows, columns=columns)


def add_bins(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["hours_to_crest_bin"] = pd.cut(out["hours_to_crest"], bins=HOURS_TO_CREST_BINS, right=False, include_lowest=True).astype(str)
    out["stage_bin_ft"] = pd.cut(out["stage_ft"], bins=STAGE_BINS, right=False, include_lowest=True).astype(str)
    out["r3_rate_bin_ft_per_hr"] = pd.cut(out["r3_ft_per_hr"], bins=RISE_RATE_BINS, right=False, include_lowest=True).astype(str)
    return out


def event_worst_errors(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "lid",
        "run_label",
        "event_id",
        "row_count",
        "max_abs_error_ft",
        "worst_error_ft",
        "worst_underforecast_ft",
        "mean_error_ft",
        "mae_ft",
        "underforecast_frequency",
        "crest_time",
        "observed_crest_stage_ft",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    out_rows: list[dict[str, Any]] = []
    for (lid, run_label, event_id), group in rows.groupby(["lid", "run_label", "event_id"], dropna=False):
        error = pd.to_numeric(group["error_ft"], errors="coerce")
        abs_error = error.abs()
        worst_idx = abs_error.idxmax()
        out_rows.append(
            {
                "lid": lid,
                "run_label": run_label,
                "event_id": event_id,
                "row_count": int(len(group)),
                "max_abs_error_ft": float(abs_error.max()),
                "worst_error_ft": float(rows.loc[worst_idx, "error_ft"]),
                "worst_underforecast_ft": float((-error).clip(lower=0).max()),
                "mean_error_ft": float(error.mean()),
                "mae_ft": float(abs_error.mean()),
                "underforecast_frequency": float((error < 0).mean()),
                "crest_time": group["crest_time"].iloc[0],
                "observed_crest_stage_ft": float(group["observed_crest_stage_ft"].iloc[0]),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["max_abs_error_ft", "lid"], ascending=[False, True])


def underforecast_frequency(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return metric_summary(rows, ["lid", "run_label"])
    out = metric_summary(rows, ["lid", "run_label"])
    return out[[
        "lid",
        "run_label",
        "row_count",
        "event_count",
        "underforecast_count",
        "underforecast_frequency",
        "mean_underforecast_ft",
        "worst_underforecast_ft",
        "bias_ft",
        "mae_ft",
    ]]


def write_reports(rows: pd.DataFrame, summary: pd.DataFrame, output_dir: Path = BACKTEST_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    binned = add_bins(rows) if not rows.empty else rows.copy()
    reports = {
        "backtest_rows": output_dir / "backtest_rows.csv",
        "summary": output_dir / "summary.csv",
        "error_by_hours_to_crest_bins": output_dir / "error_by_hours_to_crest_bins.csv",
        "error_by_stage_bins": output_dir / "error_by_stage_bins.csv",
        "error_by_rise_rate_bins": output_dir / "error_by_rise_rate_bins.csv",
        "event_level_worst_errors": output_dir / "event_level_worst_errors.csv",
        "bias_by_gage": output_dir / "bias_by_gage.csv",
        "underforecast_frequency": output_dir / "underforecast_frequency.csv",
    }

    binned.to_csv(reports["backtest_rows"], index=False)
    summary.to_csv(reports["summary"], index=False)
    metric_summary(binned, ["hours_to_crest_bin"]).to_csv(reports["error_by_hours_to_crest_bins"], index=False)
    metric_summary(binned, ["stage_bin_ft"]).to_csv(reports["error_by_stage_bins"], index=False)
    metric_summary(binned, ["r3_rate_bin_ft_per_hr"]).to_csv(reports["error_by_rise_rate_bins"], index=False)
    event_worst_errors(binned).to_csv(reports["event_level_worst_errors"], index=False)
    metric_summary(binned, ["lid", "run_label"]).to_csv(reports["bias_by_gage"], index=False)
    underforecast_frequency(binned).to_csv(reports["underforecast_frequency"], index=False)
    return reports


def backtest_profile(
    lid: str,
    raw: pd.DataFrame,
    site: pd.Series,
    profile: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = profile_settings(profile, args)
    events = event_train.detect_events(raw, site, settings)
    label = settings.label
    if len(events) < args.min_events:
        return pd.DataFrame(), {
            "lid": lid,
            "run_label": label,
            "event_set": settings.event_set,
            "status": "too_few_events",
            "event_count": int(len(events)),
            "training_rows": 0,
        }

    train = base.feature_rows(raw, events, lid, sample_interval=settings.sample_interval)
    event_count = int(train["event_id"].nunique()) if not train.empty else 0
    if len(train) < args.min_training_rows or event_count < args.min_events:
        return pd.DataFrame(), {
            "lid": lid,
            "run_label": label,
            "event_set": settings.event_set,
            "status": "too_few_training_rows",
            "event_count": int(len(events)),
            "training_rows": int(len(train)),
        }

    rows, split_method = backtest_training_rows(train, lid, label, max_loo_events=args.max_loo_events)
    status = "ok" if not rows.empty else split_method
    summary = {
        "lid": lid,
        "run_label": label,
        "event_set": settings.event_set,
        "status": status,
        "event_count": event_count,
        "training_rows": int(len(train)),
        "backtest_rows": int(len(rows)),
        "split_method": split_method,
    }
    if not rows.empty:
        metrics = metric_summary(rows, ["lid", "run_label"]).iloc[0].to_dict()
        summary.update({f"backtest_{k}": v for k, v in metrics.items() if k not in {"lid", "run_label"}})
    return rows, summary


def run_backtests(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    base.ensure_dirs()
    sites = event_train.read_sites(args.sites)
    profiles = train_model_profiles.read_profiles(args.profiles)

    selected = train_model_profiles.parse_lids(args.lids)
    if selected:
        profiles = profiles[profiles["lid"].isin(selected)].copy()
    if args.limit:
        profiles = profiles.head(args.limit)

    site_by_lid = {str(row["lid"]).upper(): row for _, row in sites.iterrows()}
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for _, profile in profiles.iterrows():
        lid = str(profile["lid"]).upper()
        event_set = str(profile["recommended_event_set"]).lower().strip()
        if event_set == "skip":
            summary_rows.append({"lid": lid, "run_label": "skip", "event_set": "skip", "status": "skipped", "event_count": 0, "training_rows": 0})
            continue
        if event_set not in event_train.EVENT_SETS:
            summary_rows.append({"lid": lid, "run_label": event_set, "event_set": event_set, "status": "bad_profile_event_set", "event_count": 0, "training_rows": 0})
            continue
        site = site_by_lid.get(lid)
        if site is None:
            summary_rows.append({"lid": lid, "run_label": event_set, "event_set": event_set, "status": "missing_site_row", "event_count": 0, "training_rows": 0})
            continue
        raw_path = base.DATA_RAW / f"{lid}_usgs_stage.csv"
        if not raw_path.exists():
            summary_rows.append({"lid": lid, "run_label": event_set, "event_set": event_set, "status": "missing_raw", "event_count": 0, "training_rows": 0})
            continue

        print(f"Backtesting {lid} ({event_set})", flush=True)
        rows, summary = backtest_profile(lid, train_model_profiles.read_clean_raw(raw_path), site, profile, args)
        summary_rows.append(summary)
        if not rows.empty:
            all_rows.append(rows)
        print(f"  {summary}", flush=True)

    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    reports = write_reports(rows_df, summary_df, Path(args.output_dir))
    return rows_df, summary_df, reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest recommended profile models and write CSV evaluation reports.")
    parser.add_argument("--sites", default=str(base.ROOT / "config" / "sites_with_usgs.csv"))
    parser.add_argument("--profiles", default=str(base.ROOT / "config" / "model_profiles.csv"))
    parser.add_argument("--lids", default="", help="Optional comma-separated LIDs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=str(BACKTEST_DIR))
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--min-training-rows", type=int, default=40)
    parser.add_argument("--max-loo-events", type=int, default=30, help="Use leave-one-event-out up to this many events, then grouped K-fold.")
    parser.add_argument("--min-total-rise", type=float, default=1.0)
    parser.add_argument("--below-hours", type=float, default=24.0)
    parser.add_argument("--h0-lookback-hours", type=float, default=48.0)
    parser.add_argument("--pre-event-hours", type=float, default=24.0)
    parser.add_argument("--sample-interval", default="1h")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _rows, summary, reports = run_backtests(args)
    ok = int((summary.get("status", pd.Series(dtype=str)) == "ok").sum()) if not summary.empty else 0
    print(f"Backtested {ok} profile model(s)")
    for name, path in reports.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
