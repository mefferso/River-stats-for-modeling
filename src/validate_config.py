#!/usr/bin/env python3
"""Validate river model configuration CSV files."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EVENT_SETS = {"all", "flood", "moderate", "major", "custom", "skip"}

SITE_REQUIRED_COLUMNS = {"lid", "name"}
PROFILE_REQUIRED_COLUMNS = {"lid", "recommended_event_set", "recommended_min_crest_stage_ft", "reason"}
OVERRIDE_REQUIRED_COLUMNS = {"lid"}

SITE_THRESHOLD_COLUMNS = [
    "event_start_threshold_ft",
    "event_threshold_ft",
    "min_crest_stage_ft",
    "action_stage_ft",
    "minor_stage_ft",
    "flood_stage_ft",
    "moderate_stage_ft",
    "major_stage_ft",
]
PROFILE_THRESHOLD_COLUMNS = ["recommended_min_crest_stage_ft"]
OVERRIDE_THRESHOLD_COLUMNS = [
    "min_forecast_stage_ft",
    "early_window_ft",
    "min_stage_rise_ft",
    "min_r3_rise_rate",
    "max_remaining_rise_ft",
]


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def normalized_lids(df: pd.DataFrame) -> pd.Series:
    if "lid" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=str)
    return df["lid"].astype(str).str.upper().str.strip()


def row_label(df_name: str, row_index: int, lid: str | None = None) -> str:
    row_num = row_index + 2  # CSV header is line 1.
    if lid:
        return f"{df_name} row {row_num} ({lid})"
    return f"{df_name} row {row_num}"


def require_columns(df: pd.DataFrame, df_name: str, required: Iterable[str], result: ValidationResult) -> None:
    missing = sorted(set(required) - set(df.columns))
    for col in missing:
        result.error(f"{df_name}: missing required column '{col}'")


def check_blank_required_fields(df: pd.DataFrame, df_name: str, fields: Iterable[str], result: ValidationResult) -> None:
    for field in fields:
        if field not in df.columns:
            continue
        blank = df[field].astype(str).str.strip() == ""
        for idx in df.index[blank]:
            lid = str(df.at[idx, "lid"]).strip().upper() if "lid" in df.columns else ""
            result.error(f"{row_label(df_name, int(idx), lid)}: blank required field '{field}'")


def check_duplicate_lids(df: pd.DataFrame, df_name: str, result: ValidationResult) -> None:
    lids = normalized_lids(df)
    blank = lids == ""
    duplicates = sorted(lid for lid in lids[~blank].unique() if (lids == lid).sum() > 1)
    for lid in duplicates:
        rows = [str(int(i) + 2) for i in df.index[lids == lid]]
        result.error(f"{df_name}: duplicate LID '{lid}' on rows {', '.join(rows)}")


def parse_optional_float(value: object) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def check_threshold_values(
    df: pd.DataFrame,
    df_name: str,
    columns: Iterable[str],
    result: ValidationResult,
    *,
    allow_zero: bool = False,
) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        for idx, raw in df[col].items():
            text = str(raw).strip()
            if text == "":
                continue
            lid = normalized_lids(df).at[idx]
            value = parse_optional_float(text)
            if value is None:
                result.error(f"{row_label(df_name, int(idx), lid)}: '{col}' must be numeric, got {text!r}")
                continue
            lower_bad = value < 0 if allow_zero else value <= 0
            if lower_bad or value >= 200:
                expected = "0 <= value < 200" if allow_zero else "0 < value < 200"
                result.error(
                    f"{row_label(df_name, int(idx), lid)}: suspicious '{col}' value {value:g}; expected {expected}"
                )


def check_site_threshold_order(sites: pd.DataFrame, result: ValidationResult) -> None:
    ordered = ["action_stage_ft", "minor_stage_ft", "moderate_stage_ft", "major_stage_ft"]
    lids = normalized_lids(sites)
    for idx, row in sites.iterrows():
        previous_col = ""
        previous_value: float | None = None
        for col in ordered:
            if col not in sites.columns:
                continue
            value = parse_optional_float(row.get(col, ""))
            if value is None:
                continue
            if previous_value is not None and value < previous_value:
                result.error(
                    f"{row_label('sites.csv', int(idx), lids.at[idx])}: suspicious thresholds; "
                    f"'{col}' ({value:g}) is below '{previous_col}' ({previous_value:g})"
                )
            previous_col = col
            previous_value = value


def validate_configs(
    sites_path: str | Path = ROOT / "config" / "sites.csv",
    profiles_path: str | Path = ROOT / "config" / "model_profiles.csv",
    overrides_path: str | Path = ROOT / "config" / "forecast_overrides.csv",
) -> ValidationResult:
    result = ValidationResult()

    sites = read_csv(sites_path)
    profiles = read_csv(profiles_path)
    overrides = read_csv(overrides_path)

    require_columns(sites, "sites.csv", SITE_REQUIRED_COLUMNS, result)
    require_columns(profiles, "model_profiles.csv", PROFILE_REQUIRED_COLUMNS, result)
    require_columns(overrides, "forecast_overrides.csv", OVERRIDE_REQUIRED_COLUMNS, result)
    if result.errors:
        return result

    check_duplicate_lids(sites, "sites.csv", result)
    check_duplicate_lids(profiles, "model_profiles.csv", result)
    check_duplicate_lids(overrides, "forecast_overrides.csv", result)

    check_blank_required_fields(sites, "sites.csv", SITE_REQUIRED_COLUMNS, result)
    check_blank_required_fields(profiles, "model_profiles.csv", {"lid", "recommended_event_set", "reason"}, result)
    check_blank_required_fields(overrides, "forecast_overrides.csv", OVERRIDE_REQUIRED_COLUMNS, result)

    check_threshold_values(sites, "sites.csv", SITE_THRESHOLD_COLUMNS, result)
    check_threshold_values(profiles, "model_profiles.csv", PROFILE_THRESHOLD_COLUMNS, result)
    check_threshold_values(overrides, "forecast_overrides.csv", OVERRIDE_THRESHOLD_COLUMNS, result, allow_zero=True)
    check_site_threshold_order(sites, result)

    site_lids = set(normalized_lids(sites)) - {""}
    site_by_lid = {lid: row for lid, row in zip(normalized_lids(sites), sites.to_dict("records"), strict=False) if lid}

    profile_lids = normalized_lids(profiles)
    for idx, profile in profiles.iterrows():
        lid = profile_lids.at[idx]
        if not lid:
            continue
        event_set = str(profile.get("recommended_event_set", "")).lower().strip()
        if event_set not in ALLOWED_EVENT_SETS:
            result.error(
                f"{row_label('model_profiles.csv', int(idx), lid)}: invalid recommended_event_set {event_set!r}; "
                f"expected one of {sorted(ALLOWED_EVENT_SETS)}"
            )
        if event_set == "custom" and str(profile.get("recommended_min_crest_stage_ft", "")).strip() == "":
            result.error(f"{row_label('model_profiles.csv', int(idx), lid)}: custom profile missing recommended_min_crest_stage_ft")
        if lid not in site_lids:
            result.error(f"{row_label('model_profiles.csv', int(idx), lid)}: profile LID not found in sites.csv")
            continue
        if event_set != "skip" and not str(site_by_lid[lid].get("usgs_site", "")).strip():
            result.error(f"{row_label('model_profiles.csv', int(idx), lid)}: missing USGS ID in sites.csv for active profile")

    override_lids = normalized_lids(overrides)
    for idx, lid in override_lids.items():
        if lid and lid not in site_lids:
            result.error(f"{row_label('forecast_overrides.csv', int(idx), lid)}: override LID not found in sites.csv")

    configured_profile_lids = set(profile_lids) - {""}
    for lid in sorted(site_lids - configured_profile_lids):
        result.warn(f"sites.csv ({lid}): no model_profiles.csv row")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate river modeling configuration CSV files.")
    parser.add_argument("--sites", default=str(ROOT / "config" / "sites.csv"), help="Path to sites.csv")
    parser.add_argument("--profiles", default=str(ROOT / "config" / "model_profiles.csv"), help="Path to model_profiles.csv")
    parser.add_argument("--overrides", default=str(ROOT / "config" / "forecast_overrides.csv"), help="Path to forecast_overrides.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_configs(args.sites, args.profiles, args.overrides)

    for warning in result.warnings:
        print(f"WARN: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if result.ok:
        print("Config validation passed")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
