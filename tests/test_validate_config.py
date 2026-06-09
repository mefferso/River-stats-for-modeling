from __future__ import annotations

from pathlib import Path

import pandas as pd

import validate_config


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> Path:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def valid_config_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    sites = write_csv(
        tmp_path / "sites.csv",
        [
            {"lid": "AAA1", "name": "Alpha", "usgs_site": "01234567", "action_stage_ft": "10", "minor_stage_ft": "12", "moderate_stage_ft": "15", "major_stage_ft": "20"},
            {"lid": "BBB1", "name": "Beta", "usgs_site": "", "action_stage_ft": "", "minor_stage_ft": "", "moderate_stage_ft": "", "major_stage_ft": ""},
            {"lid": "CCC1", "name": "Gamma", "usgs_site": "07654321", "action_stage_ft": "", "minor_stage_ft": "", "moderate_stage_ft": "", "major_stage_ft": ""},
        ],
        ["lid", "name", "usgs_site", "action_stage_ft", "minor_stage_ft", "moderate_stage_ft", "major_stage_ft"],
    )
    profiles = write_csv(
        tmp_path / "model_profiles.csv",
        [
            {"lid": "AAA1", "recommended_event_set": "flood", "recommended_min_crest_stage_ft": "", "reason": "baseline"},
            {"lid": "BBB1", "recommended_event_set": "skip", "recommended_min_crest_stage_ft": "", "reason": "no mapping"},
            {"lid": "CCC1", "recommended_event_set": "custom", "recommended_min_crest_stage_ft": "14", "reason": "manual"},
        ],
        ["lid", "recommended_event_set", "recommended_min_crest_stage_ft", "reason"],
    )
    overrides = write_csv(
        tmp_path / "forecast_overrides.csv",
        [{"lid": "AAA1", "min_forecast_stage_ft": "11", "early_window_ft": "1.5", "min_stage_rise_ft": "1", "min_r3_rise_rate": "0.1", "max_remaining_rise_ft": ""}],
        ["lid", "min_forecast_stage_ft", "early_window_ft", "min_stage_rise_ft", "min_r3_rise_rate", "max_remaining_rise_ft"],
    )
    return sites, profiles, overrides


def test_validate_config_accepts_valid_files(tmp_path: Path) -> None:
    paths = valid_config_paths(tmp_path)

    result = validate_config.validate_configs(*paths)

    assert result.errors == []


def test_validate_config_reports_cross_file_and_profile_errors(tmp_path: Path) -> None:
    sites, profiles, overrides = valid_config_paths(tmp_path)
    pd.DataFrame(
        [
            {"lid": "AAA1", "name": "Alpha", "usgs_site": "01234567"},
            {"lid": "AAA1", "name": "Alpha duplicate", "usgs_site": "01234568"},
            {"lid": "DDD1", "name": "No USGS", "usgs_site": ""},
        ]
    ).to_csv(sites, index=False)
    pd.DataFrame(
        [
            {"lid": "DDD1", "recommended_event_set": "flood", "recommended_min_crest_stage_ft": "", "reason": "active but unmapped"},
            {"lid": "EEE1", "recommended_event_set": "custom", "recommended_min_crest_stage_ft": "", "reason": "missing site and crest"},
        ]
    ).to_csv(profiles, index=False)
    pd.DataFrame([{"lid": "ZZZ1", "min_forecast_stage_ft": "10"}]).to_csv(overrides, index=False)

    result = validate_config.validate_configs(sites, profiles, overrides)
    messages = "\n".join(result.errors)

    assert "duplicate LID 'AAA1'" in messages
    assert "missing USGS ID" in messages
    assert "profile LID not found" in messages
    assert "custom profile missing recommended_min_crest_stage_ft" in messages
    assert "override LID not found" in messages


def test_validate_config_reports_blank_required_fields_and_suspicious_thresholds(tmp_path: Path) -> None:
    sites = write_csv(
        tmp_path / "sites.csv",
        [{"lid": "AAA1", "name": "", "usgs_site": "01234567", "action_stage_ft": "20", "minor_stage_ft": "10"}],
        ["lid", "name", "usgs_site", "action_stage_ft", "minor_stage_ft"],
    )
    profiles = write_csv(
        tmp_path / "model_profiles.csv",
        [{"lid": "AAA1", "recommended_event_set": "custom", "recommended_min_crest_stage_ft": "0", "reason": ""}],
        ["lid", "recommended_event_set", "recommended_min_crest_stage_ft", "reason"],
    )
    overrides = write_csv(
        tmp_path / "forecast_overrides.csv",
        [{"lid": "AAA1", "min_forecast_stage_ft": "250"}],
        ["lid", "min_forecast_stage_ft"],
    )

    result = validate_config.validate_configs(sites, profiles, overrides)
    messages = "\n".join(result.errors)

    assert "blank required field 'name'" in messages
    assert "blank required field 'reason'" in messages
    assert "suspicious 'recommended_min_crest_stage_ft' value 0" in messages
    assert "suspicious 'min_forecast_stage_ft' value 250" in messages
    assert "minor_stage_ft' (10) is below 'action_stage_ft' (20)" in messages
