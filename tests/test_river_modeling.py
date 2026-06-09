from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import crest_eventset_train as event_train
import forecast_profiles
import nwps_multigage_model as base
import train_model_profiles


class ConstantRemainingModel:
    def __init__(self, remaining: float):
        self.remaining = remaining
        self.seen_columns: list[str] | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.seen_columns = list(features.columns)
        return np.array([self.remaining] * len(features), dtype=float)


def stage_series(values: list[float], start: str = "2024-01-01T00:00:00Z", freq: str = "15min") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=len(values), freq=freq, tz="UTC"),
            "stage_ft": values,
        }
    )


def recent_stage_series(values: list[float], freq: str = "h") -> pd.DataFrame:
    end = pd.Timestamp(datetime.now(timezone.utc)).floor(freq)
    start = end - (len(values) - 1) * pd.Timedelta(1, unit=freq)
    return stage_series(values, start=start.isoformat(), freq=freq)


def forecast_args(**overrides: float) -> Namespace:
    defaults = {
        "h0_lookback_hours": 12.0,
        "early_window_ft": 0.5,
        "min_stage_rise_ft": 0.5,
        "min_r3_rise_rate": 0.05,
        "max_data_age_hours": 6.0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_stage_cleaning_drops_sentinel_junk_and_preserves_valid_rows(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=11, freq="h", tz="UTC"),
            "stage_ft": ["", np.nan, -999999, -1000, -1001, "bad", 1.2, 199.9, 200.0, 250.0, 3.4],
        }
    )

    cleaned_base = base.clean_stage_data(raw)
    assert cleaned_base["stage_ft"].tolist() == [1.2, 199.9, 3.4]

    raw_path = tmp_path / "TEST_usgs_stage.csv"
    raw.to_csv(raw_path, index=False)
    cleaned_training = train_model_profiles.read_clean_raw(raw_path)
    assert cleaned_training["stage_ft"].tolist() == [1.2, 199.9, 3.4]

    smoothed = base.prep_stage(raw, freq="1h")
    assert smoothed["stage_ft"].between(-1000, 200, inclusive="neither").all()


def test_event_detection_uses_thresholds_and_filters_by_min_crest() -> None:
    stages = [8.0] * 12 + list(np.linspace(8.0, 12.2, 32)) + list(np.linspace(12.0, 8.0, 40))
    site = {
        "event_start_threshold_ft": "10.0",
        "minor_stage_ft": "11.0",
        "moderate_stage_ft": "13.0",
        "major_stage_ft": "15.0",
    }

    flood_settings = event_train.EventSetSettings(
        event_set="flood",
        min_total_rise=1.0,
        below_hours=1.0,
        h0_lookback_hours=12.0,
        pre_event_hours=1.0,
    )
    events = event_train.detect_events(stage_series(stages), site, flood_settings)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["event_set"] == "flood"
    assert event["event_start_threshold_used_ft"] == pytest.approx(10.0)
    assert event["min_crest_stage_used_ft"] == pytest.approx(11.0)
    assert event["min_crest_stage_source"] == "minor_stage_ft"
    assert event["crest_stage_ft"] > 12.0
    assert event["total_rise_ft"] >= 2.5

    moderate_settings = event_train.EventSetSettings(
        event_set="moderate",
        min_total_rise=1.0,
        below_hours=1.0,
        h0_lookback_hours=12.0,
        pre_event_hours=1.0,
    )
    assert event_train.detect_events(stage_series(stages), site, moderate_settings).empty


def test_rate_of_rise_calculations_use_latest_prior_stage() -> None:
    df = base.prep_stage(
        stage_series(
            [8.0, 8.2, 8.4, 8.6, 8.8],
            start="2024-01-01T00:00:00Z",
            freq="h",
        )
    )
    current_time = pd.Timestamp("2024-01-01T04:00:00Z")
    current_stage = 8.8

    assert forecast_profiles.rate(df, current_time, current_stage, 1) == pytest.approx(0.2)
    assert forecast_profiles.rate(df, current_time, current_stage, 3) == pytest.approx(0.2)
    assert forecast_profiles.rate(df, current_time, current_stage, 12) == 0.0


def test_feature_row_generation_builds_model_inputs_from_synthetic_event() -> None:
    stages = [7.0] * 8 + list(np.linspace(7.0, 13.0, 80)) + list(np.linspace(12.8, 8.0, 20))
    raw = stage_series(stages)
    site = {"event_start_threshold_ft": "9.0", "minor_stage_ft": "10.0"}
    settings = event_train.EventSetSettings(
        event_set="flood",
        min_total_rise=1.0,
        below_hours=1.0,
        h0_lookback_hours=24.0,
        pre_event_hours=4.0,
    )
    events = event_train.detect_events(raw, site, settings)
    rows = base.feature_rows(raw, events, "TEST", sample_interval="1h")

    assert not rows.empty
    assert set(base.FEATURES).issubset(rows.columns)
    assert rows[base.FEATURES + ["remaining_rise_ft"]].notna().all().all()

    sample = rows.iloc[0]
    assert sample["lid"] == "TEST"
    assert sample["remaining_rise_ft"] == pytest.approx(sample["observed_crest_stage_ft"] - sample["stage_ft"])
    assert sample["stage_above_h0_ft"] == pytest.approx(sample["stage_ft"] - sample["h0_stage_ft"])
    assert sample["momentum_r1_minus_r3"] == pytest.approx(sample["r1_ft_per_hr"] - sample["r3_ft_per_hr"])


def test_active_event_flag_distinguishes_active_early_rise_and_inactive() -> None:
    df = base.prep_stage(stage_series([8.0, 8.2, 8.4, 8.8, 9.2, 9.5], freq="h"))
    site = pd.Series({"lid": "TEST", "event_start_threshold_ft": "10.0"})
    args = forecast_args(early_window_ft=0.75, min_stage_rise_ft=1.0, min_r3_rise_rate=0.3)

    at_threshold = forecast_profiles.active_event_flag(site, df, 10.1, 8.0, 0.7, {}, args)
    assert at_threshold == (True, 10.0, "current stage at/above forecast threshold")

    early_rise = forecast_profiles.active_event_flag(site, df, 9.5, 8.0, 0.5, {}, args)
    assert early_rise[0] is True
    assert early_rise[1] == pytest.approx(10.0)
    assert "early rise" in early_rise[2]

    inactive = forecast_profiles.active_event_flag(site, df, 9.0, 8.6, 0.02, {}, args)
    assert inactive[0] is False
    assert inactive[1] == pytest.approx(10.0)
    assert "inactive" in inactive[2]


def forecast_site_and_profile() -> tuple[pd.Series, pd.Series]:
    site = pd.Series({"lid": "TEST", "name": "Synthetic", "usgs_site": "01234567", "event_start_threshold_ft": "10.0"})
    profile = pd.Series({"lid": "TEST", "recommended_event_set": "flood", "recommended_min_crest_stage_ft": "", "reason": "test"})
    return site, profile


def setup_forecast_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: pd.DataFrame,
    remaining: float = 2.25,
) -> tuple[pd.Series, pd.Series]:
    raw_dir = tmp_path / "raw"
    model_dir = tmp_path / "models"
    raw_dir.mkdir()
    model_dir.mkdir()
    monkeypatch.setattr(base, "DATA_RAW", raw_dir)
    monkeypatch.setattr(base, "MODEL_DIR", model_dir)
    raw.to_csv(raw_dir / "TEST_usgs_stage.csv", index=False)
    joblib.dump({"model": ConstantRemainingModel(remaining), "metadata": {"skill": {}}}, model_dir / "TEST_flood_plus_ridge_model.joblib")
    return forecast_site_and_profile()


def test_forecast_one_returns_inactive_without_prediction_and_ok_when_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, profile = setup_forecast_files(tmp_path, monkeypatch, recent_stage_series([8.0, 8.1, 8.2, 8.3, 8.4, 8.5]))

    inactive = forecast_profiles.forecast_one(site, profile, {}, {}, forecast_args())
    assert inactive["forecast_status"] == "inactive"
    assert inactive["confidence"] == "none"
    assert "pred_remaining_rise_ft" not in inactive

    active_raw = recent_stage_series([8.0, 8.4, 8.8, 9.2, 9.6, 10.1, 250.0])
    active_raw.to_csv(base.DATA_RAW / "TEST_usgs_stage.csv", index=False)
    active = forecast_profiles.forecast_one(
        site,
        profile,
        {"event_count": 10, "holdout_events": 3, "mae_ft": 0.8, "bias_ft": 0.1, "r2": 0.7},
        {"max_remaining_rise_ft": "2.0"},
        forecast_args(),
    )

    assert active["forecast_status"] == "ok"
    assert active["current_stage_ft"] < 11.0
    assert active["data_age_hours"] <= 6.0
    assert active["confidence"] == "high"
    assert active["pred_remaining_rise_ft"] == pytest.approx(2.0)
    assert active["pred_crest_likely_ft"] == pytest.approx(active["current_stage_ft"] + 2.0)
    assert "capped" in active["forecast_note"]


def test_stale_data_becomes_stale_data_without_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, profile = setup_forecast_files(tmp_path, monkeypatch, stage_series([9.0, 9.5, 10.2, 10.4], start="2024-01-01T00:00:00Z", freq="h"))

    stale = forecast_profiles.forecast_one(site, profile, {}, {}, forecast_args(max_data_age_hours=6.0))

    assert stale["forecast_status"] == "stale_data"
    assert stale["confidence"] == "none"
    assert stale["forecast_note"] == "latest observation is too old for forecast use"
    assert stale["data_age_hours"] > 6.0
    assert "pred_crest_likely_ft" not in stale
    assert "pred_crest_conservative_ft" not in stale
    assert "pred_crest_high_end_ft" not in stale


def test_fresh_data_can_still_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, profile = setup_forecast_files(tmp_path, monkeypatch, recent_stage_series([8.0, 8.6, 9.4, 10.2]), remaining=1.5)

    fresh = forecast_profiles.forecast_one(
        site,
        profile,
        {"event_count": 10, "holdout_events": 3, "mae_ft": 0.8, "bias_ft": 0.1, "r2": 0.7},
        {},
        forecast_args(max_data_age_hours=6.0),
    )

    assert fresh["forecast_status"] == "ok"
    assert fresh["data_age_hours"] <= 6.0
    assert fresh["pred_crest_likely_ft"] == pytest.approx(fresh["current_stage_ft"] + 1.5)


def test_above_threshold_but_falling_becomes_receding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, profile = setup_forecast_files(tmp_path, monkeypatch, recent_stage_series([11.0, 10.8, 10.5, 10.2]), remaining=1.5)

    receding = forecast_profiles.forecast_one(site, profile, {}, {}, forecast_args(max_data_age_hours=6.0))

    assert receding["forecast_status"] == "receding"
    assert receding["confidence"] == "none"
    assert receding["r3_ft_per_hr"] <= -0.05
    assert receding["forecast_note"] == "current stage is at/above threshold but falling; rising crest forecast suppressed."
    assert "pred_crest_likely_ft" not in receding
    assert "pred_crest_conservative_ft" not in receding
    assert "pred_crest_high_end_ft" not in receding


def test_at_threshold_flat_tiny_remaining_becomes_cresting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, profile = setup_forecast_files(tmp_path, monkeypatch, recent_stage_series([10.2, 10.2, 10.2, 10.2]), remaining=0.05)

    cresting = forecast_profiles.forecast_one(
        site,
        profile,
        {"event_count": 10, "holdout_events": 3, "mae_ft": 0.8, "bias_ft": 0.1, "r2": 0.7},
        {},
        forecast_args(max_data_age_hours=6.0),
    )

    assert cresting["forecast_status"] == "cresting"
    assert cresting["confidence"] == "high"
    assert cresting["r1_ft_per_hr"] <= 0.02
    assert cresting["r3_ft_per_hr"] <= 0.02
    assert cresting["pred_remaining_rise_ft"] == 0.0
    assert cresting["pred_crest_likely_ft"] == pytest.approx(cresting["current_stage_ft"])
    assert cresting["pred_crest_conservative_ft"] == pytest.approx(cresting["current_stage_ft"])
    assert cresting["pred_crest_high_end_ft"] == pytest.approx(cresting["current_stage_ft"] + 0.8)
    assert cresting["forecast_note"] == "current stage is near/at crest with little remaining rise expected"


def test_above_threshold_and_rising_can_still_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, profile = setup_forecast_files(tmp_path, monkeypatch, recent_stage_series([9.0, 9.4, 9.8, 10.2]), remaining=1.5)

    rising = forecast_profiles.forecast_one(
        site,
        profile,
        {"event_count": 10, "holdout_events": 3, "mae_ft": 0.8, "bias_ft": 0.1, "r2": 0.7},
        {},
        forecast_args(max_data_age_hours=6.0),
    )

    assert rising["forecast_status"] == "ok"
    assert rising["r3_ft_per_hr"] > -0.05
    assert rising["pred_crest_likely_ft"] == pytest.approx(rising["current_stage_ft"] + 1.5)


def test_confidence_bucket_assignment() -> None:
    assert forecast_profiles.confidence_bucket("inactive", {}) == "none"
    assert forecast_profiles.confidence_bucket("ok", {"event_count": 4}) == "none"
    assert forecast_profiles.confidence_bucket("ok", {"event_count": 7, "holdout_events": 2, "mae_ft": 0.9}) == "low"
    assert forecast_profiles.confidence_bucket(
        "ok", {"event_count": 9, "holdout_events": 2, "mae_ft": 0.9, "bias_ft": 0.5, "r2": 0.6}
    ) == "high"
    assert forecast_profiles.confidence_bucket(
        "ok", {"event_count": 9, "holdout_events": 2, "mae_ft": 1.4, "bias_ft": 1.0, "r2": 0.2}
    ) == "medium"
    assert forecast_profiles.confidence_bucket(
        "ok", {"event_count": 9, "holdout_events": 2, "mae_ft": 2.0, "bias_ft": 2.0, "r2": 0.2}
    ) == "low"


def test_model_profile_parsing_normalizes_lids_and_labels_and_validates_columns(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.csv"
    pd.DataFrame(
        [
            {"lid": " test ", "recommended_event_set": " Flood ", "recommended_min_crest_stage_ft": "", "reason": "ok"},
            {"lid": "custom", "recommended_event_set": "Custom", "recommended_min_crest_stage_ft": "12.5", "reason": "manual"},
        ]
    ).to_csv(profiles_path, index=False)

    profiles = train_model_profiles.read_profiles(profiles_path)
    assert profiles["lid"].tolist() == ["TEST", "CUSTOM"]
    assert profiles["recommended_event_set"].tolist() == ["flood", "custom"]

    event_set, min_crest, label = forecast_profiles.profile_label(profiles.iloc[1])
    assert event_set == "custom"
    assert min_crest == pytest.approx(12.5)
    assert label == "custom_12_5ft_plus"
    assert train_model_profiles.parse_lids("test; abc, def") == ["TEST", "ABC", "DEF"]

    bad_path = tmp_path / "bad_profiles.csv"
    pd.DataFrame([{"lid": "TEST"}]).to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="missing profile columns"):
        train_model_profiles.read_profiles(bad_path)
