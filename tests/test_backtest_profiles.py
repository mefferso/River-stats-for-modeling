from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import backtest_profiles
import nwps_multigage_model as base


def synthetic_training_rows() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for event_num in range(3):
        crest = 12.0 + event_num
        for hour in range(4):
            stage = 8.0 + event_num + hour * 0.6
            r1 = 0.6
            r3 = 0.5 + event_num * 0.02
            r6 = 0.3
            rows.append(
                {
                    "lid": "TEST",
                    "event_id": f"E{event_num}",
                    "datetime": f"2024-01-0{event_num + 1}T0{hour}:00:00Z",
                    "stage_ft": stage,
                    "h0_stage_ft": 8.0 + event_num,
                    "elapsed_hr_since_rise_start": float(hour),
                    "r1_ft_per_hr": r1,
                    "r3_ft_per_hr": r3,
                    "r6_ft_per_hr": r6,
                    "r12_ft_per_hr": 0.2,
                    "momentum_r1_minus_r3": r1 - r3,
                    "momentum_r3_minus_r6": r3 - r6,
                    "stage_above_h0_ft": stage - (8.0 + event_num),
                    "remaining_rise_ft": crest - stage,
                    "observed_crest_stage_ft": crest,
                    "hours_to_crest": float(4 - hour),
                    "crest_time": f"2024-01-0{event_num + 1}T04:00:00Z",
                }
            )
    return pd.DataFrame(rows)


def test_backtest_training_rows_uses_leave_one_event_out() -> None:
    train = synthetic_training_rows()

    rows, split_method = backtest_profiles.backtest_training_rows(train, "TEST", "flood_plus", max_loo_events=10)

    assert split_method == "leave_one_event_out"
    assert len(rows) == len(train)
    assert rows["split_num"].nunique() == train["event_id"].nunique()
    assert {"pred_crest_stage_ft", "error_ft", "abs_error_ft", "is_underforecast"}.issubset(rows.columns)
    assert rows[base.FEATURES].notna().all().all()


def test_backtest_reports_include_requested_outputs(tmp_path: Path) -> None:
    rows, _split_method = backtest_profiles.backtest_training_rows(synthetic_training_rows(), "TEST", "flood_plus")
    summary = backtest_profiles.metric_summary(rows, ["lid", "run_label"])

    reports = backtest_profiles.write_reports(rows, summary, tmp_path)

    expected = {
        "error_by_hours_to_crest_bins",
        "error_by_stage_bins",
        "error_by_rise_rate_bins",
        "event_level_worst_errors",
        "bias_by_gage",
        "underforecast_frequency",
    }
    assert expected.issubset(reports)
    for name in expected:
        report = pd.read_csv(reports[name])
        assert not report.empty, name

    under = pd.read_csv(reports["underforecast_frequency"])
    assert "underforecast_frequency" in under.columns
    assert np.isfinite(under["underforecast_frequency"]).all()
