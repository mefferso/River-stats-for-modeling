from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_pearl_river_lids_removed_from_normal_config_processing() -> None:
    removed = {"BXAL1", "PERL1"}
    for csv_path in ["config/sites.csv", "config/sites_with_usgs.csv", "config/model_profiles.csv"]:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        assert removed.isdisjoint(set(df["lid"].str.upper())), csv_path

    for dashboard_path in ["docs/latest_forecasts.json", "docs/forecast_dashboard.html"]:
        text = Path(dashboard_path).read_text(encoding="utf-8")
        assert "BXAL1" not in text
        assert "PERL1" not in text


def test_dashboard_hides_skipped_rows_by_default() -> None:
    for dashboard_path in ["docs/index.html", "docs/forecast_dashboard.html"]:
        dashboard = Path(dashboard_path).read_text(encoding="utf-8")
        assert "hideSkippedByDefault" in dashboard
        assert "!status && r.forecast_status === 'skipped'" in dashboard
        assert "!hideSkippedByDefault" in dashboard
