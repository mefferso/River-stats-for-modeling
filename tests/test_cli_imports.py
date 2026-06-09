from __future__ import annotations

import importlib
import subprocess
import sys


SCRIPT_MODULES = [
    "model_core",
    "nwps_multigage_model",
    "crest_eventset_train",
    "filter_sites",
    "model_status",
    "train_model_profiles",
    "forecast_profiles",
    "build_standalone_dashboard",
    "backtest_profiles",
    "validate_config",
]

SCRIPT_PATHS = [
    "src/nwps_multigage_model.py",
    "src/crest_eventset_train.py",
    "src/filter_sites.py",
    "src/model_status.py",
    "src/train_model_profiles.py",
    "src/forecast_profiles.py",
    "src/build_standalone_dashboard.py",
    "src/backtest_profiles.py",
    "src/validate_config.py",
]


def test_scripts_import_without_side_effects() -> None:
    for module_name in SCRIPT_MODULES:
        importlib.import_module(module_name)


def test_script_help_commands_run() -> None:
    for script_path in SCRIPT_PATHS:
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "usage:" in result.stdout.lower()
