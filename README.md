# LIX multi-gage river crest model builder

Builds station-specific crest / remaining-rise models from historical USGS stage data for local river forecast points.

This is meant to be an **operational aid**, not official guidance. The core question is:

> Given the current stage, recent rate of rise, momentum, and rise-start stage, how much more rise usually remains at this forecast point?

## What it produces

For each forecast point with a valid USGS gage-height mapping, the toolkit can produce:

- detected historical rise/crest events
- training rows sampled through each event
- station-specific Ridge regression model
- readable equation JSON
- model skill summary with MAE/RMSE/bias/R²
- likely/conservative/high-conservative crest estimates

Generated output lands in:

```text
data/processed/
output/models/
output/reports/
```

Raw USGS CSVs land in:

```text
data/raw/
```

Generated data/model files are gitignored so the repo does not become a bloated river-data swamp monster.

---

## Repo structure

```text
config/
  sites.csv                    # LIDs, USGS IDs, optional thresholds
  sites_with_usgs.csv          # discovered LID-to-USGS mapping
  model_profiles.csv           # recommended event-set choice per gage
src/
  model_core.py                # shared stage cleaning/prep, event, feature, and scoring helpers
  nwps_multigage_model.py      # discovery, download, original trainer, forecast/status
  crest_eventset_train.py      # flood/moderate/major/custom event-set trainer
  train_model_profiles.py      # trains recommended model profile per gage
  forecast_profiles.py         # generates latest forecasts from recommended profile models
  build_standalone_dashboard.py # writes the static dashboard files under docs/
  backtest_profiles.py         # evaluation-only profile-model backtests and CSV reports
  validate_config.py           # validates config CSVs before runs
  filter_sites.py              # filters sites to selected LIDs
  model_status.py              # enhanced status report
data/
  raw/                         # downloaded USGS stage CSVs
  processed/                   # events + training/scored rows
output/
  models/                      # joblib model files
  reports/                     # summary CSV + equation JSONs
.github/workflows/
  python-check.yml
  run-river-model.yml
```

---

## Recommended local workflow

### 0) Validate config CSVs

Before downloading/training, validate the config files for duplicate LIDs, missing required fields, profile/override LIDs not present in `sites.csv`, missing USGS IDs for active profiles, missing custom crest thresholds, and suspicious threshold values:

```bash
python src/validate_config.py
```

### 1) Discover USGS IDs

```bash
python src/nwps_multigage_model.py discover-sites --sites config/sites.csv --output config/sites_with_usgs.csv
```

### 2) Try to fill flood-category thresholds

```bash
python src/crest_eventset_train.py discover-thresholds --sites config/sites_with_usgs.csv --output config/sites_with_usgs.csv
```

This attempts to fill:

```text
action_stage_ft
minor_stage_ft
flood_stage_ft
moderate_stage_ft
major_stage_ft
```

Review those values before trusting them. Threshold discovery is best-effort because NWPS metadata can be annoyingly inconsistent.

### 3) Download 15 years of USGS stage data

```bash
python src/nwps_multigage_model.py download --sites config/sites_with_usgs.csv --years 15
```

Smoke test first site only:

```bash
python src/nwps_multigage_model.py download --sites config/sites_with_usgs.csv --years 1 --limit 1
```

### 4) Train event-set models

Flood/minor-plus events:

```bash
python src/crest_eventset_train.py train --sites config/sites_with_usgs.csv --event-set flood
```

Moderate-plus events:

```bash
python src/crest_eventset_train.py train --sites config/sites_with_usgs.csv --event-set moderate
```

Custom minimum crest stage:

```bash
python src/crest_eventset_train.py train --sites config/sites_with_usgs.csv --event-set custom --min-crest-stage 18
```

### 5) Train recommended profile models

After comparing flood/moderate/custom performance, use the profile trainer:

```bash
python src/train_model_profiles.py --sites config/sites_with_usgs.csv --profiles config/model_profiles.csv
```

Selected sparse gages only:

```bash
python src/train_model_profiles.py --sites config/sites_with_usgs.csv --profiles config/model_profiles.csv --lids DARL1,MAGL1,OLVL1,AMIL1,ORAM6
```

---

## How event selection works

The event-set trainer separates two concepts:

| Concept | Purpose |
|---|---|
| Event-start threshold | Finds the event window and captures the rising limb |
| Minimum crest stage | Decides whether the event is worth training on |

Event windows start using the first available threshold in this order:

1. `event_start_threshold_ft`
2. `event_threshold_ft`
3. `action_stage_ft`
4. `minor_stage_ft`
5. `flood_stage_ft`
6. automatic upper-percentile fallback

Training inclusion then depends on the event set:

| `--event-set` | Keeps events where crest reaches... |
|---|---|
| `flood` | `minor_stage_ft`, falling back to `flood_stage_ft` |
| `moderate` | `moderate_stage_ft` |
| `major` | `major_stage_ft` |
| `custom` | `--min-crest-stage` |
| `all` | no crest-stage filter beyond minimum total rise |

This prevents the model from learning from every little 1-foot wiggle while still capturing early rising-limb behavior before the crest.

---

## Model profiles

`config/model_profiles.csv` stores the recommended model strategy per gage.

Examples:

```text
DARL1 → custom, min crest 16 ft
MAGL1 → custom, min crest 47 ft
CREM6 → moderate-plus
LYMM6 → moderate-plus
ROBL1 → flood-plus
BYML1 → skip until USGS/raw mapping exists
```

The profile trainer writes:

```text
output/reports/model_profile_summary.csv
output/reports/model_summary.csv
```

### Backtest profile models

Backtesting is **evaluation/reporting only**. It trains temporary fold models using event-grouped splits (leave-one-event-out when practical) and does not replace operational model files. Reports are written under `output/reports/backtests/`:

```bash
python src/backtest_profiles.py --sites config/sites_with_usgs.csv --profiles config/model_profiles.csv
```

The generated CSVs include error by hours-to-crest bins, stage bins, rise-rate bins, event-level worst errors, bias by gage, and underforecast frequency.

---

## Check status

```bash
python src/model_status.py --sites config/sites_with_usgs.csv
```

This reports raw data range, latest stage, max stage, and model labels found for each LID.

---

## GitHub Actions

Go to:

```text
Actions → Run river model → Run workflow
```

Recommended smoke test:

```text
mode = smoke
years = 1
limit = 1
event_set = flood
selected_lids = blank
```

### Operational forecast modes

`profile-forecast` is the reliable full operational mode for a fresh GitHub Actions runner. It discovers sites, refreshes thresholds, downloads stage data, trains recommended profile models, generates forecasts, builds the dashboard, and runs status:

```text
mode = profile-forecast
years = 15
limit = 0
selected_lids = blank
```

`profile-run` is training-only for recommended profile models. It discovers sites, refreshes thresholds, downloads data, trains profile models, and runs status, but it does **not** generate forecasts or build/publish the dashboard:

```text
mode = profile-run
years = 15
limit = 0
selected_lids = blank
```

`forecast-only` only runs the forecast/dashboard step against files that already exist in the runner workspace. It requires generated `data/raw/*_usgs_stage.csv` files and trained `output/models/*_ridge_model.joblib` files. It will fail on a fresh GitHub Actions runner; use `profile-forecast` for a fresh full operational forecast run.

```text
mode = forecast-only
# only when data/raw and output/models already exist in the runner
```

### Backtesting mode

`backtest-profiles` runs the full profile preparation/training path and then writes evaluation reports under `output/reports/backtests/`:

```text
mode = backtest-profiles
years = 15
limit = 0
selected_lids = blank
```

### Other useful modes

Full flood-plus event-set training:

```text
mode = run-all
years = 15
limit = 0
event_set = flood
selected_lids = blank
```

Selected sparse-gage custom/profile test:

```text
mode = profile-run
years = 15
limit = 0
selected_lids = DARL1,MAGL1,OLVL1,AMIL1,ORAM6
```

The workflow uploads `river-model-output` containing discovered mappings, selected active sites, processed event/training rows, reports (including `output/reports/backtests/`), dashboard docs, and models.
