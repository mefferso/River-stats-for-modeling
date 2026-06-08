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
src/
  nwps_multigage_model.py      # discovery, download, original trainer, forecast/status
  crest_eventset_train.py      # flood/moderate/major event-set trainer
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

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

PowerShell may need:

```powershell
.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Recommended workflow

### 1) Discover USGS IDs

```bash
python src/nwps_multigage_model.py discover-sites --sites config/sites.csv --output config/sites_with_usgs.csv
```

Review `config/sites_with_usgs.csv`. Some LIDs may not auto-map cleanly and need manual USGS IDs.

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

Review those values before trusting them. The threshold discovery is best-effort because NWPS metadata can be annoyingly inconsistent.

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

Major-plus events:

```bash
python src/crest_eventset_train.py train --sites config/sites_with_usgs.csv --event-set major
```

Custom minimum crest stage:

```bash
python src/crest_eventset_train.py train --sites config/sites_with_usgs.csv --event-set custom --min-crest-stage 18
```

Smoke test first site only:

```bash
python src/crest_eventset_train.py train --sites config/sites_with_usgs.csv --event-set flood --limit 1
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

## Forecast manually

The original forecast command still works for models produced by `nwps_multigage_model.py`:

```bash
python src/nwps_multigage_model.py forecast --lid MNLM6 --stage 13.8 --h0 12.3 --r1 1.4 --r3 1.1 --r6 0.8 --r12 0.5 --elapsed 4
```

Event-set forecast wiring will be the next cleanup step after we inspect flood/moderate model summaries.

---

## Check status

```bash
python src/nwps_multigage_model.py status --sites config/sites_with_usgs.csv
```

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
```

Then try full flood-plus:

```text
mode = run-all
years = 15
limit = 0
event_set = flood
```

Then compare moderate-plus:

```text
mode = run-all
years = 15
limit = 0
event_set = moderate
```

The workflow uploads `river-model-output` containing discovered mappings, processed event/training rows, reports, and models.
