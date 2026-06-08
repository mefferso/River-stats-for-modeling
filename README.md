# LIX multi-gage river crest model builder

Builds station-specific crest / remaining-rise models from historical USGS stage data for local river forecast points.

This is meant to be an **operational aid**, not official guidance. The whole point is to build something auditable that can answer:

> Given the current stage, recent rate of rise, momentum, and rise-start stage, how much more rise usually remains at this forecast point?

## What it produces

For each river forecast point with a valid USGS gage-height mapping, the script can produce:

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

Those generated files are gitignored so the repo does not become a bloated river-data swamp monster.

---

## Repo structure

```text
config/
  sites.csv                 # LIDs, names, optional USGS IDs and thresholds
src/
  nwps_multigage_model.py   # main CLI
data/
  raw/                      # downloaded USGS stage CSVs
  processed/                # events + training/scored rows
output/
  models/                   # joblib model files
  reports/                  # summary CSV + equation JSONs
.github/workflows/
  python-check.yml
  run-river-model.yml
```

---

## Local setup

From the repo folder:

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

## Basic workflow

### 1) Discover USGS IDs

```bash
python src/nwps_multigage_model.py discover-sites --sites config/sites.csv --output config/sites_with_usgs.csv
```

Then open `config/sites_with_usgs.csv` and review the `usgs_site` column.

This step is best-effort. Some LIDs may not auto-map cleanly from NWPS metadata. Those blanks need manual USGS site IDs. Bad mappings will wreck the model, because garbage in / garbage out is undefeated.

### 2) Download 15 years of stage data

```bash
python src/nwps_multigage_model.py download --sites config/sites_with_usgs.csv --years 15
```

To test only the first site:

```bash
python src/nwps_multigage_model.py download --sites config/sites_with_usgs.csv --years 1 --limit 1
```

### 3) Train station-specific models

```bash
python src/nwps_multigage_model.py train --sites config/sites_with_usgs.csv
```

To test only the first site:

```bash
python src/nwps_multigage_model.py train --sites config/sites_with_usgs.csv --limit 1
```

### 4) Run the whole pipeline

```bash
python src/nwps_multigage_model.py run-all --sites config/sites.csv --years 15
```

Smoke test only first site / one year:

```bash
python src/nwps_multigage_model.py run-all --sites config/sites.csv --years 1 --limit 1
```

---

## Forecast manually

Example using McNeil-style inputs:

```bash
python src/nwps_multigage_model.py forecast --lid MNLM6 --stage 13.8 --h0 12.3 --r1 1.4 --r3 1.1 --r6 0.8 --r12 0.5 --elapsed 4
```

The forecast output includes:

- `pred_crest_stage_ft` = most likely model crest
- `low_reasonable_crest_ft` = likely minus model MAE, not below current stage
- `conservative_crest_ft` = likely plus MAE / 75th percentile underforecast allowance
- `high_conservative_crest_ft` = likely plus 90th percentile underforecast allowance

That gives you something closer to the way you actually think operationally: most likely, conservative, and “don’t get burned by the bastard” range.

---

## Forecast from latest downloaded data

After training a model and downloading fresh-ish data:

```bash
python src/nwps_multigage_model.py forecast-latest --lid MNLM6
```

This infers:

- latest stage
- H0 from the last 72 hours
- elapsed rise time
- R1/R3/R6/R12
- momentum terms

You can change the H0 lookback:

```bash
python src/nwps_multigage_model.py forecast-latest --lid MNLM6 --h0-lookback-hours 96
```

---

## Check status

```bash
python src/nwps_multigage_model.py status --sites config/sites_with_usgs.csv
```

This shows each LID, USGS mapping, raw data availability, raw date range, and whether a model exists.

---

## Event detection knobs

Training uses an event threshold in this order:

1. `event_threshold_ft` from `config/sites.csv`, if filled
2. `action_stage_ft`, if filled
3. `flood_stage_ft`, if filled
4. automatic fallback using upper-stage percentiles

Optional train/run-all knobs:

```bash
--min-total-rise 1.0
--below-hours 24
--h0-lookback-hours 48
--pre-event-hours 24
--sample-interval 1h
```

Examples:

```bash
python src/nwps_multigage_model.py train --sites config/sites_with_usgs.csv --min-total-rise 2.0 --h0-lookback-hours 72
```

```bash
python src/nwps_multigage_model.py run-all --sites config/sites.csv --years 15 --below-hours 36
```

---

## GitHub Actions

Two workflows are included.

### Python check

Runs automatically on push and verifies the script compiles and the CLI loads.

### Run river model

Go to:

```text
Actions → Run river model → Run workflow
```

Modes:

- `smoke`: discover + 1 year download + train first N sites
- `discover`: only create `config/sites_with_usgs.csv`
- `download`: download stage data using `config/sites_with_usgs.csv`
- `train`: train models using existing downloaded data
- `run-all`: discover + download + train
- `status`: print repo/model status

Start with:

```text
mode = smoke
years = 1
limit = 1
```

Then graduate to:

```text
mode = run-all
years = 15
limit = 0
```

The workflow uploads `river-model-output` as an artifact containing discovered mappings, processed data, reports, and models.

---

## Current site list

The initial list includes Amite, Comite, Tickfaw, Natalbany, Tangipahoa, Tchefuncte, Bogue Falaya, Bogue Chitto, Pearl, Hobolochitto, Jourdan, Wolf, Biloxi, Tchoutacabouffa, Pascagoula, and Escatawpa forecast points.

Edit `config/sites.csv` to add/remove stations or to manually set USGS IDs and thresholds.
