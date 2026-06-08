# LIX multi-gage river crest model builder

Builds station-specific crest/remaining-rise models from historical USGS stage data.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Fill USGS site IDs
The script tries to pull USGS IDs from NWPS metadata, but not every NWS LID maps cleanly. Run:
```bash
python src/nwps_multigage_model.py discover-sites --sites config/sites.csv
```
Then check `config/sites_with_usgs.csv`. Manually fill blanks if needed.

## 2) Download data
```bash
python src/nwps_multigage_model.py download --sites config/sites_with_usgs.csv --years 15
```

## 3) Build events and train models
```bash
python src/nwps_multigage_model.py train --sites config/sites_with_usgs.csv --years 15
```

Outputs land in `output/`.

## 4) Forecast a current situation
```bash
python src/nwps_multigage_model.py forecast --lid MNLM6 --stage 13.8 --r1 1.4 --r3 1.1 --r6 0.8 --r12 0.5 --elapsed 4 --h0 12.3
```

## Notes
- Uses USGS parameter code 00065 = gage height/stage.
- NWPS is used for metadata/crest history where available, not long historical stage time series.
- Model target is remaining rise, so final crest = current stage + predicted remaining rise.
