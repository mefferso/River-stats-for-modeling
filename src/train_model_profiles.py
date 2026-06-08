#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import crest_eventset_train as event_train
import nwps_multigage_model as base


def parse_lids(text: str) -> list[str]:
    return [x.strip().upper() for x in text.replace(';', ',').split(',') if x.strip()]


def read_profiles(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    needed = {'lid', 'recommended_event_set', 'recommended_min_crest_stage_ft', 'reason'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'missing profile columns: {sorted(missing)}')
    df['lid'] = df['lid'].astype(str).str.upper().str.strip()
    df['recommended_event_set'] = df['recommended_event_set'].astype(str).str.lower().str.strip()
    return df


def read_clean_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['stage_ft'] = pd.to_numeric(df.get('stage_ft'), errors='coerce')
    # Drop no-data sentinels and physically impossible junk before event detection/training.
    before = len(df)
    df = df[(df['stage_ft'] > -1000) & (df['stage_ft'] < 200)].copy()
    dropped = before - len(df)
    if dropped:
        print(f'  dropped {dropped:,} impossible stage rows from {path.name}')
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description='Train recommended per-gage model profiles.')
    parser.add_argument('--sites', default=str(base.ROOT / 'config' / 'sites_with_usgs.csv'))
    parser.add_argument('--profiles', default=str(base.ROOT / 'config' / 'model_profiles.csv'))
    parser.add_argument('--lids', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--min-total-rise', type=float, default=1.0)
    parser.add_argument('--below-hours', type=float, default=24.0)
    parser.add_argument('--h0-lookback-hours', type=float, default=48.0)
    parser.add_argument('--pre-event-hours', type=float, default=24.0)
    parser.add_argument('--sample-interval', default='1h')
    args = parser.parse_args()

    base.ensure_dirs()
    sites = event_train.read_sites(args.sites)
    profiles = read_profiles(args.profiles)

    selected = parse_lids(args.lids)
    if selected:
        profiles = profiles[profiles['lid'].isin(selected)].copy()
    if args.limit:
        profiles = profiles.head(args.limit)

    site_by_lid = {str(row['lid']).upper(): row for _, row in sites.iterrows()}
    rows: list[dict[str, object]] = []

    for _, profile in profiles.iterrows():
        lid = str(profile['lid']).upper()
        event_set = str(profile['recommended_event_set']).lower().strip()
        reason = str(profile.get('reason', ''))

        if event_set == 'skip':
            print(f'SKIP {lid}: {reason}')
            rows.append({'lid': lid, 'run_label': 'skip', 'event_set': 'skip', 'status': 'skipped', 'profile_reason': reason})
            continue
        if event_set not in event_train.EVENT_SETS:
            print(f'SKIP {lid}: bad event_set {event_set}')
            rows.append({'lid': lid, 'run_label': event_set, 'event_set': event_set, 'status': 'bad_profile_event_set', 'profile_reason': reason})
            continue
        if lid not in site_by_lid:
            print(f'SKIP {lid}: missing site row')
            rows.append({'lid': lid, 'run_label': event_set, 'event_set': event_set, 'status': 'missing_site_row', 'profile_reason': reason})
            continue

        raw = base.DATA_RAW / f'{lid}_usgs_stage.csv'
        if not raw.exists():
            print(f'SKIP {lid}: missing raw data')
            rows.append({'lid': lid, 'run_label': event_set, 'event_set': event_set, 'status': 'missing_raw', 'profile_reason': reason})
            continue

        min_crest = event_train.as_float(profile.get('recommended_min_crest_stage_ft', ''))
        settings = event_train.EventSetSettings(
            event_set=event_set,
            min_crest_stage=min_crest,
            min_total_rise=args.min_total_rise,
            below_hours=args.below_hours,
            h0_lookback_hours=args.h0_lookback_hours,
            pre_event_hours=args.pre_event_hours,
            sample_interval=args.sample_interval,
        )
        print(f'Training {lid}: {settings.label}')
        result = event_train.train_one(lid, read_clean_raw(raw), site_by_lid[lid], settings)
        result['profile_event_set'] = event_set
        result['profile_min_crest_stage_ft'] = min_crest if min_crest is not None else ''
        result['profile_reason'] = reason
        rows.append(result)
        print(f'  {result}')

    summary = pd.DataFrame(rows)
    out = base.REPORT_DIR / 'model_profile_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    summary.to_csv(base.REPORT_DIR / 'model_summary.csv', index=False)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
