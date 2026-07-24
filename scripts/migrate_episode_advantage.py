#!/usr/bin/env python3
"""One-time migration: copy per-episode `advantage`/`intervention` values out
of each episode parquet file's schema metadata and into `meta/episodes.jsonl`.

Background: `advantage`/`intervention` are stored as pyarrow schema-level
metadata (`table.schema.metadata[b'advantage']`), one scalar per episode
parquet file — not a data column, and not present in `episodes.jsonl` or
`info.json` (those only declare the feature's dtype/shape, not its value).

Re-scanning every episode's parquet schema at every `ParquetDataset.__init__`
call (i.e. on every training run, and again per DataLoader worker process) is
wasted, repeated I/O. This script does that scan once, offline, and writes
the values into `meta/episodes.jsonl` (next to the existing `length` field),
so `ParquetDataset` can read them as cheaply as everything else already
loaded from that file.

This is a temporary bridge: the underlying data-collection pipeline should
eventually write `advantage`/`intervention` directly into `episodes.jsonl`
at collection time, making this script unnecessary for newly collected data.

Usage:
    python scripts/migrate_episode_advantage.py <dataset_root> [<dataset_root> ...]

Each <dataset_root> must contain `meta/episodes.jsonl` and
`data/chunk-*/episode_*.parquet` (LeRobot v2.1-style layout, one file per
episode).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _find_episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    """Locate the parquet file for a given episode index.

    LeRobot v2.1 layout: data/chunk-XXX/episode_YYYYYY.parquet. We don't
    assume a fixed chunk size — just search for the matching filename.
    """
    matches = list(dataset_root.glob(
        f'data/chunk-*/episode_{episode_index:06d}.parquet'))
    if not matches:
        raise FileNotFoundError(
            f'No parquet file found for episode {episode_index} under '
            f'{dataset_root}/data/chunk-*/')
    if len(matches) > 1:
        raise RuntimeError(
            f'Multiple parquet files found for episode {episode_index}: '
            f'{matches}')
    return matches[0]


def migrate_dataset(dataset_root: Path) -> None:
    episodes_path = dataset_root / 'meta' / 'episodes.jsonl'
    if not episodes_path.exists():
        raise FileNotFoundError(f'{episodes_path} not found')

    with open(episodes_path, 'r', encoding='utf-8') as f:
        episode_records = [json.loads(line) for line in f if line.strip()]

    migrated = 0
    skipped = []
    for record in episode_records:
        episode_index = record['episode_index']
        try:
            parquet_path = _find_episode_parquet(dataset_root, episode_index)
            schema_metadata = pq.read_schema(parquet_path).metadata or {}
            advantage = schema_metadata.get(b'advantage')
            intervention = schema_metadata.get(b'intervention')
            if advantage is None:
                skipped.append((episode_index, 'no advantage key in schema '
                                'metadata'))
                continue
            record['advantage'] = int(advantage)
            record['intervention'] = (
                int(intervention) if intervention is not None else None)
            migrated += 1
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            skipped.append((episode_index, str(exc)))

    # Atomic replace: write to a temp file in the same directory, then
    # os.replace it over episodes.jsonl. If the process crashes or the disk
    # fills up mid-write, the original file is left untouched instead of
    # being left truncated/corrupted.
    tmp_path = episodes_path.with_suffix(episodes_path.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for record in episode_records:
            f.write(json.dumps(record) + '\n')
    os.replace(tmp_path, episodes_path)

    print(f'{dataset_root}: migrated {migrated}/{len(episode_records)} '
          f'episodes.')
    if skipped:
        print(f'  {len(skipped)} episode(s) skipped (left without '
              f'advantage/intervention fields):')
        for episode_index, reason in skipped:
            print(f'    episode {episode_index}: {reason}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset_root',
        nargs='+',
        type=Path,
        help='One or more dataset root directories to migrate in place.')
    args = parser.parse_args()

    exit_code = 0
    for dataset_root in args.dataset_root:
        try:
            migrate_dataset(dataset_root)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f'{dataset_root}: FAILED - {exc}', file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
