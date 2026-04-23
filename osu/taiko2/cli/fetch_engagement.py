"""Fetch playcount / passcount / favourites / user ratings / mapper /
genre / language for every chart in a dataset, from osu! API v2.

Credentials come from the `credentials` module (env vars
``OSU_CLIENT_ID`` / ``OSU_CLIENT_SECRET``, or ``osu/taiko2/.env``, or
``osu/taiko2/secrets.json``).

Usage::

    python -m osu.taiko2.cli.fetch_engagement --dataset taiko2_v1

Writes two sidecar files next to the manifest:

  - ``manifest_engagement.json``   structured, includes raw rating buckets
  - ``manifest_engagement.csv``    flat scalars for quick analysis

Both are keyed by ``beatmap_id`` and do NOT modify the training
manifest. Charts that fail to resolve are simply absent from the
output — re-running the command is safe and idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..fetch.engagement import write_engagement_sidecar


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch popularity / engagement metadata for a dataset.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or dataset root path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ds_root = Path(args.dataset)
    if not ds_root.is_absolute() and not ds_root.exists():
        ds_root = args.datasets_dir / args.dataset
    ds_root = ds_root.resolve()

    manifest_path = ds_root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 2

    json_path, csv_path = write_engagement_sidecar(
        manifest_path, progress=not args.no_progress,
    )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
