"""Fetch star ratings from osu! API v2 and merge them into a manifest.

Credentials come from the secrets module (env vars `OSU_CLIENT_ID` /
`OSU_CLIENT_SECRET`, or `osu/taiko2/.env`, or `osu/taiko2/secrets.json`).

Usage::

    python -m osu.taiko2.fetch_stars --dataset taiko2_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fetch.stars import update_manifest_stars


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch star ratings for a dataset's charts and rewrite its manifest."
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or path to dataset root.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent / "datasets")
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
        print(f"ERROR: manifest missing: {manifest_path}", file=sys.stderr)
        return 2

    updated = update_manifest_stars(manifest_path, progress=not args.no_progress)
    with_stars = sum(1 for c in updated.charts if c.star_rating is not None)
    print(f"\nDone. {with_stars}/{len(updated.charts)} charts have star_rating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
