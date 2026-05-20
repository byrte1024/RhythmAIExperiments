"""Find charts in a dataset by song name.

Usage::

    osu/taiko2/.venv/bin/python -m osu.taiko2.cli.find_chart \
        --dataset taiko2_v1 \
        --query "zombie remix"

Searches artist, title, difficulty version, and chart_id fields.
Shows exact matches first, then ranked fuzzy matches.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _word_overlap(query_tokens: set[str], target: str) -> float:
    target_tokens = _tokenize(target)
    if not query_tokens or not target_tokens:
        return 0.0
    return len(query_tokens & target_tokens) / len(query_tokens)


def _subsequence_match(query: str, target: str) -> bool:
    """True if all characters of query appear in order in target."""
    q = query.lower()
    t = target.lower()
    qi = 0
    for c in t:
        if qi < len(q) and c == q[qi]:
            qi += 1
    return qi == len(q)


def _contains_score(query: str, target: str) -> float:
    """1.0 if query is a substring, else 0."""
    return 1.0 if query.lower() in target.lower() else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Find charts in a dataset manifest by song name.",
    )
    p.add_argument("--dataset", required=True, help="Dataset name.")
    p.add_argument(
        "--datasets-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets",
    )
    p.add_argument("--query", required=True, help="Search query.")
    p.add_argument(
        "--top", type=int, default=20,
        help="Max fuzzy results to show (default 20).",
    )
    args = p.parse_args(argv)

    manifest_path = args.datasets_dir / args.dataset / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        return 2

    from ..persistence.manifest import load_manifest
    manifest = load_manifest(manifest_path)
    charts = manifest.charts
    print(f"Loaded {len(charts):,} charts from {args.dataset}")
    print()

    query = args.query
    query_tokens = _tokenize(query)
    query_lower = query.lower()

    # Build searchable text per chart.
    entries: list[tuple[object, str]] = []
    for c in charts:
        text = f"{c.artist} {c.title} {c.difficulty_version} {c.chart_id}"
        entries.append((c, text))

    # 1. Exact substring matches.
    exact = [(c, t) for c, t in entries if query_lower in t.lower()]

    # 2. Score everything.
    scored: list[tuple[float, float, float, bool, object, str]] = []
    for c, text in entries:
        sub = _contains_score(query, text)
        word = _word_overlap(query_tokens, text)
        subseq = 1.0 if _subsequence_match(query, text) else 0.0
        # Composite: substring > word overlap > subsequence
        score = sub * 3.0 + word * 2.0 + subseq * 0.5
        if score > 0:
            scored.append((score, word, sub, subseq > 0, c, text))

    scored.sort(key=lambda x: -x[0])

    # Group by beatmapset_id (same song, different difficulties).
    from collections import defaultdict

    def _group_by_set(
        items: list[tuple],
    ) -> list[tuple[str, list]]:
        groups: dict[str, list] = defaultdict(list)
        order: list[str] = []
        for item in items:
            c = item[-2] if len(item) > 2 else item[0]
            sid = c.beatmapset_id
            if sid not in groups:
                order.append(sid)
            groups[sid].append(item)
        return [(sid, groups[sid]) for sid in order]

    # Print results.
    if exact:
        exact_groups = _group_by_set([(c, t) for c, t in exact])
        print(f"=== Exact substring matches ({len(exact)} charts, {len(exact_groups)} songs) ===")
        print()
        for _, group in exact_groups[:args.top]:
            c0 = group[0][0]
            diffs = [g[0].difficulty_version for g in group]
            stars = [g[0].star_rating for g in group if g[0].star_rating]
            _print_song(c0, diffs, stars)
        if len(exact_groups) > args.top:
            print(f"  ... and {len(exact_groups) - args.top} more songs")
    else:
        print("No exact substring matches.")

    print()
    print(f"=== Top {args.top} fuzzy matches (by song) ===")
    print()
    scored_groups = _group_by_set(scored)
    shown = 0
    for _, group in scored_groups:
        best = group[0]
        score, word_pct, sub, subseq, c, text = best
        tags = []
        if sub > 0:
            tags.append("substring")
        if word_pct >= 1.0:
            tags.append("all-words")
        elif word_pct > 0:
            tags.append(f"{word_pct:.0%}-words")
        if subseq:
            tags.append("subseq")
        tag_str = ", ".join(tags)
        diffs = [g[-2].difficulty_version for g in group]
        stars = [g[-2].star_rating for g in group if g[-2].star_rating]
        print(f"  [{score:.2f}] {tag_str}")
        _print_song(c, diffs, stars, indent=4)
        shown += 1
        if shown >= args.top:
            break

    if not scored:
        print("  No matches found.")

    return 0


def _print_song(
    c: object,
    diffs: list[str],
    stars: list[float | None],
    indent: int = 2,
) -> None:
    pad = " " * indent
    star_range = ""
    valid_stars = [s for s in stars if s]
    if valid_stars:
        lo, hi = min(valid_stars), max(valid_stars)
        star_range = f" [{lo:.1f}-{hi:.1f}*]" if lo != hi else f" [{lo:.1f}*]"
    print(f"{pad}{c.artist} - {c.title}{star_range}")
    diff_str = ", ".join(diffs)
    print(f"{pad}  set={c.beatmapset_id}  diffs({len(diffs)}): {diff_str}")


if __name__ == "__main__":
    raise SystemExit(main())
