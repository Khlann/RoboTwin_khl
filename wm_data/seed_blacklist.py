"""Persist and reuse simulation seeds that failed (e.g. UnStableError)."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_bad_seeds_path(save_path: str | Path) -> Path:
    custom = os.environ.get("ROBOTWIN_BAD_SEEDS_FILE", "").strip()
    if custom:
        return Path(custom).resolve()
    return Path(save_path).resolve() / "bad_seeds.txt"


def load_bad_seeds(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    out: set[int] = set()
    for tok in text.split():
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return out


def write_bad_seeds(path: Path, seeds: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(seeds)
    payload = " ".join(str(s) for s in ordered)
    if ordered:
        payload += " "
    path.write_text(payload, encoding="utf-8")


def record_bad_seed(path: Path, seed: int) -> bool:
    bad = load_bad_seeds(path)
    if seed in bad:
        return False
    bad.add(seed)
    write_bad_seeds(path, bad)
    return True


def next_seed_skipping_bad(
    start: int,
    bad: set[int],
    *,
    step: int = 1,
    max_scan: int = 10_000,
) -> int | None:
    seed = start
    for _ in range(max_scan):
        if seed not in bad:
            return seed
        seed += step
    return None
