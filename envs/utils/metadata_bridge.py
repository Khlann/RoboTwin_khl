"""
Parse ROBOTWIN_FORCE_SLOTS_JSON injected by worldarena_robotwin_labeler/run_robotwin_from_metadata.py.

Each element is typically:
  {"slot": "A", "modelname": "100_seal", "model_id": 2, "attrs": {"seal_color": "#aabbcc", ...}}
"""
from __future__ import annotations

import json
import os
from typing import Any


def parse_forced_slot_a(modelname: str) -> tuple[int | None, dict[str, Any] | None]:
    raw = os.getenv("ROBOTWIN_FORCE_SLOTS_JSON", "").strip()
    if not raw:
        return None, None
    try:
        slots = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(slots, list):
        return None, None
    want = (modelname or "").strip()
    model_id: int | None = None
    attrs: dict[str, Any] | None = None
    for it in slots:
        if not isinstance(it, dict):
            continue
        if str(it.get("slot", "")).strip() != "A":
            continue
        mn = str(it.get("modelname", "")).strip()
        if mn != want:
            continue
        try:
            model_id = int(it["model_id"])
        except (TypeError, ValueError, KeyError):
            model_id = None
        a = it.get("attrs")
        if isinstance(a, dict) and a:
            attrs = dict(a)
        else:
            attrs = None
        break
    return model_id, attrs


def parse_blocks_ranking_forced_colors() -> list[tuple[float, float, float]] | None:
    """
    读取槽位 A/B/C 上 attrs.block_color（#rrggbb），用于 blocks_ranking_size 三块 sapien 方块上色。
    必须三条齐全且颜色合法，否则返回 None（保持环境随机色）。
    """
    raw = os.getenv("ROBOTWIN_FORCE_SLOTS_JSON", "").strip()
    if not raw:
        return None
    try:
        slots = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(slots, list):
        return None
    by_slot: dict[str, dict[str, Any]] = {}
    for it in slots:
        if not isinstance(it, dict):
            continue
        sn = str(it.get("slot", "")).strip()
        if sn in ("A", "B", "C"):
            by_slot[sn] = it
    out: list[tuple[float, float, float]] = []
    for sn in ("A", "B", "C"):
        if sn not in by_slot:
            return None
        attrs = by_slot[sn].get("attrs")
        if not isinstance(attrs, dict):
            return None
        hx = attrs.get("block_color")
        rgb = rgb01_from_hex(hx) if isinstance(hx, str) else None
        if rgb is None:
            return None
        out.append(rgb)
    return out


def rgb01_from_hex(s: str | None) -> tuple[float, float, float] | None:
    if not s or not isinstance(s, str):
        return None
    t = s.strip().lstrip("#")
    if len(t) != 6:
        return None
    try:
        r = int(t[0:2], 16) / 255.0
        g = int(t[2:4], 16) / 255.0
        b = int(t[4:6], 16) / 255.0
    except ValueError:
        return None
    return (r, g, b)
