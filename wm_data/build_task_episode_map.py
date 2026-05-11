#!/usr/bin/env python3
"""
Scan ``<workspace>/robotwin_generated`` and write ``task_episode_map.json``.

All paths in the JSON are POSIX strings relative to the workspace root
(parent directory of ``RoboTwin``).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _resolve_workspace import robotwin_repo_root, workspace_root


def scan_generated(gen_root: Path) -> dict[str, Any]:
    if not gen_root.is_dir():
        raise FileNotFoundError(f"robotwin_generated not found: {gen_root}")

    tasks: dict[str, list[str]] = {}
    episodes_flat: list[dict[str, Any]] = []

    for task_dir in sorted(gen_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "preview":
            continue
        task = task_dir.name
        eps: list[str] = []
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
                continue
            ep = ep_dir.name
            eps.append(ep)
            meta = ep_dir / "json" / "metadata.json"
            head = ep_dir / "preview" / f"{ep}_head.png"
            rel_dir = Path("robotwin_generated") / task / ep
            episodes_flat.append(
                {
                    "task": task,
                    "episode": ep,
                    "rel_dir": rel_dir.as_posix(),
                    "has_metadata": meta.is_file(),
                    "has_preview_head": head.is_file(),
                }
            )
        if eps:
            tasks[task] = eps

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paths_relative_to": "workspace_root",
        "workspace_contains": ["RoboTwin", "robotwin_generated"],
        "robotwin_generated": "robotwin_generated",
        "task_count": len(tasks),
        "episode_count": len(episodes_flat),
        "tasks": tasks,
        "episodes": episodes_flat,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build task/episode map from robotwin_generated.")
    p.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Override workspace root (default: parent of RoboTwin).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: RoboTwin/task_episode_map.json).",
    )
    args = p.parse_args()

    wm = (args.workspace_root or workspace_root()).resolve()
    gen = wm / "robotwin_generated"
    out_path = (args.out or (robotwin_repo_root() / "task_episode_map.json")).resolve()

    data = scan_generated(gen)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(wm) if str(out_path).startswith(str(wm)) else out_path}")
    print(f"tasks={data['task_count']} episodes={data['episode_count']}")


if __name__ == "__main__":
    main()
