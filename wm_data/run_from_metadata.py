#!/usr/bin/env python3
"""
Run RoboTwin collect_data from one episode ``metadata.json``, then copy hdf5/mp4
next to that metadata under ``<episode>/tra1/``.

Designed to work when **only the RoboTwin repo** is present: pass
``--metadata wm_data/episodes/<task>/<episode>/json/metadata.json`` (paths
produced by ``sync_episodes_into_repo.py``).

From a full monorepo you can still call this file the same way, or use
``worldarena_robotwin_labeler/run_robotwin_from_metadata.py`` as a thin wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def episode_name_to_seed(episode_name: str) -> int:
    m = re.match(r"episode(\d+)$", str(episode_name).strip())
    return int(m.group(1)) if m else 0


def arms_json_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Map metadata ``arms`` to ROBOTWIN_FORCE_ARMS_JSON (skip when task picks arm itself)."""
    arms = metadata.get("arms")
    if arms is None:
        return None
    if isinstance(arms, dict):
        return json.dumps(arms, ensure_ascii=False)
    if not isinstance(arms, str):
        return None
    s = arms.strip()
    if not s or s.lower() in ("none", "未明确指明"):
        return None
    if s in ("left", "right"):
        return json.dumps({"mode": "single", "primary_arm": s}, ensure_ascii=False)
    return json.dumps({"mode": "single", "primary_arm": s}, ensure_ascii=False)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def newest_file(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate files found.")
    return max(paths, key=lambda p: p.stat().st_mtime)


def _default_robotwin_root() -> Path:
    """``RoboTwin/wm_data/this_file.py`` -> ``RoboTwin``."""
    return Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="Run RoboTwin data collection from one metadata.json and copy outputs back.")
    p.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Episode metadata path, e.g. wm_data/episodes/<task>/<episode>/json/metadata.json",
    )
    p.add_argument("--task-config", default="demo_clean", help="RoboTwin task_config name")
    p.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument(
        "--robotwin-root",
        type=Path,
        default=None,
        help="RoboTwin repository root (default: parent of wm_data/).",
    )
    p.add_argument("--skip-run", action="store_true", help="Only copy newest existing output without running collect_data")
    p.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear RoboTwin/data/<task>/<task_config> before run (default clears to force fresh output).",
    )
    p.add_argument(
        "--snapshot-only",
        action="store_true",
        help="After scene init, save one head_camera PNG to <episode>/preview and exit (no hdf5/mp4).",
    )
    p.add_argument(
        "--preview-root",
        type=Path,
        default=None,
        help="Override directory for snapshot PNGs (default: current episode's preview directory).",
    )
    p.add_argument(
        "--snapshot-max-retries",
        type=int,
        default=_env_int("ROBOTWIN_SNAPSHOT_MAX_RETRIES", 32),
        help="When --snapshot-only: max attempts if setup fails (e.g. UnStableError); seed increases each try.",
    )
    p.add_argument(
        "--snapshot-retry-step",
        type=int,
        default=max(1, _env_int("ROBOTWIN_SNAPSHOT_RETRY_STEP", 1)),
        help="When --snapshot-only: increment added to base seed per retry (default 1).",
    )
    args = p.parse_args()

    robotwin_root = (args.robotwin_root or _default_robotwin_root()).resolve()
    metadata_path = args.metadata.resolve()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task = metadata["task_name"]
    episode_name = metadata["episode_name"]
    episode_root = metadata_path.parent.parent
    base_seed = episode_name_to_seed(episode_name)
    used_seed: int | None = None

    out_root = robotwin_root / "data" / task / args.task_config
    hdf5_dir = out_root / "data"
    video_dir = out_root / "video"

    if args.snapshot_only and args.skip_run:
        print("--snapshot-only cannot be used with --skip-run", file=sys.stderr)
        return 1

    snapshot_path: Path | None = None
    if args.snapshot_only:
        if args.preview_root is not None:
            preview_root = Path(args.preview_root).resolve()
        else:
            preview_root = episode_root / "preview"
        snapshot_path = preview_root / f"{episode_name}_head.png"

    env = os.environ.copy()
    if not args.skip_run:
        cmd = ["bash", "collect_data.sh", task, args.task_config, str(args.gpu_id)]
        env["ROBOTWIN_FORCE_TASK_NAME"] = task
        try:
            confirmed = metadata.get("object_binding", {}).get("confirmed_objects", [])
            if confirmed:
                env["ROBOTWIN_FORCE_SLOTS_JSON"] = json.dumps(confirmed, ensure_ascii=False)
                env["ROBOTWIN_FORCE_MODEL_NAME"] = str(confirmed[0]["modelname"])
                env["ROBOTWIN_FORCE_MODEL_ID"] = str(confirmed[0]["model_id"])
        except Exception:
            pass
        arms_json = arms_json_from_metadata(metadata)
        if arms_json:
            env["ROBOTWIN_FORCE_ARMS_JSON"] = arms_json

        if args.snapshot_only and snapshot_path is not None:
            max_r = max(1, int(args.snapshot_max_retries))
            step = max(1, int(args.snapshot_retry_step))
            base_seed = episode_name_to_seed(episode_name)
            last_rc: int | None = None
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(max_r):
                seed = base_seed + attempt * step
                if snapshot_path.exists():
                    try:
                        snapshot_path.unlink()
                    except OSError:
                        pass
                if out_root.exists() and not args.keep_existing:
                    shutil.rmtree(out_root)
                env["ROBOTWIN_SNAPSHOT_ONLY"] = "1"
                env["ROBOTWIN_SNAPSHOT_PATH"] = str(snapshot_path)
                env["ROBOTWIN_SNAPSHOT_SEED"] = str(seed)
                print(f"Running (snapshot {attempt + 1}/{max_r}, seed={seed}):", " ".join(cmd))
                r = subprocess.run(cmd, cwd=str(robotwin_root), env=env, check=False)
                last_rc = r.returncode
                if snapshot_path.is_file():
                    print("Snapshot saved:")
                    print("-", snapshot_path.resolve())
                    print(f"(snapshot seed used: {seed}, try {attempt + 1}/{max_r})")
                    return 0
            print(
                f"Snapshot not created after {max_r} tries (last returncode={last_rc}): {snapshot_path}",
                file=sys.stderr,
            )
            return 1

        if out_root.exists() and not args.keep_existing:
            shutil.rmtree(out_root)
        env.pop("ROBOTWIN_SNAPSHOT_ONLY", None)
        env.pop("ROBOTWIN_SNAPSHOT_PATH", None)
        env.pop("ROBOTWIN_SNAPSHOT_SEED", None)

        def _collect_outputs_ready() -> bool:
            return bool(list(hdf5_dir.glob("episode*.hdf5")) and list(video_dir.glob("episode*.mp4")))

        max_r = max(1, _env_int("ROBOTWIN_VIDEO_MAX_RETRIES", 8))
        step = max(1, _env_int("ROBOTWIN_VIDEO_RETRY_STEP", 1))
        last_rc: int | None = None
        used_seed: int | None = None
        for attempt in range(max_r):
            seed = base_seed + attempt * step
            if out_root.exists() and not args.keep_existing:
                shutil.rmtree(out_root)
            env["ROBOTWIN_FORCE_SEED"] = str(seed)
            print(f"Running (video {attempt + 1}/{max_r}, seed={seed}):", " ".join(cmd))
            r = subprocess.run(cmd, cwd=str(robotwin_root), env=env, check=False)
            last_rc = r.returncode
            if r.returncode == 0 and _collect_outputs_ready():
                used_seed = seed
                break
            if r.returncode != 0:
                print(f"collect_data exit {r.returncode} (seed={seed}), retrying...")
            elif not _collect_outputs_ready():
                print(f"collect_data exit 0 but no hdf5/mp4 (seed={seed}), retrying...")
        else:
            print(
                f"collect_data failed after {max_r} tries (last returncode={last_rc}, "
                f"base_seed={base_seed}, step={step})",
                file=sys.stderr,
            )
            return 1

    if args.snapshot_only:
        if snapshot_path is None or not snapshot_path.is_file():
            print(f"Snapshot not created: {snapshot_path}", file=sys.stderr)
            return 1
        print("Snapshot saved:")
        print("-", snapshot_path.resolve())
        return 0

    hdf5_files = sorted(hdf5_dir.glob("episode*.hdf5"))
    mp4_files = sorted(video_dir.glob("episode*.mp4"))
    if not hdf5_files or not mp4_files:
        print(f"No outputs found under {out_root}", file=sys.stderr)
        return 1

    src_hdf5 = newest_file(hdf5_files)
    src_mp4 = newest_file(mp4_files)

    tra1_dir = episode_root / "tra1"
    (tra1_dir / "data").mkdir(parents=True, exist_ok=True)
    (tra1_dir / "video").mkdir(parents=True, exist_ok=True)

    dst_hdf5 = tra1_dir / "data" / f"{episode_name}.hdf5"
    dst_mp4 = tra1_dir / "video" / f"{episode_name}.mp4"
    shutil.copy2(src_hdf5, dst_hdf5)
    shutil.copy2(src_mp4, dst_mp4)

    log_obj: dict[str, Any] = {
        "task_name": task,
        "episode_name": episode_name,
        "task_config": args.task_config,
        "gpu_id": str(args.gpu_id),
        "source_hdf5": str(src_hdf5),
        "source_mp4": str(src_mp4),
        "copied_hdf5": str(dst_hdf5),
        "copied_mp4": str(dst_mp4),
        "forced_task_name": env.get("ROBOTWIN_FORCE_TASK_NAME", ""),
        "forced_slots_json": env.get("ROBOTWIN_FORCE_SLOTS_JSON", ""),
        "forced_legacy_model_name": env.get("ROBOTWIN_FORCE_MODEL_NAME", ""),
        "forced_legacy_model_id": env.get("ROBOTWIN_FORCE_MODEL_ID", ""),
        "note": "桥接器会在采集进程注入 ROBOTWIN_FORCE_TASK_NAME、ROBOTWIN_FORCE_SLOTS_JSON 与 ROBOTWIN_FORCE_MODEL_ID（兼容旧逻辑）。",
    }
    if used_seed is not None:
        log_obj["collect_seed"] = used_seed
    (tra1_dir / "run_log.json").write_text(json.dumps(log_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if used_seed is not None and used_seed != base_seed:
        seed_info = episode_root / ".seed_info.json"
        seed_info.write_text(
            json.dumps({"seed": used_seed, "base_seed": base_seed, "mode": "video"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("Done.")
    print("Copied:")
    print("-", dst_hdf5)
    print("-", dst_mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
