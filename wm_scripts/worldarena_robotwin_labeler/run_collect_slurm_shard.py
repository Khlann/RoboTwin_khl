#!/usr/bin/env python3
"""
Slurm 采集入口：枚举 robotwin_generated 下各 task/episode 的 tra1..traN，
按 rank 分片；可选 --slurm-shard 时在每台机器上起多 GPU 子进程池。

环境变量::
  NUM_TRAJECTORIES  每个 episode 的 tra 条数，默认 20
  DYNAMIC_POOL_GPUS  --slurm-shard 时每节点并行子进程数，默认 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _repo_workspace_root() -> Path:
    """khl_workspace：支持 .../worldarena_robotwin_labeler/ 与 .../wm_scripts/worldarena_robotwin_labeler/。"""
    labeler = Path(__file__).resolve().parent
    if labeler.parent.name == "wm_scripts":
        return labeler.parent.parent.parent
    return labeler.parent


WORKSPACE = _repo_workspace_root()
LABELER = Path(__file__).resolve().parent
RUN_META = LABELER / "run_robotwin_from_metadata.py"

EP_RE = re.compile(r"^episode\d+$")


@dataclass(frozen=True)
class Job:
    task: str
    episode_name: str
    tra_index: int
    metadata: Path
    tra_dir: Path


def _tra_complete(tra_dir: Path) -> bool:
    vdir = tra_dir / "video"
    ddir = tra_dir / "data"
    if not vdir.is_dir() or not ddir.is_dir():
        return False
    mp4s = list(vdir.glob("*.mp4"))
    hdfs = list(ddir.glob("*.hdf5"))
    if not mp4s or not hdfs:
        return False
    try:
        if any(p.stat().st_size < 4096 for p in mp4s + hdfs):
            return False
    except OSError:
        return False
    return True


def enumerate_jobs(root: Path, num_tra: int) -> list[Job]:
    jobs: list[Job] = []
    if not root.is_dir():
        return jobs
    for task_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for ep_dir in sorted(task_dir.iterdir(), key=lambda p: p.name):
            if not ep_dir.is_dir() or not EP_RE.match(ep_dir.name):
                continue
            meta = ep_dir / "json" / "metadata.json"
            if not meta.is_file():
                continue
            for i in range(1, num_tra + 1):
                tra_dir = ep_dir / f"tra{i}"
                jobs.append(
                    Job(
                        task=task_dir.name,
                        episode_name=ep_dir.name,
                        tra_index=i,
                        metadata=meta,
                        tra_dir=tra_dir,
                    )
                )
    return jobs


def _shard_indices(n: int, rank: int, world: int) -> range:
    return range(rank, n, world)


def _append_failure(rec: dict) -> None:
    p = LABELER / "batch_run_failures.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _invoke(meta: Path, tra_dir: Path, task_cfg: str, gpu: int) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(RUN_META),
        "--metadata",
        str(meta),
        "--tra-dir",
        str(tra_dir),
        "--task-config",
        task_cfg,
    ]
    return subprocess.run(cmd, cwd=str(WORKSPACE), env=env).returncode


def _run_parallel(
    jobs: list[Job],
    task_cfg: str,
    max_workers: int,
) -> tuple[int, int]:
    ok, bad = 0, 0
    if max_workers <= 1:
        for j in jobs:
            if _tra_complete(j.tra_dir):
                print(f"[跳过] tra{j.tra_index} 已有输出 {j.episode_name}", flush=True)
                ok += 1
                continue
            rc = _invoke(j.metadata, j.tra_dir, task_cfg, 0)
            if rc == 0:
                ok += 1
            else:
                bad += 1
                _append_failure(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "mode": "slurm_shard",
                        "task": j.task,
                        "episode": j.episode_name,
                        "tra": j.tra_index,
                        "metadata": str(j.metadata),
                        "returncode": rc,
                    }
                )
        return ok, bad

    pending: list[Job] = []
    for j in jobs:
        if _tra_complete(j.tra_dir):
            print(f"[跳过] tra{j.tra_index} 已有输出 {j.episode_name}", flush=True)
            ok += 1
        else:
            pending.append(j)

    gpu_cycle = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for j in pending:
            gpu = gpu_cycle % max_workers
            gpu_cycle += 1
            fut = ex.submit(_invoke, j.metadata, j.tra_dir, task_cfg, gpu)
            futs[fut] = j
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                rc = fut.result()
            except Exception as e:
                bad += 1
                _append_failure(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "mode": "slurm_shard",
                        "task": j.task,
                        "episode": j.episode_name,
                        "tra": j.tra_index,
                        "metadata": str(j.metadata),
                        "returncode": -1,
                        "error": repr(e),
                    }
                )
                continue
            if rc == 0:
                ok += 1
            else:
                bad += 1
                _append_failure(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "mode": "slurm_shard",
                        "task": j.task,
                        "episode": j.episode_name,
                        "tra": j.tra_index,
                        "metadata": str(j.metadata),
                        "returncode": rc,
                    }
                )
    return ok, bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="默认 $COLLECT_ROOT 或 robotwin_generated（相对工作区根目录）",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=os.environ.get("COLLECT_TASK_CONFIG", "demo_clean"),
    )
    parser.add_argument(
        "--slurm-shard",
        action="store_true",
        help="每 Slurm 进程内并行多 GPU（适合每节点 1 task、多卡）",
    )
    args = parser.parse_args()

    root_arg = args.root or Path(os.environ.get("COLLECT_ROOT", "robotwin_generated"))
    root = (WORKSPACE / root_arg).resolve() if not root_arg.is_absolute() else root_arg.resolve()
    num_tra = int(os.environ.get("NUM_TRAJECTORIES", "20"))
    pool = int(os.environ.get("DYNAMIC_POOL_GPUS", "8"))

    all_jobs = enumerate_jobs(root, num_tra)
    total = len(all_jobs)
    if total == 0:
        print(f"[collect] 未找到可采集单元（{root} 下无 episode*/json/metadata.json）", flush=True)
        raise SystemExit(0)

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    if world < 1:
        world = 1

    idxs = list(_shard_indices(total, rank, world))
    my_jobs = [all_jobs[i] for i in idxs]
    host = socket.gethostname()

    print(
        f"[slurm-shard] rank={rank}/{world} HOST={host} 本 rank 工作单元={len(my_jobs)} / 全局 {total}",
        flush=True,
    )

    workers = max(1, pool) if args.slurm_shard else 1
    ok, bad = _run_parallel(my_jobs, args.task_config, workers)
    print(f"[collect] rank={rank} 完成 ok={ok} fail={bad}", flush=True)
    if bad > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
