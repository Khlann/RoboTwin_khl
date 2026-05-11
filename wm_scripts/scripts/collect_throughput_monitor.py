#!/usr/bin/env python3
"""
扫描 robotwin_generated（或 --root）下各 task/episode/tra* 的产出，估算采集吞吐并在终端刷新。

判定「一条 trajectory 已完成」：对应 tra{i} 下同时存在非空 .mp4 与 .hdf5（video/ 与 data/）。

用法示例::
  cd /disk/worldmodel/ziyang/khl_workspace
  python3 scripts/collect_throughput_monitor.py \\
    --root robotwin_generated \\
    --num-trajectories 20 \\
    -i 2
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EPISODE_DIR_RE = re.compile(r"^episode\d+$")

R = "\033[0m"
B = "\033[1m"
DIM = "\033[2m"
GR = "\033[32m"
YL = "\033[33m"
CY = "\033[36m"
MAG = "\033[35m"


def _tra_done(tra_dir: Path) -> bool:
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


def count_episode_slots(ep_dir: Path, num_tra: int) -> int:
    n = 0
    for i in range(1, num_tra + 1):
        td = ep_dir / f"tra{i}"
        if td.is_dir() and _tra_done(td):
            n += 1
    return n


@dataclass
class TaskStats:
    name: str
    episodes: int
    slots_done: int
    target_slots: int

    @property
    def avg_per_ep(self) -> float:
        if self.episodes <= 0:
            return 0.0
        return self.slots_done / self.episodes


def scan_root(root: Path, num_tra: int) -> list[TaskStats]:
    if not root.is_dir():
        return []
    rows: list[TaskStats] = []
    for task_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not task_dir.is_dir():
            continue
        if task_dir.name.startswith("."):
            continue
        eps = 0
        slots = 0
        try:
            for ch in task_dir.iterdir():
                if not ch.is_dir() or not EPISODE_DIR_RE.match(ch.name):
                    continue
                eps += 1
                slots += count_episode_slots(ch, num_tra)
        except OSError:
            continue
        if eps == 0:
            continue
        rows.append(
            TaskStats(
                name=task_dir.name,
                episodes=eps,
                slots_done=slots,
                target_slots=eps * num_tra,
            )
        )
    return rows


def _bar(pct: float, width: int) -> str:
    p = max(0.0, min(100.0, pct))
    n = int(round(p / 100.0 * width))
    n = max(0, min(width, n))
    return GR + "█" * n + DIM + "░" * (width - n) + R


def render(
    root: Path,
    num_tra: int,
    rows: list[TaskStats],
    rate_per_min: float | None,
    eta_s: float | None,
    interval: float,
) -> str:
    tw = shutil.get_terminal_size(fallback=(100, 24)).columns
    tw = max(72, min(140, tw))
    bar_w = max(10, min(24, tw - 62))

    total_ep = sum(r.episodes for r in rows)
    total_done = sum(r.slots_done for r in rows)
    total_target = sum(r.target_slots for r in rows)
    pct = 100.0 * total_done / total_target if total_target > 0 else 0.0
    avg_ep = total_done / total_ep if total_ep > 0 else 0.0

    rate_s = f"{rate_per_min:.1f} tra/min" if rate_per_min is not None else "…"
    if eta_s is None or eta_s <= 0 or rate_per_min is None or rate_per_min < 1e-6:
        eta_s_str = "…"
    else:
        m = int(eta_s // 60)
        s = int(eta_s % 60)
        eta_s_str = f"{m}m{s}s" if m else f"{s}s"

    lines: list[str] = []
    title = f"{B}{MAG}采集吞吐{R}  {DIM}{root}{R}  {DIM}tra 上限={num_tra}{R}  {DIM}刷新 {interval:g}s{R}"
    lines.append(title)
    lines.append(
        f"{B}全局{R}  已完成 {CY}{total_done}{R}/{total_target}  ({pct:.1f}%)  "
        f"{_bar(pct, bar_w)}  {YL}速率 {rate_s}{R}  {DIM}ETA {eta_s_str}{R}"
    )
    lines.append(f"{B}每 ep 平均 tra{R}: {GR}{avg_ep:.2f}{R}  {DIM}(总 tra ÷ episode 数){R}")
    lines.append(DIM + "—" * min(tw - 2, 120) + R)
    prog_head = ("进度".ljust(bar_w))[:bar_w]
    jn_head = f"均/{num_tra}"
    hdr = (
        f"{B}{'TASK':<26} {'EP':>5} {'DONE':>7} {'TARGET':>8} "
        f"{'PCT':>5}  {prog_head}  {jn_head:>7}{R}"
    )
    lines.append(hdr)
    lines.append(DIM + "-" * min(tw - 2, 120) + R)

    key = lambda r: (-r.slots_done / max(1, r.target_slots), r.name.lower())
    for r in sorted(rows, key=key):
        p = 100.0 * r.slots_done / r.target_slots if r.target_slots > 0 else 0.0
        bar = _bar(p, bar_w)
        jn = f"{r.avg_per_ep:.1f}/{num_tra}"
        short = r.name[:26]
        lines.append(
            f"{short:<26} {r.episodes:>5} {r.slots_done:>7} {r.target_slots:>8} "
            f"{p:>4.0f}%  {bar}  {CY}{jn:>7}{R}"
        )

    lines.append(DIM + "—" * min(tw - 2, 120) + R)
    lines.append(DIM + "Ctrl+C 退出" + R)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="RoboTwin 采集目录吞吐监控（终端）")
    p.add_argument("--root", type=Path, default=Path("robotwin_generated"))
    p.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="每个 episode 期望的 tra 条数（tra1..traN）",
    )
    p.add_argument("-i", "--interval", type=float, default=2.0, help="刷新间隔（秒）")
    p.add_argument(
        "--no-clear",
        action="store_true",
        help="不清屏（便于重定向日志）",
    )
    args = p.parse_args()
    root = args.root.resolve()
    num_tra = max(1, int(args.num_trajectories))
    interval = max(0.5, float(args.interval))

    hist: list[tuple[float, int]] = []
    window_s = 120.0

    try:
        while True:
            t0 = time.monotonic()
            rows = scan_root(root, num_tra)
            total_done = sum(r.slots_done for r in rows)
            hist.append((t0, total_done))
            cut = t0 - window_s
            hist = [h for h in hist if h[0] >= cut]
            rate_per_min: float | None = None
            if len(hist) >= 2:
                t_a, d_a = hist[0]
                t_b, d_b = hist[-1]
                dt = t_b - t_a
                if dt > 1e-3:
                    rate_per_min = (d_b - d_a) / dt * 60.0

            total_target = sum(r.target_slots for r in rows)
            eta_s: float | None = None
            if rate_per_min is not None and rate_per_min > 1e-6:
                left = max(0, total_target - total_done)
                eta_s = left / rate_per_min * 60.0

            out = render(root, num_tra, rows, rate_per_min, eta_s, interval)
            if args.no_clear:
                sys.stdout.write("\n" + out + "\n")
            else:
                sys.stdout.write("\033[2J\033[H" + out + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        if not args.no_clear:
            sys.stdout.write("\033[2J\033[H")
        print("已退出。")


if __name__ == "__main__":
    main()
