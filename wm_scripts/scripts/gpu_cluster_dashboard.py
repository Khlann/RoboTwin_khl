#!/usr/bin/env python3
"""
跨节点 GPU 占用终端看板：SSH 各机执行 nvidia-smi，汇总 SM/显存/温度等。

用法示例：
  python3 scripts/gpu_cluster_dashboard.py 67 3          # 从 Slurm 作业 67 取节点列表，每 3 秒刷新
  python3 scripts/gpu_cluster_dashboard.py -i 2        # 默认节点段（见 --prefix/--from/--to），2 秒刷新
  python3 scripts/gpu_cluster_dashboard.py --hosts a,b -i 3
  python3 scripts/gpu_cluster_dashboard.py --job 67 -i 3

环境变量（可选）：
  GPU_DASHBOARD_HOSTS   逗号分隔主机名，覆盖默认节点列表（在未指定 --job/--hosts 时）
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

# --- ANSI ---
R = "\033[0m"
B = "\033[1m"
DIM = "\033[2m"
RD = "\033[31m"
GR = "\033[32m"
YL = "\033[33m"
CY = "\033[36m"
MAG = "\033[35m"


@dataclass
class GpuRow:
    host: str
    gpu_index: int
    sm_pct: float | None
    mem_used_mib: int | None
    mem_total_mib: int | None
    mem_util_pct: float | None
    temp_c: int | None
    name: str
    error: str | None = None


def _bar(pct: float | None, width: int, filled: str = "█", empty: str = "░") -> str:
    if pct is None or width <= 0:
        return empty * max(0, width)
    p = max(0.0, min(100.0, float(pct)))
    n = int(round(p / 100.0 * width))
    n = max(0, min(width, n))
    return filled * n + empty * (width - n)


def _parse_util_pct(s: str) -> float | None:
    s = s.strip().replace("%", "")
    if not s or s.lower() in ("n/a", "na", "[n/a]"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int_mib(s: str) -> int | None:
    s = s.strip().split()[0] if s.strip() else ""
    if not s or not s.isdigit():
        return None
    return int(s)


def _parse_temp(s: str) -> int | None:
    s = s.strip()
    if not s or s.lower() in ("n/a", "na"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _mem_pct_float(used: int | None, total: int | None) -> float | None:
    if used is None or total is None or total <= 0:
        return None
    return 100.0 * used / total


def _temp_style(t: int | None) -> str:
    if t is None:
        return DIM
    if t >= 85:
        return RD
    if t >= 72:
        return YL
    return GR


def _short_host(h: str, max_len: int = 30) -> str:
    if len(h) <= max_len:
        return h
    return h[: max_len - 1] + "…"


def expand_slurm_nodelist(nodelist: str) -> list[str]:
    nodelist = nodelist.strip()
    if not nodelist:
        return []
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.split()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # 简单逗号切分（无范围展开）
    return [x.strip() for x in nodelist.split(",") if x.strip()]


def hosts_from_slurm_job(job_id: int) -> list[str]:
    for cmd in (
        ["squeue", "-j", str(job_id), "-h", "-o", "%N"],
        ["sacct", "-j", str(job_id), "-n", "-X", "-o", "Nodelist", "-P"],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if out.returncode != 0 or not out.stdout.strip():
            continue
        line = out.stdout.strip().splitlines()[-1].strip()
        hosts = expand_slurm_nodelist(line)
        if hosts:
            return sorted(set(hosts))
    return []


def default_hosts(prefix: str, lo: int, hi: int) -> list[str]:
    return [f"{prefix}-{i}" for i in range(lo, hi + 1)]


def parse_gpu_csv(host: str, text: str) -> list[GpuRow]:
    rows: list[GpuRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        sm = _parse_util_pct(parts[1])
        mem_u = _parse_util_pct(parts[2])
        used = _parse_int_mib(parts[3])
        total = _parse_int_mib(parts[4])
        temp = _parse_temp(parts[5])
        name = parts[6][:40] if len(parts) > 6 else ""
        mem_pct = mem_u if mem_u is not None else _mem_pct_float(used, total)
        rows.append(
            GpuRow(
                host=host,
                gpu_index=idx,
                sm_pct=sm,
                mem_used_mib=used,
                mem_total_mib=total,
                mem_util_pct=mem_pct,
                temp_c=temp,
                name=name,
            )
        )
    return rows


_NV_QUERY = (
    "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,name"
)


def query_host(host: str, timeout: int = 12) -> tuple[str, list[GpuRow] | None, str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=6",
        host,
        "nvidia-smi",
        f"--query-gpu={_NV_QUERY}",
        "--format=csv,noheader",
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return host, None, "timeout"
    except OSError as e:
        return host, None, str(e)
    if out.returncode != 0:
        err = (out.stderr or out.stdout or "nvidia-smi failed").strip()
        err = re.sub(r"\s+", " ", err)[:200]
        return host, None, err
    rows = parse_gpu_csv(host, out.stdout)
    if not rows:
        return host, None, "no gpu rows"
    return host, rows, ""


def collect_rows(hosts: Sequence[str], workers: int) -> list[GpuRow]:
    all_rows: list[GpuRow] = []
    errors: list[tuple[str, str]] = []
    w = max(4, min(workers, len(hosts) or 1))
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = {ex.submit(query_host, h): h for h in hosts}
        for fut in as_completed(futs):
            host, rows, err = fut.result()
            if rows is None:
                errors.append((host, err))
                all_rows.append(
                    GpuRow(
                        host=host,
                        gpu_index=-1,
                        sm_pct=None,
                        mem_used_mib=None,
                        mem_total_mib=None,
                        mem_util_pct=None,
                        temp_c=None,
                        name="",
                        error=err,
                    )
                )
            else:
                all_rows.extend(rows)
    # 稳定顺序：按 host 名、再 gpu index
    def sort_key(r: GpuRow):
        return (r.host, r.gpu_index if r.gpu_index >= 0 else 999)

    all_rows.sort(key=sort_key)
    return all_rows


def render(rows: list[GpuRow], title: str) -> str:
    tw = shutil.get_terminal_size(fallback=(100, 24)).columns
    tw = max(80, min(132, tw))
    sm_w = max(8, min(28, (tw - 72) // 2))
    mem_w = sm_w

    ok = [r for r in rows if r.error is None]
    err_hosts = {r.host for r in rows if r.error is not None}

    def avg(xs: Iterable[float | None]) -> float | None:
        vals = [x for x in xs if x is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    avg_sm = avg(r.sm_pct for r in ok)
    avg_mem = avg(r.mem_util_pct for r in ok)
    n_gpu = len(ok)

    sm_s = f"{avg_sm:.1f}%" if avg_sm is not None else "n/a"
    mem_s = f"{avg_mem:.1f}%" if avg_mem is not None else "n/a"
    summary = (
        f"{B}{CY}GPU{n_gpu}{R}  {DIM}|{R} 平均 {B}SM{R} {sm_s}  {DIM}|{R} 平均 "
        f"{B}MEM{R} {mem_s}  {DIM}|{R} 异常节点 {RD}{len(err_hosts)}{R}/{len({r.host for r in rows})}"
    )

    lines: list[str] = []
    lines.append(f"{B}{MAG}{title}{R}")
    lines.append(summary)
    lines.append(DIM + "—" * min(tw - 2, 120) + R)
    lines.append(
        f"{B}{'NODE':<30} {'#':>2} {'SM':>4} {' ' * sm_w} {'MiB used/total':>16} {' ' * mem_w} {'M%':>5} "
        f"{'℃':>4}  {'NAME':<18}{R}"
    )
    lines.append(DIM + "-" * min(tw - 2, 120) + R)

    for r in rows:
        short = _short_host(r.host, 30)
        if r.error:
            lim = max(1, tw - 55)
            lines.append(f"{RD}{short:<30}  !!  ssh/nvidia-smi 失败  {r.error[:lim]}{R}")
            continue
        sm = r.sm_pct
        mem_pct = r.mem_util_pct
        used, tot = r.mem_used_mib, r.mem_total_mib
        if used is not None and tot is not None:
            mib_s = f"{used}/{tot}"
        else:
            mib_s = "n/a"
        sm_bar = _bar(sm, sm_w)
        mem_bar = _bar(mem_pct, mem_w)
        ts = _temp_style(r.temp_c)
        tc = f"{r.temp_c}" if r.temp_c is not None else "-"
        sm_txt = f"{sm:.0f}" if sm is not None else "-"
        mp_txt = f"{mem_pct:.0f}" if mem_pct is not None else "-"
        nm = (r.name or "")[:18]
        lines.append(
            f"{short:<30} {r.gpu_index:>2} {CY}{sm_txt:>4}{R} {GR}{sm_bar}{R} "
            f"{mib_s:>16} {MAG}{mem_bar}{R} {mp_txt:>5} "
            f"{ts}{tc:>4}{R}  {nm}"
        )

    lines.append(DIM + "—" * min(tw - 2, 120) + R)
    lines.append(DIM + "Ctrl+C 退出" + R)
    return "\n".join(lines)


def parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU cluster SSH dashboard")
    p.add_argument(
        "rest",
        nargs="*",
        help="兼容: <slurm_job_id> <interval> 两个整数",
    )
    p.add_argument("-i", "--interval", type=float, default=3.0, help="刷新秒数")
    p.add_argument("-j", "--job", type=int, default=None, help="Slurm 作业 ID，用于取节点列表")
    p.add_argument("--hosts", type=str, default=None, help="逗号分隔主机名")
    p.add_argument("--prefix", type=str, default="ZJYKP-A100x8-INTEL-114")
    p.add_argument("--from", dest="host_lo", type=int, default=111)
    p.add_argument("--to", dest="host_hi", type=int, default=118)
    p.add_argument("-w", "--workers", type=int, default=32, help="并发 SSH 数")
    ns = p.parse_args(argv)

    r = ns.rest or []
    if len(r) >= 2 and ns.job is None and ns.hosts is None:
        try:
            ns.job = int(r[0])
            ns.interval = float(r[1])
        except ValueError:
            pass
    elif len(r) == 1 and ns.job is None and ns.hosts is None:
        try:
            v = float(r[0])
            if v <= 60:
                ns.interval = v
            else:
                ns.job = int(v)
        except ValueError:
            pass
    return ns


def resolve_hosts(ns: argparse.Namespace) -> list[str]:
    if ns.hosts:
        return sorted({h.strip() for h in ns.hosts.split(",") if h.strip()})
    if ns.job is not None:
        h = hosts_from_slurm_job(ns.job)
        if h:
            return h
        print(f"{RD}无法从 Slurm 获取作业 {ns.job} 的节点列表；改用默认 prefix 段。{R}", file=sys.stderr)
    env_h = os.environ.get("GPU_DASHBOARD_HOSTS", "").strip()
    if env_h and not ns.job:
        return sorted({x.strip() for x in env_h.split(",") if x.strip()})
    slurm_nl = os.environ.get("SLURM_JOB_NODELIST", "").strip()
    if slurm_nl:
        h = expand_slurm_nodelist(slurm_nl)
        if h:
            return sorted(set(h))
    return default_hosts(ns.prefix, ns.host_lo, ns.host_hi)


def main() -> None:
    ns = parse_argv(sys.argv[1:])
    hosts = resolve_hosts(ns)
    if not hosts:
        print("没有可用主机。", file=sys.stderr)
        sys.exit(2)
    interval = max(0.5, float(ns.interval))
    title = f"GPU 集群看板  {DIM}({len(hosts)} 台){R}"
    if ns.job is not None:
        title += f"  {DIM}job {ns.job}{R}"

    try:
        while True:
            rows = collect_rows(hosts, ns.workers)
            out = render(rows, title)
            sys.stdout.write("\033[2J\033[H" + out + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\033[2J\033[H")
        print("已退出。")


if __name__ == "__main__":
    main()
