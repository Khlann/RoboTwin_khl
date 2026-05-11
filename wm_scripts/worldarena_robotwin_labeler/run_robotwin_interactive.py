#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_FAILURE_JSONL = Path(__file__).resolve().parent / "batch_run_failures.jsonl"
_LAST_SUMMARY_JSON = Path(__file__).resolve().parent / "batch_run_last_summary.json"


def _failure_record(mode: str, metadata: Path, returncode: int) -> dict[str, str | int]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "task": metadata.parent.parent.parent.name,
        "episode": metadata.parent.parent.name,
        "metadata": str(metadata.resolve()),
        "returncode": returncode,
    }


def _append_failure_jsonl(rec: dict[str, str | int]) -> None:
    _FAILURE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _FAILURE_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_last_summary(obj: dict) -> None:
    _LAST_SUMMARY_JSON.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def episode_sort_key(path: Path) -> tuple[int, str]:
    # .../<task>/episode77/json/metadata.json
    name = path.parent.parent.name
    m = re.match(r"episode(\d+)$", name)
    return (int(m.group(1)) if m else 10**9, name)


def discover_metadata(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in root.glob("*/*/json/metadata.json"):
        task = p.parent.parent.parent.name
        out.setdefault(task, []).append(p)
    for t in out:
        out[t] = sorted(out[t], key=episode_sort_key)
    return dict(sorted(out.items()))


def pick_first_episode_per_task(task_map: dict[str, list[Path]]) -> list[Path]:
    first_items: list[Path] = []
    for task, items in task_map.items():
        if not items:
            continue
        # items already sorted by episode number
        first_items.append(items[0])
    return first_items


def _runner_script() -> Path:
    return Path(__file__).resolve().parent / "run_robotwin_from_metadata.py"


def _episode_name_from_metadata(metadata: Path) -> str:
    """Prefer metadata.json episode_name so copied filenames match run_from_metadata."""
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
        name = data.get("episode_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return metadata.parent.parent.name


def _video_outputs_complete(metadata: Path) -> bool:
    """True when full collect+copy already done (same paths as run_from_metadata)."""
    ep = _episode_name_from_metadata(metadata)
    root = metadata.parent.parent
    mp4 = root / "tra1" / "video" / f"{ep}.mp4"
    hdf5 = root / "tra1" / "data" / f"{ep}.hdf5"
    return mp4.is_file() and hdf5.is_file()


def _snapshot_output_exists(metadata: Path) -> bool:
    ep = _episode_name_from_metadata(metadata)
    root = metadata.parent.parent
    png = root / "preview" / f"{ep}_head.png"
    return png.is_file()


def run_one(metadata: Path, task_config: str, gpu_id: str, skip_run: bool, snapshot_only: bool = False) -> int:
    cmd = [
        sys.executable,
        str(_runner_script()),
        "--metadata",
        str(metadata),
        "--task-config",
        task_config,
        "--gpu-id",
        gpu_id,
    ]
    if skip_run:
        cmd.append("--skip-run")
    if snapshot_only:
        cmd.append("--snapshot-only")
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd)


def ask_choice(prompt: str, options: list[str]) -> int:
    print(prompt)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        val = input("请选择编号: ").strip()
        if val.isdigit() and 1 <= int(val) <= len(options):
            return int(val) - 1
        print("输入无效，请重新输入。")


def normalize_episode_dir_name(raw: str) -> str | None:
    """Accept '602', '077', 'episode602' -> folder name 'episode602'."""
    t = raw.strip().lower()
    if not t:
        return None
    if t.startswith("episode"):
        t = t[7:].lstrip("_")
    if not t.isdigit():
        return None
    return f"episode{int(t)}"


def parse_episode_dir_names(line: str) -> list[str]:
    """
    One line with multiple episodes: split on whitespace and common separators.
    Examples: "730 779", "730,779", "episode730 episode779", "730，779;802"
    Order preserved, duplicates removed.
    """
    s = line.strip()
    if not s:
        return []
    for ch in (",", "，", ";", "；"):
        s = s.replace(ch, " ")
    out: list[str] = []
    seen: set[str] = set()
    for tok in s.split():
        ep = normalize_episode_dir_name(tok)
        if ep is None:
            continue
        if ep in seen:
            continue
        seen.add(ep)
        out.append(ep)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive batch runner for run_robotwin_from_metadata.py")
    p.add_argument("--root", type=Path, default=Path("/home/arlen-n/workspace/cy/wm/robotwin_generated"))
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--skip-run", action="store_true")
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="批量模式下即使已有 tra1 或 preview 输出也重新跑（默认：已有完整输出则跳过）。",
    )
    p.add_argument("--list-only", action="store_true", help="Only list discovered metadata and exit.")
    args = p.parse_args()
    skip_existing = not args.no_skip_existing

    task_map = discover_metadata(args.root)
    if not task_map:
        print(f"未发现 metadata: {args.root}")
        return

    if args.list_only:
        print(f"发现任务数: {len(task_map)}")
        for t, items in task_map.items():
            print(f"- {t}: {len(items)}")
        return

    print(f"root={args.root}")
    print(
        f"task_config={args.task_config}, gpu_id={args.gpu_id}, skip_run={args.skip_run}, "
        f"skip_existing_outputs={skip_existing}"
    )
    mode_idx = ask_choice(
        "选择运行模式",
        [
            "单条 metadata",
            "单任务 batch（该任务下全部 episode）",
            "全任务 batch",
            "全任务首条（每个任务只跑第一个 episode）",
            "单条 metadata：只拍一张 head 相机图后退出（PNG 保存到该 episode/preview）",
            "单 episode 首图：选任务后输入多个 episode 编号（空格/逗号分隔），依次拍 preview",
            "单任务 batch：每条 metadata 只拍一张 head 相机图后退出（PNG 保存到各自 episode/preview）",
            "全任务 batch：所有任务的所有 episode 只拍首图（PNG 保存到各自 episode/preview）",
        ],
    )

    if mode_idx == 0:
        tasks = list(task_map.keys())
        ti = ask_choice("先选任务", tasks)
        task = tasks[ti]
        items = task_map[task]
        labels = [f"{p.parent.parent.name} ({p})" for p in items]
        mi = ask_choice(f"再选 {task} 的 episode", labels)
        md = items[mi]
        rc = run_one(md, args.task_config, args.gpu_id, args.skip_run)
        if rc != 0:
            rec = _failure_record("single_episode_video", md, rc)
            _append_failure_jsonl(rec)
            _write_last_summary(
                {
                    "mode": "single_episode_video",
                    "task": task,
                    "total": 1,
                    "ok": 0,
                    "fail": 1,
                    "failures": [rec],
                    "failure_log_jsonl": str(_FAILURE_JSONL),
                }
            )
            print(f"失败已记录: {_FAILURE_JSONL} / {_LAST_SUMMARY_JSON}")
        print(f"\n完成，返回码: {rc}")
        return

    if mode_idx == 4:
        tasks = list(task_map.keys())
        ti = ask_choice("先选任务", tasks)
        task = tasks[ti]
        items = task_map[task]
        labels = [f"{p.parent.parent.name} ({p})" for p in items]
        mi = ask_choice(f"再选 {task} 的 episode（将保存快照）", labels)
        if args.skip_run:
            print("--skip-run 与快照模式不兼容，已忽略 skip_run。")
        md = items[mi]
        rc = run_one(md, args.task_config, args.gpu_id, skip_run=False, snapshot_only=True)
        if rc != 0:
            rec = _failure_record("single_episode_snapshot", md, rc)
            _append_failure_jsonl(rec)
            _write_last_summary(
                {
                    "mode": "single_episode_snapshot",
                    "task": task,
                    "total": 1,
                    "ok": 0,
                    "fail": 1,
                    "failures": [rec],
                    "failure_log_jsonl": str(_FAILURE_JSONL),
                }
            )
            print(f"失败已记录: {_FAILURE_JSONL} / {_LAST_SUMMARY_JSON}")
        print(f"\n完成，返回码: {rc}")
        return

    if mode_idx == 5:
        tasks = list(task_map.keys())
        ti = ask_choice("选择任务（输入一个或多个 episode 编号，依次拍 preview）", tasks)
        task = tasks[ti]
        while True:
            raw_line = input(
                "请输入 episode 编号，多个用空格或逗号分隔（如 730 779 或 episode602,803），留空取消: "
            ).strip()
            if not raw_line:
                print("已取消。")
                return
            ep_dirs = parse_episode_dir_names(raw_line)
            if not ep_dirs:
                print("未能解析出任何有效编号，请使用数字或 episode 前缀，例如 602 episode779。")
                continue
            bad_tokens = []
            for t0 in re.split(r"[\s,，;；]+", raw_line.strip()):
                t0 = t0.strip()
                if not t0:
                    continue
                if normalize_episode_dir_name(t0) is None:
                    bad_tokens.append(t0)
            if bad_tokens:
                print(f"已忽略无法解析的片段: {', '.join(bad_tokens)}")
            meta_paths: list[Path] = []
            missing: list[str] = []
            for ep_dir in ep_dirs:
                mp = args.root / task / ep_dir / "json" / "metadata.json"
                if mp.is_file():
                    meta_paths.append(mp)
                else:
                    missing.append(str(mp))
            if missing:
                for m in missing:
                    print(f"跳过（无 metadata）: {m}")
            if not meta_paths:
                print("没有可运行的 episode，请重新输入。")
                continue
            break
        if args.skip_run:
            print("--skip-run 与快照模式不兼容，已忽略 skip_run。")
        print(f"\n将依次处理 {len(meta_paths)} 条: {[p.parent.parent.name for p in meta_paths]}")
        ok = 0
        skipped = 0
        failed: list[dict[str, str | int]] = []
        mode_name = "multi_episode_snapshot"
        for i, meta_path in enumerate(meta_paths, 1):
            print(f"\n[{i}/{len(meta_paths)}] {meta_path.parent.parent.name}")
            if skip_existing and _snapshot_output_exists(meta_path):
                print("  -> 跳过（已有 preview PNG）")
                skipped += 1
                continue
            rc = run_one(meta_path, args.task_config, args.gpu_id, skip_run=False, snapshot_only=True)
            if rc == 0:
                ok += 1
            else:
                print(f"  -> 失败，返回码 {rc}")
                rec = _failure_record(mode_name, meta_path, rc)
                _append_failure_jsonl(rec)
                failed.append(rec)
        _write_last_summary(
            {
                "mode": mode_name,
                "task": task,
                "total": len(meta_paths),
                "skipped": skipped,
                "ok": ok,
                "fail": len(failed),
                "failures": failed,
                "failure_log_jsonl": str(_FAILURE_JSONL),
            }
        )
        print(f"\n单 episode 首图批量结束: 成功 {ok}, 跳过 {skipped}, 失败 {len(failed)} / 共 {len(meta_paths)}")
        print(f"失败明细已追加: {_FAILURE_JSONL}")
        print(f"本次汇总: {_LAST_SUMMARY_JSON}")
        return

    if mode_idx == 6:
        tasks = list(task_map.keys())
        ti = ask_choice("选择任务（批量拍单图）", tasks)
        task = tasks[ti]
        items = task_map[task]
        print(f"\n开始单图 batch: {task}，共 {len(items)} 条")
        ok = 0
        skipped = 0
        failed: list[dict[str, str | int]] = []
        mode_name = "single_task_snapshot_batch"
        for i, md in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {md.parent.parent.name}")
            if skip_existing and _snapshot_output_exists(md):
                print("  -> 跳过（已有 preview PNG）")
                skipped += 1
                continue
            rc = run_one(md, args.task_config, args.gpu_id, skip_run=False, snapshot_only=True)
            if rc == 0:
                ok += 1
            else:
                print(f"  -> 失败，返回码 {rc}")
                rec = _failure_record(mode_name, md, rc)
                _append_failure_jsonl(rec)
                failed.append(rec)
        _write_last_summary(
            {
                "mode": mode_name,
                "task": task,
                "total": len(items),
                "skipped": skipped,
                "ok": ok,
                "fail": len(failed),
                "failures": failed,
                "failure_log_jsonl": str(_FAILURE_JSONL),
            }
        )
        print(f"\n单图 batch 结束: 成功 {ok}, 跳过 {skipped}, 失败 {len(failed)} / 共 {len(items)}")
        print(f"失败明细已追加: {_FAILURE_JSONL}")
        print(f"本次汇总: {_LAST_SUMMARY_JSON}")
        return

    if mode_idx == 7:
        all_items = [p for items in task_map.values() for p in items]
        print(f"\n开始全任务单图 batch，共 {len(all_items)} 条")
        ok = 0
        skipped = 0
        failed: list[dict[str, str | int]] = []
        mode_name = "all_tasks_snapshot_batch"
        for i, md in enumerate(all_items, 1):
            print(f"[{i}/{len(all_items)}] {md.parent.parent.parent.name}/{md.parent.parent.name}")
            if skip_existing and _snapshot_output_exists(md):
                print("  -> 跳过（已有 preview PNG）")
                skipped += 1
                continue
            rc = run_one(md, args.task_config, args.gpu_id, skip_run=False, snapshot_only=True)
            if rc == 0:
                ok += 1
            else:
                print(f"  -> 失败，返回码 {rc}")
                rec = _failure_record(mode_name, md, rc)
                _append_failure_jsonl(rec)
                failed.append(rec)
        _write_last_summary(
            {
                "mode": mode_name,
                "total": len(all_items),
                "skipped": skipped,
                "ok": ok,
                "fail": len(failed),
                "failures": failed,
                "failure_log_jsonl": str(_FAILURE_JSONL),
            }
        )
        print(f"\n全任务单图 batch 结束: 成功 {ok}, 跳过 {skipped}, 失败 {len(failed)} / 共 {len(all_items)}")
        print(f"失败明细已追加: {_FAILURE_JSONL}")
        print(f"本次汇总: {_LAST_SUMMARY_JSON}")
        return

    if mode_idx == 1:
        tasks = list(task_map.keys())
        ti = ask_choice("选择任务", tasks)
        task = tasks[ti]
        items = task_map[task]
        print(f"\n开始 batch: {task}，共 {len(items)} 条")
        ok = 0
        skipped = 0
        failed: list[dict[str, str | int]] = []
        mode_name = "single_task_video_batch"
        for i, md in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {md.parent.parent.name}")
            if skip_existing and _video_outputs_complete(md):
                print("  -> 跳过（已有 tra1/video + tra1/data）")
                skipped += 1
                continue
            rc = run_one(md, args.task_config, args.gpu_id, args.skip_run)
            if rc == 0:
                ok += 1
            else:
                print(f"  -> 失败，返回码 {rc}")
                rec = _failure_record(mode_name, md, rc)
                _append_failure_jsonl(rec)
                failed.append(rec)
        _write_last_summary(
            {
                "mode": mode_name,
                "task": task,
                "total": len(items),
                "skipped": skipped,
                "ok": ok,
                "fail": len(failed),
                "failures": failed,
                "failure_log_jsonl": str(_FAILURE_JSONL),
            }
        )
        print(f"\nbatch 结束: 成功 {ok}, 跳过 {skipped}, 失败 {len(failed)} / 共 {len(items)}")
        print(f"失败明细已追加: {_FAILURE_JSONL}")
        print(f"本次汇总: {_LAST_SUMMARY_JSON}")
        return

    if mode_idx == 2:
        all_items = [p for items in task_map.values() for p in items]
        title = "全任务 batch"
    else:
        all_items = pick_first_episode_per_task(task_map)
        title = "全任务首条 batch"

    print(f"\n开始{title}，共 {len(all_items)} 条")
    ok = 0
    skipped = 0
    failed: list[dict[str, str | int]] = []
    mode_name = "all_tasks_video_batch" if mode_idx == 2 else "all_tasks_first_episode_video_batch"
    for i, md in enumerate(all_items, 1):
        print(f"[{i}/{len(all_items)}] {md.parent.parent.parent.name}/{md.parent.parent.name}")
        if skip_existing and _video_outputs_complete(md):
            print("  -> 跳过（已有 tra1/video + tra1/data）")
            skipped += 1
            continue
        rc = run_one(md, args.task_config, args.gpu_id, args.skip_run)
        if rc == 0:
            ok += 1
        else:
            print(f"  -> 失败，返回码 {rc}")
            rec = _failure_record(mode_name, md, rc)
            _append_failure_jsonl(rec)
            failed.append(rec)
    _write_last_summary(
        {
            "mode": mode_name,
            "title": title,
            "total": len(all_items),
            "skipped": skipped,
            "ok": ok,
            "fail": len(failed),
            "failures": failed,
            "failure_log_jsonl": str(_FAILURE_JSONL),
        }
    )
    print(f"\n{title}结束: 成功 {ok}, 跳过 {skipped}, 失败 {len(failed)} / 共 {len(all_items)}")
    print(f"失败明细已追加: {_FAILURE_JSONL}")
    print(f"本次汇总: {_LAST_SUMMARY_JSON}")


if __name__ == "__main__":
    main()

