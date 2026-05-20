#!/usr/bin/env python3
"""
run_robotwin_interactive.py — 按 metadata 跑 RoboTwin + 交互式批量菜单（已合并原 run_robotwin_from_metadata）

功能：
  - 带 --metadata：直接调用 RoboTwin/wm_data/run_from_metadata.py（单条 collect / snapshot）
  - 不带 --metadata：扫描 --root 下 metadata.json，终端菜单批量执行

启动（交互）：
  cd /home/arlen-n/workspace/cy/wm/wm_labeler/data_gen
  python3 run_robotwin_interactive.py \
    --root /home/arlen-n/workspace/cy/wm/log/robotwin_generated_v2 \
    --task-config demo_clean --gpu-id 0

  python3 run_robotwin_interactive.py --list-only

  # 非交互（跳过两级菜单）：
  python3 run_robotwin_interactive.py --root .../robotwin_generated_v2 \\
    --mode preview-multi --task adjust_bottle --episodes "53 70"

启动（单条，与原 run_robotwin_from_metadata.py 相同）：
  python3 run_robotwin_interactive.py \\
    --metadata /path/to/episode/json/metadata.json \\
    --task-config demo_clean --gpu-id 0

  python3 run_robotwin_from_metadata.py ...   # 兼容别名，内部转调本文件

输出：
  <episode>/tra1|tra2|preview/...
  gen_config/batch_run_failures.jsonl、gen_config/batch_run_last_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_GEN_CONFIG_DIR = _SCRIPT_DIR / "gen_config"
_FAILURE_JSONL = _GEN_CONFIG_DIR / "batch_run_failures.jsonl"
_LAST_SUMMARY_JSON = _GEN_CONFIG_DIR / "batch_run_last_summary.json"


def resolve_wm_data_dir() -> Path:
    """Locate RoboTwin/wm_data (run_from_metadata.py) relative to wm repo."""
    candidates = [
        _SCRIPT_DIR.parent.parent / "RoboTwin" / "wm_data",
        _SCRIPT_DIR.parent / "RoboTwin" / "wm_data",
        _SCRIPT_DIR / "RoboTwin" / "wm_data",
    ]
    for wm_data in candidates:
        if (wm_data / "run_from_metadata.py").is_file():
            return wm_data.resolve()
    tried = "\n  ".join(str(p) for p in candidates)
    raise SystemExit(f"Missing RoboTwin/wm_data/run_from_metadata.py. Tried:\n  {tried}")


def invoke_run_from_metadata(argv: list[str]) -> int:
    """Run RoboTwin/wm_data/run_from_metadata.main() with given CLI args (no script name)."""
    wm_data = resolve_wm_data_dir()
    wm_data_str = str(wm_data)
    if wm_data_str not in sys.path:
        sys.path.insert(0, wm_data_str)
    from run_from_metadata import main as rob_main  # noqa: E402

    saved_argv = sys.argv[:]
    try:
        sys.argv = [str(wm_data / "run_from_metadata.py"), *argv]
        rc = rob_main()
        return int(rc) if rc is not None else 0
    except (RuntimeError, FileNotFoundError) as e:
        print(f"run_from_metadata error: {e}", file=sys.stderr)
        return 1
    finally:
        sys.argv = saved_argv


def main_run_from_metadata() -> int:
    """CLI entry when invoked as run_robotwin_from_metadata.py or with --metadata."""
    return invoke_run_from_metadata(sys.argv[1:])


def _is_direct_metadata_cli() -> bool:
    return "--metadata" in sys.argv[1:]


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
    argv = [
        "--metadata",
        str(metadata),
        "--task-config",
        task_config,
        "--gpu-id",
        gpu_id,
    ]
    if skip_run:
        argv.append("--skip-run")
    if snapshot_only:
        argv.append("--snapshot-only")
    print("\n>>> run_from_metadata", " ".join(argv))
    return invoke_run_from_metadata(argv)


@dataclass
class RunPlan:
    mode_id: str
    snapshot: bool
    items: list[Path]
    task: str | None = None
    title: str = ""


def _episode_menu_label(md: Path, snapshot: bool) -> str:
    ep = md.parent.parent.name
    if snapshot:
        status = "已有 preview" if _snapshot_output_exists(md) else "待拍 preview"
    else:
        status = "已有 tra1" if _video_outputs_complete(md) else "待跑 tra1"
    return f"{ep}  [{status}]"


def print_session_banner(args: argparse.Namespace, task_map: dict[str, list[Path]]) -> None:
    total_eps = sum(len(v) for v in task_map.values())
    print("\n" + "=" * 56)
    print(f"  数据根目录: {args.root}")
    print(f"  任务数: {len(task_map)}    episode 数: {total_eps}")
    print(f"  task_config={args.task_config}  gpu_id={args.gpu_id}")
    skip_existing = not args.no_skip_existing
    print(f"  跳过已有输出: {skip_existing}")
    print("=" * 56)


def ask_choice(prompt: str, options: list[str], *, allow_cancel: bool = False) -> int | None:
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  [{i}] {o}")
    hint = "输入编号"
    if allow_cancel:
        hint += "；留空或 q 取消"
    print(f"  （{hint}）")
    while True:
        val = input("> ").strip()
        if allow_cancel and (not val or val.lower() in ("q", "quit", "c", "cancel")):
            return None
        if val.isdigit() and 1 <= int(val) <= len(options):
            return int(val) - 1
        print("输入无效，请重新输入。")


def pick_task_name(task_map: dict[str, list[Path]], prompt: str = "选择任务") -> str | None:
    tasks = list(task_map.keys())
    print(f"\n{prompt}（共 {len(tasks)} 个）")
    print("  输入编号 / 任务名前缀（如 adj）/ list 列出全部 / q 取消")
    while True:
        val = input("> ").strip()
        if not val or val.lower() in ("q", "quit", "c"):
            return None
        if val.lower() == "list":
            for i, t in enumerate(tasks, 1):
                print(f"  [{i:>3}] {t}  ({len(task_map[t])} ep)")
            continue
        if val.isdigit():
            idx = int(val)
            if 1 <= idx <= len(tasks):
                return tasks[idx - 1]
            print(f"编号超出范围 1–{len(tasks)}")
            continue
        low = val.lower()
        exact = [t for t in tasks if t.lower() == low]
        if len(exact) == 1:
            return exact[0]
        prefix = [t for t in tasks if t.lower().startswith(low)]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            print("匹配到多个任务，请输入更完整的前缀或编号：")
            for t in prefix[:20]:
                print(f"  - {t}")
            if len(prefix) > 20:
                print(f"  ... 还有 {len(prefix) - 20} 个")
            continue
        print(f"未找到任务 «{val}»")


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


def resolve_metadata_paths(
    root: Path,
    task: str,
    episode_line: str,
    *,
    known_items: list[Path] | None = None,
) -> tuple[list[Path], list[str]]:
    """Parse episode numbers / names into metadata paths under one task."""
    ep_dirs = parse_episode_dir_names(episode_line)
    if not ep_dirs:
        return [], []
    meta_paths: list[Path] = []
    missing: list[str] = []
    known_by_ep = {p.parent.parent.name: p for p in (known_items or [])}
    for ep_dir in ep_dirs:
        if ep_dir in known_by_ep:
            meta_paths.append(known_by_ep[ep_dir])
            continue
        mp = root / task / ep_dir / "json" / "metadata.json"
        if mp.is_file():
            meta_paths.append(mp)
        else:
            missing.append(ep_dir)
    return meta_paths, missing


def pick_single_metadata(task: str, items: list[Path], snapshot: bool) -> Path | None:
    if len(items) <= 30:
        labels = [_episode_menu_label(md, snapshot) for md in items]
        idx = ask_choice(f"选择 {task} 的 episode", labels, allow_cancel=True)
        return items[idx] if idx is not None else None

    print(f"\n{task} 共 {len(items)} 条：输入 list 列出 / 列表编号 / episode 编号（如 53）/ q 取消")
    while True:
        val = input("> ").strip()
        if not val or val.lower() in ("q", "quit", "c"):
            return None
        if val.lower() == "list":
            for i, md in enumerate(items, 1):
                print(f"  [{i:>3}] {_episode_menu_label(md, snapshot)}")
            continue
        if val.isdigit():
            idx = int(val)
            if 1 <= idx <= len(items):
                return items[idx - 1]
            print(f"列表编号超出范围 1–{len(items)}")
            continue
        ep = normalize_episode_dir_name(val)
        if ep:
            for md in items:
                if md.parent.parent.name == ep:
                    return md
            print(f"未找到 {ep}（该任务下无 metadata）")
            continue
        print("请输入 list、列表编号或 episode 编号")


def prompt_episode_line(task: str) -> str | None:
    print(f"\n任务 «{task}»：输入 episode 编号，多个用空格/逗号分隔")
    print("  示例: 53  或  53,70  或  episode53 episode70")
    raw = input("> ").strip()
    return raw if raw else None


def execute_batch_run(
    plan: RunPlan,
    args: argparse.Namespace,
    skip_existing: bool,
) -> None:
    items = plan.items
    snapshot = plan.snapshot
    mode_name = plan.mode_id
    if not items:
        print("没有可执行的 episode。")
        return
    if snapshot and args.skip_run:
        print("（快照模式忽略 --skip-run）")

    print(f"\n—— {plan.title or mode_name}：共 {len(items)} 条 ——")
    ok = skipped = 0
    failed: list[dict[str, str | int]] = []
    for i, md in enumerate(items, 1):
        task_ep = f"{md.parent.parent.parent.name}/{md.parent.parent.name}"
        print(f"\n[{i}/{len(items)}] {task_ep}")
        if skip_existing:
            if snapshot and _snapshot_output_exists(md):
                print("  -> 跳过（已有 preview）")
                skipped += 1
                continue
            if not snapshot and _video_outputs_complete(md):
                print("  -> 跳过（已有 tra1）")
                skipped += 1
                continue
        rc = run_one(
            md,
            args.task_config,
            args.gpu_id,
            skip_run=False if snapshot else args.skip_run,
            snapshot_only=snapshot,
        )
        if rc == 0:
            ok += 1
        else:
            print(f"  -> 失败，返回码 {rc}")
            rec = _failure_record(mode_name, md, rc)
            _append_failure_jsonl(rec)
            failed.append(rec)

    summary: dict = {
        "mode": mode_name,
        "title": plan.title,
        "total": len(items),
        "skipped": skipped,
        "ok": ok,
        "fail": len(failed),
        "failures": failed,
        "failure_log_jsonl": str(_FAILURE_JSONL),
    }
    if plan.task:
        summary["task"] = plan.task
    _write_last_summary(summary)
    print(f"\n结束: 成功 {ok}, 跳过 {skipped}, 失败 {len(failed)} / 共 {len(items)}")
    print(f"失败明细: {_FAILURE_JSONL}")
    print(f"本次汇总: {_LAST_SUMMARY_JSON}")


def build_plan_interactive(task_map: dict[str, list[Path]], root: Path) -> RunPlan | None:
    kind_idx = ask_choice(
        "① 选择产出类型",
        [
            "Preview 快照（head 相机 PNG → episode/preview/）【常用】",
            "完整轨迹（hdf5 + mp4 → episode/tra1/）",
        ],
        allow_cancel=True,
    )
    if kind_idx is None:
        return None
    snapshot = kind_idx == 0

    if snapshot:
        scope_opts = [
            "★ 指定任务 + 输入多个 episode 编号（推荐）",
            "单条 episode",
            "该任务下全部 episode",
            "全部任务的全部 episode",
        ]
    else:
        scope_opts = [
            "单条 episode",
            "该任务下全部 episode",
            "全部任务的全部 episode",
            "每个任务只跑第一条 episode",
        ]
    scope_idx = ask_choice("② 选择范围", scope_opts, allow_cancel=True)
    if scope_idx is None:
        return None

    if snapshot:
        if scope_idx == 0:
            task = pick_task_name(task_map, "③ 选择任务")
            if task is None:
                return None
            line = prompt_episode_line(task)
            if line is None:
                return None
            items, missing = resolve_metadata_paths(root, task, line, known_items=task_map[task])
            for ep in missing:
                print(f"跳过（无 metadata）: {task}/{ep}")
            if not items:
                print("没有可运行的 episode。")
                return None
            print(f"将处理: {[p.parent.parent.name for p in items]}")
            return RunPlan(
                mode_id="multi_episode_snapshot",
                snapshot=True,
                items=items,
                task=task,
                title=f"Preview · {task} · 指定多条",
            )
        if scope_idx == 1:
            task = pick_task_name(task_map)
            if task is None:
                return None
            md = pick_single_metadata(task, task_map[task], snapshot=True)
            if md is None:
                return None
            return RunPlan(
                mode_id="single_episode_snapshot",
                snapshot=True,
                items=[md],
                task=task,
                title=f"Preview · {task} · 单条",
            )
        if scope_idx == 2:
            task = pick_task_name(task_map)
            if task is None:
                return None
            items = task_map[task]
            return RunPlan(
                mode_id="single_task_snapshot_batch",
                snapshot=True,
                items=items,
                task=task,
                title=f"Preview · {task} · 全任务",
            )
        all_items = [p for v in task_map.values() for p in v]
        return RunPlan(
            mode_id="all_tasks_snapshot_batch",
            snapshot=True,
            items=all_items,
            title="Preview · 全部任务",
        )

    # video
    if scope_idx == 0:
        task = pick_task_name(task_map)
        if task is None:
            return None
        md = pick_single_metadata(task, task_map[task], snapshot=False)
        if md is None:
            return None
        return RunPlan(
            mode_id="single_episode_video",
            snapshot=False,
            items=[md],
            task=task,
            title=f"轨迹 · {task} · 单条",
        )
    if scope_idx == 1:
        task = pick_task_name(task_map)
        if task is None:
            return None
        return RunPlan(
            mode_id="single_task_video_batch",
            snapshot=False,
            items=task_map[task],
            task=task,
            title=f"轨迹 · {task} · 全任务",
        )
    if scope_idx == 2:
        all_items = [p for v in task_map.values() for p in v]
        return RunPlan(
            mode_id="all_tasks_video_batch",
            snapshot=False,
            items=all_items,
            title="轨迹 · 全部任务",
        )
    all_items = pick_first_episode_per_task(task_map)
    return RunPlan(
        mode_id="all_tasks_first_episode_video_batch",
        snapshot=False,
        items=all_items,
        title="轨迹 · 每任务首条",
    )


def build_plan_from_cli(
    args: argparse.Namespace,
    task_map: dict[str, list[Path]],
) -> RunPlan | None:
    mode = args.mode
    if not mode:
        return None
    snapshot = mode.startswith("preview")
    if args.task and args.task not in task_map:
        raise SystemExit(f"未知任务: {args.task!r}，可用 --list-only 查看")

    if mode == "preview-multi":
        if not args.task or not args.episodes:
            raise SystemExit("--mode preview-multi 需要 --task 与 --episodes")
        items, missing = resolve_metadata_paths(
            args.root, args.task, args.episodes, known_items=task_map.get(args.task, [])
        )
        for ep in missing:
            print(f"跳过（无 metadata）: {args.task}/{ep}")
        return RunPlan(
            mode_id="multi_episode_snapshot",
            snapshot=True,
            items=items,
            task=args.task,
            title=f"CLI preview-multi · {args.task}",
        )
    if mode == "preview-one":
        if not args.task or not args.episodes:
            raise SystemExit("--mode preview-one 需要 --task 与 --episodes（单条编号）")
        items, _ = resolve_metadata_paths(args.root, args.task, args.episodes, known_items=task_map.get(args.task, []))
        if not items:
            raise SystemExit("未找到 metadata")
        return RunPlan(mode_id="single_episode_snapshot", snapshot=True, items=items[:1], task=args.task)
    if mode == "preview-task":
        if not args.task:
            raise SystemExit("--mode preview-task 需要 --task")
        return RunPlan(
            mode_id="single_task_snapshot_batch",
            snapshot=True,
            items=task_map[args.task],
            task=args.task,
        )
    if mode == "preview-all":
        return RunPlan(
            mode_id="all_tasks_snapshot_batch",
            snapshot=True,
            items=[p for v in task_map.values() for p in v],
        )
    if mode == "video-one":
        if not args.task or not args.episodes:
            raise SystemExit("--mode video-one 需要 --task 与 --episodes")
        items, _ = resolve_metadata_paths(args.root, args.task, args.episodes, known_items=task_map.get(args.task, []))
        if not items:
            raise SystemExit("未找到 metadata")
        return RunPlan(mode_id="single_episode_video", snapshot=False, items=items[:1], task=args.task)
    if mode == "video-task":
        if not args.task:
            raise SystemExit("--mode video-task 需要 --task")
        return RunPlan(
            mode_id="single_task_video_batch",
            snapshot=False,
            items=task_map[args.task],
            task=args.task,
        )
    if mode == "video-all":
        return RunPlan(
            mode_id="all_tasks_video_batch",
            snapshot=False,
            items=[p for v in task_map.values() for p in v],
        )
    if mode == "video-first":
        return RunPlan(
            mode_id="all_tasks_first_episode_video_batch",
            snapshot=False,
            items=pick_first_episode_per_task(task_map),
        )
    raise SystemExit(f"未知 --mode: {mode}")


def main_interactive() -> None:
    p = argparse.ArgumentParser(description="Interactive batch runner for RoboTwin run_from_metadata")
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
    p.add_argument(
        "--mode",
        choices=(
            "preview-multi",
            "preview-one",
            "preview-task",
            "preview-all",
            "video-one",
            "video-task",
            "video-all",
            "video-first",
        ),
        default=None,
        help="非交互：跳过菜单（如 preview-multi 需配合 --task --episodes）",
    )
    p.add_argument("--task", default=None, help="任务名，配合 --mode 使用")
    p.add_argument("--episodes", default=None, help="episode 编号，如 53,70 或 53 70")
    args = p.parse_args()
    skip_existing = not args.no_skip_existing

    task_map = discover_metadata(args.root)
    if not task_map:
        print(f"未发现 metadata: {args.root}")
        return

    if args.list_only:
        print(f"发现任务数: {len(task_map)}")
        for t, items in task_map.items():
            print(f"  {t}: {len(items)} ep")
        return

    print_session_banner(args, task_map)

    plan = build_plan_from_cli(args, task_map)
    if plan is None:
        plan = build_plan_interactive(task_map, args.root)
    if plan is None:
        print("已取消。")
        return

    execute_batch_run(plan, args, skip_existing)


def main() -> int:
    if _is_direct_metadata_cli():
        return main_run_from_metadata()
    main_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

