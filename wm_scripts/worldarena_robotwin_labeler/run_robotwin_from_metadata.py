#!/usr/bin/env python3
"""
在 RoboTwin_khl 内根据 robotwin_generated 下的 metadata.json 做一次专家轨迹采集，
写入指定 tra 目录（data/*.hdf5 + video/*.mp4）。

依赖：在 RoboTwin_khl 目录下可导入 envs、task_config；需已配置 conda 与 GPU。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
import importlib
from pathlib import Path

import yaml


def _repo_workspace_root() -> Path:
    """khl_workspace：支持 .../worldarena_robotwin_labeler/ 与 .../wm_scripts/worldarena_robotwin_labeler/。"""
    labeler = Path(__file__).resolve().parent
    if labeler.parent.name == "wm_scripts":
        return labeler.parent.parent.parent
    return labeler.parent


WORKSPACE = _repo_workspace_root()
ROBOT = WORKSPACE / "RoboTwin_khl"


def _prepare_robot_path() -> None:
    os.chdir(ROBOT)
    for p in (str(ROBOT), str(ROBOT / "policy"), str(ROBOT / "description" / "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file: str) -> dict:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())


def build_collect_args(task_name: str, task_config: str, save_path: Path, ep_num: int, seed: int) -> dict:
    from envs import CONFIGS_PATH

    cfg_name = task_config if str(task_config).endswith(".yml") else f"{task_config}.yml"
    with open(os.path.join(CONFIGS_PATH, cfg_name), "r", encoding="utf-8") as f:
        args = yaml.safe_load(f.read())

    args["task_name"] = task_name
    args["task_config"] = str(task_config).replace(".yml", "")
    args.setdefault("ckpt_setting", "collect")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f.read())

    def get_embodiment_file(et: str):
        robot_file = _embodiment_types[et]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.safe_load(f.read())

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    args["save_path"] = str(save_path.resolve())
    args["now_ep_num"] = ep_num
    args["save_data"] = True
    args["eval_mode"] = False
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None
    args["seed"] = int(seed)
    return args


def slots_from_metadata(md: dict) -> list[dict]:
    obs = md.get("object_binding") or {}
    out: list[dict] = []
    for o in obs.get("confirmed_objects") or []:
        if not isinstance(o, dict):
            continue
        slot = str(o.get("slot", "A")).strip() or "A"
        mn = str(o.get("modelname", "")).strip()
        if not mn:
            continue
        try:
            mid = int(o["model_id"])
        except (KeyError, TypeError, ValueError):
            continue
        item = {"slot": slot, "modelname": mn, "model_id": mid}
        if isinstance(o.get("attrs"), dict) and o["attrs"]:
            item["attrs"] = dict(o["attrs"])
        out.append(item)
    if out:
        return out
    po = obs.get("primary_object")
    if isinstance(po, dict) and po.get("modelname"):
        try:
            mid = int(po["model_id"])
        except (KeyError, TypeError, ValueError):
            return []
        return [{"slot": "A", "modelname": str(po["modelname"]).strip(), "model_id": mid}]
    return []


def episode_num_from_name(episode_name: str) -> int:
    m = re.match(r"^episode(\d+)$", episode_name.strip())
    if not m:
        raise ValueError(f"无法解析 episode 目录名: {episode_name!r}")
    return int(m.group(1))


def read_seed(tra_dir: Path, override: int | None) -> int:
    if override is not None:
        return int(override)
    p = tra_dir / "collect_seeds.txt"
    if p.is_file():
        try:
            line = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            return int(line)
        except (ValueError, IndexError):
            pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tra-dir", type=Path, required=True, help="输出 tra 目录，如 .../episode216/tra3")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    meta_path = args.metadata.resolve()
    tra_dir = args.tra_dir.resolve()
    if not meta_path.is_file():
        raise SystemExit(f"metadata 不存在: {meta_path}")

    ep_dir = tra_dir.parent
    episode_name = ep_dir.name
    task_name_dir = ep_dir.parent.name
    ep_num = episode_num_from_name(episode_name)

    md = json.loads(meta_path.read_text(encoding="utf-8"))
    task_name = str(md.get("task_name", task_name_dir)).strip()
    if task_name != task_name_dir:
        print(
            f"[warn] metadata.task_name={task_name!r} 与目录 {task_name_dir!r} 不一致，以目录名为准采集。",
            file=sys.stderr,
        )
        task_name = task_name_dir

    slots = slots_from_metadata(md)
    if not slots:
        raise SystemExit("metadata 中缺少可用的 object_binding.confirmed_objects 或 primary_object")

    os.environ["ROBOTWIN_FORCE_TASK_NAME"] = task_name
    os.environ["ROBOTWIN_FORCE_SLOTS_JSON"] = json.dumps(slots, ensure_ascii=False)
    os.environ.pop("ROBOTWIN_FORCE_MODEL_NAME", None)
    os.environ.pop("ROBOTWIN_FORCE_MODEL_ID", None)

    seed = read_seed(tra_dir, args.seed)
    tra_dir.mkdir(parents=True, exist_ok=True)
    (tra_dir / "data").mkdir(parents=True, exist_ok=True)
    (tra_dir / "video").mkdir(parents=True, exist_ok=True)

    _prepare_robot_path()
    from envs.utils.create_actor import UnStableError  # noqa: E402

    try:
        env_args = build_collect_args(task_name, args.task_config, tra_dir, ep_num, seed)
        TASK_ENV = class_decorator(task_name)
        TASK_ENV.setup_demo(is_test=False, **env_args)
        TASK_ENV.play_once()
        TASK_ENV.merge_pkl_to_hdf5_video()
        TASK_ENV.close_env(clear_cache=True)
    except UnStableError as e:
        print(f"[UnStableError] {e}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from e
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

    run_log = {
        "task_name": task_name,
        "episode_name": episode_name,
        "task_config": str(args.task_config).replace(".yml", ""),
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "metadata": str(meta_path),
        "note": "worldarena_robotwin_labeler/run_robotwin_from_metadata.py 生成",
    }
    (tra_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
