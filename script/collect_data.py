import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
_wm_data_dir = os.path.normpath(os.path.join(parent_directory, "..", "wm_data"))
if _wm_data_dir not in sys.path:
    sys.path.insert(0, _wm_data_dir)
from seed_blacklist import (  # noqa: E402
    load_bad_seeds,
    next_seed_skipping_bad,
    record_bad_seed,
    resolve_bad_seeds_path,
)

try:
    from envs.utils.create_actor import UnStableError  # noqa: E402
except ImportError:
    UnStableError = Exception  # type: ignore[misc, assignment]


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

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
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    forced_seed_raw = os.environ.get("ROBOTWIN_FORCE_SEED", "").strip()
    forced_seed = int(forced_seed_raw) if forced_seed_raw else None
    epid = forced_seed if forced_seed is not None else 0
    suc_num, fail_num, seed_list = 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    def _env_truthy(key: str) -> bool:
        v = os.environ.get(key, "").strip().lower()
        return v in ("1", "true", "yes", "on")

    # One head-camera frame then exit (used by worldarena_robotwin_labeler --snapshot-only).
    if _env_truthy("ROBOTWIN_SNAPSHOT_ONLY"):
        snapshot_path = os.environ.get("ROBOTWIN_SNAPSHOT_PATH", "").strip()
        if not snapshot_path:
            raise RuntimeError("ROBOTWIN_SNAPSHOT_PATH is required when ROBOTWIN_SNAPSHOT_ONLY is set")
        seed = int(os.environ.get("ROBOTWIN_SNAPSHOT_SEED", "0"))
        bad_path = resolve_bad_seeds_path(args["save_path"])
        os.makedirs(args["save_path"], exist_ok=True)
        try:
            TASK_ENV.setup_demo(now_ep_num=0, seed=seed, **args)
            TASK_ENV.save_camera_rgb(snapshot_path, camera_name="head_camera")
            print(f"\033[92m[ROBOTWIN_SNAPSHOT] saved:\033[0m {snapshot_path}")
        except Exception:
            if record_bad_seed(bad_path, seed):
                print(f"\033[93m[bad_seeds] recorded seed {seed} -> {bad_path}\033[0m")
            raise
        finally:
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
            if args.get("render_freq"):
                try:
                    TASK_ENV.viewer.close()
                except Exception:
                    pass
        return

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if forced_seed is None and os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        bad_path = resolve_bad_seeds_path(args["save_path"])
        bad_seeds = load_bad_seeds(bad_path)
        if bad_seeds and forced_seed is None:
            print(f"\033[93m[bad_seeds] loaded {len(bad_seeds)} known-bad seed(s) from {bad_path}\033[0m")

        def _write_seed_file() -> None:
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        def _attempt_seed(try_epid: int, now_ep_num: int) -> bool:
            nonlocal fail_num, bad_seeds
            try:
                TASK_ENV.setup_demo(now_ep_num=now_ep_num, seed=try_epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {now_ep_num} success! (seed = {try_epid})")
                    seed_list.append(try_epid)
                    TASK_ENV.save_traj_data(now_ep_num)
                    TASK_ENV.close_env()
                    if args["render_freq"]:
                        TASK_ENV.viewer.close()
                    return True

                print(f"simulate data episode {now_ep_num} fail! (seed = {try_epid})")
                fail_num += 1
                if record_bad_seed(bad_path, try_epid):
                    bad_seeds.add(try_epid)
                TASK_ENV.close_env()
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {now_ep_num} fail! (seed = {try_epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                if record_bad_seed(bad_path, try_epid):
                    bad_seeds.add(try_epid)
                TASK_ENV.close_env()
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                print(" -------------")
                print(f"simulate data episode {now_ep_num} fail! (seed = {try_epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                if record_bad_seed(bad_path, try_epid):
                    bad_seeds.add(try_epid)
                TASK_ENV.close_env()
                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)
            return False

        def _single_pass_metadata_collect(try_seed: int) -> bool:
            """Plan once and record hdf5/mp4; skip phase-2 trajectory replay (avoids branch mismatch)."""
            nonlocal fail_num, suc_num
            print(
                f"\033[93m[ROBOTWIN_FORCE_SEED] single-pass collect with seed {try_seed} "
                f"(plan + save_data, no replay phase)\033[0m"
            )
            collect_args = dict(args)
            collect_args["need_plan"] = True
            collect_args["save_data"] = True
            collect_args["render_freq"] = 0
            try:
                TASK_ENV.setup_demo(now_ep_num=0, seed=try_seed, **collect_args)
                info = TASK_ENV.play_once()
                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode 0 success! (seed = {try_seed})")
                    info_file_path = os.path.join(args["save_path"], "scene_info.json")
                    info_db = {}
                    if os.path.exists(info_file_path):
                        with open(info_file_path, "r", encoding="utf-8") as file:
                            info_db = json.load(file)
                    info_db["episode_0"] = info
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump(info_db, file, ensure_ascii=False, indent=4)
                    TASK_ENV.merge_pkl_to_hdf5_video()
                    TASK_ENV.close_env(clear_cache=True)
                    if args.get("render_freq"):
                        TASK_ENV.viewer.close()
                    seed_list.append(try_seed)
                    suc_num = 1
                    _write_seed_file()
                    return True
                why = []
                if not TASK_ENV.plan_success:
                    why.append("plan_success=False")
                if not TASK_ENV.check_success():
                    why.append("check_success=False")
                detail = f" ({', '.join(why)})" if why else ""
                print(f"simulate data episode 0 fail! (seed = {try_seed}){detail}")
                fail_num += 1
                TASK_ENV.close_env()
                if args.get("render_freq"):
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode 0 fail! (seed = {try_seed})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()
                if args.get("render_freq"):
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                print(" -------------")
                print(f"simulate data episode 0 fail! (seed = {try_seed})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()
                if args.get("render_freq"):
                    TASK_ENV.viewer.close()
                time.sleep(1)
            return False

        if forced_seed is not None:
            if not _single_pass_metadata_collect(forced_seed):
                print(f"\033[91m[ROBOTWIN_FORCE_SEED] failed for seed {forced_seed}\033[0m")
                sys.exit(1)
        else:
            while suc_num < args["episode_num"]:
                if epid in bad_seeds:
                    print(f"skip seed {epid} (blacklisted)")
                    epid += 1
                    continue

                if _attempt_seed(epid, suc_num):
                    suc_num += 1

                # Single→Multi fallback: if 0 success after 5 failures, try multi
                if suc_num == 0 and fail_num >= 5:
                    raw = os.environ.get("ROBOTWIN_FORCE_ARMS_JSON", "").strip()
                    if raw:
                        try:
                            arms_cfg = json.loads(raw)
                            if isinstance(arms_cfg, str):
                                arms_cfg = {"mode": "single", "primary_arm": arms_cfg}
                            if isinstance(arms_cfg, dict) and arms_cfg.get("mode") == "single":
                                print(
                                    f"\n\033[93m[ARMS FALLBACK] single mode failed {fail_num} times, "
                                    "switching to multi\033[0m"
                                )
                                os.environ["ROBOTWIN_FORCE_ARMS_JSON"] = json.dumps(
                                    {"mode": "multi"}, ensure_ascii=False
                                )
                                fallback_marker = os.path.join(
                                    args["save_path"], ".arms_fallback_single_to_multi"
                                )
                                with open(fallback_marker, "w") as fm:
                                    fm.write("1")
                                bad_seeds.clear()
                                if os.path.exists(bad_path):
                                    os.remove(bad_path)
                                fail_num = 0
                        except Exception:
                            pass

                epid += 1
                _write_seed_file()

            if suc_num < args["episode_num"]:
                print(
                    f"\033[91mSeed collection incomplete: {suc_num}/{args['episode_num']} "
                    f"after {fail_num} failure(s)\033[0m"
                )
                sys.exit(1)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries ", end="")
        if bad_seeds and forced_seed is None:
            print(f"(blacklist: {len(bad_seeds)} seeds in {bad_path})")
        else:
            print()
        print()
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        if forced_seed is not None:
            print(
                "\033[93m[ROBOTWIN_FORCE_SEED] skip replay-phase data collection "
                "(already saved in single-pass)\033[0m\n"
            )
            command = (
                f"cd description && bash gen_episode_instructions.sh "
                f"{args['task_name']} {args['task_config']} {args['language_num']}"
            )
            os.system(command)
            return

        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
