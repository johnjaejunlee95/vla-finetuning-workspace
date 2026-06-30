import csv
import gc
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["absl_logging_verbosity"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import draccus
import numpy as np
import pandas as pd
import torch

sys.path.append("../../LIBERO-original")
sys.path.append("../")

relative_new_path = "../../LIBERO-original"
relative_config_path = "../../LIBERO-original/libero/"
os.environ["PYTHONPATH"] = os.path.abspath(relative_new_path)
os.environ["LIBERO_CONFIG_PATH"] = os.path.abspath(relative_config_path)

from libero.libero import benchmark
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from libero_utils import (
    create_clean_summary,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    inject_noise2action,
    quat2axisangle,
    save_rollout_video,
)
from openvla_utils import get_processor
from robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

EPISODE_RESULT_COLUMNS = ["episode_name", "success counts", "total_counts", "success_rate"]
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 260,
    "libero_goal": 280,
    "libero_10": 520,
    "libero_90": 400,
}


def format_duration(seconds):
    if seconds is None:
        return "unknown"

    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimate_time_left(elapsed_seconds, completed, total):
    if completed <= 0:
        return None

    remaining = total - completed
    if remaining <= 0:
        return 0

    return elapsed_seconds / completed * remaining


def build_observation(obs, image):
    return {
        "full_image": image,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = "./"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    vla_path: str = "openvla/openvla-7b"
    center_crop: bool = True

    prompt_file: str = "original"
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    sam_type: str = "SAM"
    noise_type: Optional[str] = None

    noise_ratio: float = 0.0
    seed: int = 1122


def save_results(payload, dirs, cfg):
    meta = payload["meta"]
    success = "True" if meta["success"] else "False"
    episode_name = meta["episode_name"]
    filename_base = f"{meta['ver']}_{cfg.task_suite_name}_{episode_name}_success-{success}"

    save_rollout_video(
        rollout_dir=f"results/rollouts/{cfg.task_suite_name}/{meta['ver']}",
        rollout_images=payload["video_frames"],
        idx=episode_name,
        success=meta["success"],
        task_description=meta["desc"],
    )

    torch.save(torch.as_tensor(payload["action"]), f"{dirs['action']}/{filename_base}.pth")
    torch.save(torch.as_tensor(payload["eef"]), f"{dirs['eef']}/{filename_base}.pth")
    torch.save(torch.as_tensor(payload["joint"]), f"{dirs['joint']}/{filename_base}.pth")
    torch.save(torch.as_tensor(payload["logits"]), f"{dirs['logits']}/{filename_base}.pth")


def initialize_csv(csv_path, columns):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, row, columns):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def initialize_episode_results_csv(csv_path):
    initialize_csv(csv_path, EPISODE_RESULT_COLUMNS)


def append_episode_result_csv(csv_path, episode_name, success_counts, total_counts):
    success_rate = success_counts / total_counts if total_counts else 0.0
    row = {
        "episode_name": episode_name,
        "success counts": success_counts,
        "total_counts": total_counts,
        "success_rate": success_rate,
    }
    append_csv_row(csv_path, row, EPISODE_RESULT_COLUMNS)


@draccus.wrap()
def main(cfg: GenerateConfig):
    random_seeds = cfg.seed
    set_seed_everywhere(random_seeds)

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    else:
        major = None
    attn_impl = "sdpa" if major != 8 else "flash_attention_2"
    cfg.attn_impl = attn_impl

    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)

    if cfg.model_family == "openvla":
        target_key = f"{cfg.unnorm_key}_no_noops"
        if cfg.unnorm_key not in model.norm_stats and target_key in model.norm_stats:
            cfg.unnorm_key = target_key
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found!"
        processor = get_processor(cfg)
    else:
        processor = None

    base_save_dir = "results/trajectories"
    save_dirs = {
        "action": f"{base_save_dir}/{cfg.task_suite_name}/{cfg.sam_type}/action_history",
        "eef": f"{base_save_dir}/{cfg.task_suite_name}/{cfg.sam_type}/eefpos_history",
        "joint": f"{base_save_dir}/{cfg.task_suite_name}/{cfg.sam_type}/jointpos_history",
        "logits": f"{base_save_dir}/{cfg.task_suite_name}/{cfg.sam_type}/logits_history",
        "metrics": f"results/metrics/{cfg.task_suite_name}",
    }

    for dir_type, path in save_dirs.items():
        if os.path.isdir(path):
            if dir_type == "metrics":
                continue
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    max_steps_limit = MAX_STEPS_BY_SUITE.get(cfg.task_suite_name, 400)
    print(f"\n===== Starting trial (seed={random_seeds}) =====")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    resize_size = get_image_resize_size(cfg)
    total_planned_episodes = num_tasks_in_suite * cfg.num_trials_per_task
    print(f"Planned episodes: {total_planned_episodes}")

    total_episodes, total_successes = 0, 0
    all_trial_results = []
    episode_csv_path = f"{save_dirs['metrics']}/{cfg.sam_type}_episodes.csv"
    initialize_episode_results_csv(episode_csv_path)

    if cfg.prompt_file != "original":
        with open(f"prompts/{cfg.prompt_file}.json", "r") as f:
            prompts = json.load(f)

    model.eval()
    overall_start_time = time.monotonic()
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        if cfg.prompt_file != "original":
            if cfg.task_suite_name in prompts and task_id < len(prompts[cfg.task_suite_name]):
                modified_task_description = prompts[cfg.task_suite_name][task_id]["new_prompt"]
                if not modified_task_description:
                    print(f"Warning: Task {task_id} has an empty new_prompt; using original task description.")
                else:
                    print(f"Using modified task description: \n\t{task_description}->\n\t{modified_task_description}")
                    task_description = modified_task_description
            else:
                print(
                    f"Warning: Suite {cfg.task_suite_name} task {task_id} has no prompt in "
                    f"{cfg.prompt_file}; using original task description."
                )

        task_episodes, task_successes = 0, 0
        print(f"\nTask {task_id + 1}/{num_tasks_in_suite}: {task_description}")

        for trial_index in range(cfg.num_trials_per_task):
            real_random_seed = np.random.randint(0, 1000)

            set_seed_everywhere(real_random_seed)
            env.seed(real_random_seed)
            env.reset()
            obs = env.set_init_state(initial_states[trial_index])

            action_history, eef_pos_history, joint_pos_history, video_frames, logits_history = [], [], [], [], []

            t = 0
            done = False
            episode_name = f"task{task_id + 1}_trial{trial_index + 1}"
            print(
                f"Episode {total_episodes + 1}/{total_planned_episodes} "
                f"({episode_name})..."
            )
            total_allowed_steps = max_steps_limit + cfg.num_steps_wait

            while t < total_allowed_steps:
                if t < cfg.num_steps_wait:
                    action = get_libero_dummy_action()
                    obs, _, _, _ = env.step(action)
                    t += 1
                    continue

                original_img = get_libero_image(obs, resize_size)

                observation = build_observation(obs, original_img)

                raw_action, logits, _ = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                )

                if cfg.noise_ratio > 0.0:
                    raw_action = inject_noise2action(raw_action, noise_std=cfg.noise_ratio)

                action = normalize_gripper_action(raw_action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                raw_action = raw_action.tolist() if isinstance(raw_action, np.ndarray) else raw_action

                obs, _, done, _ = env.step(action)

                action_history.append(raw_action)
                eef_pos_history.append(obs["robot0_eef_pos"])
                joint_pos_history.append(obs["robot0_joint_pos"])
                if torch.is_tensor(logits):
                    logits_history.append(logits.detach().cpu().numpy()[0])
                else:
                    logits_history.append(np.asarray(logits)[0])

                video_frames.append(original_img)

                t += 1
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1
            success_flag = 1 if done else 0

            all_trial_results.append(
                {
                    "task_id": task_id + 1,
                    "task_description": task_description,
                    "trial_index": trial_index + 1,
                    "success": success_flag,
                    "steps_taken": t,
                }
            )

            task_rate = task_successes / task_episodes
            total_rate = total_successes / total_episodes
            overall_elapsed = time.monotonic() - overall_start_time
            overall_left = estimate_time_left(
                overall_elapsed,
                total_episodes,
                total_planned_episodes,
            )
            all_trial_results[-1].update(
                {
                    "total_elapsed_sec": overall_elapsed,
                    "total_time_left_sec": overall_left,
                }
            )
            print(
                f"Result: {episode_name} | {'Success' if done else 'Fail'} "
                f"| Task Success/Total: {task_successes}/{task_episodes} ({task_rate*100:.1f}%) "
                f"| Total Success/Total: {total_successes}/{total_episodes} ({total_rate*100:.1f}%) "
                f"| Total elapsed/left: {format_duration(overall_elapsed)}/{format_duration(overall_left)}"
            )

            logits_array = np.asarray(logits_history)
            action_logits = logits_array[:, :, -256:] if logits_array.ndim >= 3 else logits_array

            save_data_payload = {
                "action": np.asarray(action_history),
                "eef": np.asarray(eef_pos_history),
                "joint": np.asarray(joint_pos_history),
                "logits": action_logits,
                "video_frames": video_frames,
                "meta": {
                    "success": done,
                    "desc": task_description,
                    "ver": cfg.sam_type,
                    "episode_name": episode_name,
                },
            }

            save_results(save_data_payload, save_dirs, cfg)

        append_episode_result_csv(
            episode_csv_path,
            task_description,
            task_successes,
            task_episodes,
        )

        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"Final Task Success/Total: {task_successes}/{task_episodes} "
            f"| Success Rate: {task_successes / task_episodes:.4f}"
        )

    append_episode_result_csv(
        episode_csv_path,
        "Total",
        total_successes,
        total_episodes,
    )
    print(
        f"Final Total Success/Total: {total_successes}/{total_episodes} "
        f"| Success Rate: {total_successes / total_episodes:.4f}"
    )

    trial_df = pd.DataFrame(all_trial_results)
    df_success_clean = create_clean_summary(trial_df, value_col="success")
    df_success_clean.to_csv(f"{save_dirs['metrics']}/{cfg.sam_type}_summary.csv")


if __name__ == "__main__":
    main()
