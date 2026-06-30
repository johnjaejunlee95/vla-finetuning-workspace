import csv
import gc
import json
import logging
import os
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

import draccus
import numpy as np
import pandas as pd
import torch

sys.path.append("../../LIBERO-plus")
sys.path.append("../")

relative_new_path = "../../LIBERO-plus"
relative_config_path = "../../LIBERO-plus/libero/"
os.environ["PYTHONPATH"] = os.path.abspath(relative_new_path)
os.environ["LIBERO_CONFIG_PATH"] = os.path.abspath(relative_config_path)

sys.path.append("../")
from libero.libero import benchmark
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from libero_utils import (
    create_clean_summary,
    inject_noise2action,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
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


TRIAL_RESULT_COLUMNS = [
    "task_suite",
    "sam_type",
    "noise_ratio",
    "seed",
    "task_id",
    "task_name",
    "task_description",
    "category",
    "difficulty_level",
    "trial_index",
    "episode_name",
    "success",
    "steps_taken",
    "total_elapsed_sec",
    "total_time_left_sec",
]

EPISODE_RESULT_COLUMNS = ["episode_name", "success counts", "total_counts", "success_rate"]
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 240,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 450,
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


def parse_task_categories(task_categories):
    if not task_categories:
        return None

    return [category.strip() for category in task_categories.split(",") if category.strip()]


def setup_logger(log_dir, suite_name, run_name):
    suite_log_dir = os.path.join(log_dir, suite_name)
    os.makedirs(suite_log_dir, exist_ok=True)
    log_path = os.path.join(suite_log_dir, f"{run_name}.log")

    logger = logging.getLogger("analysisv3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def create_statistics_table(trial_df):
    columns = [
        "scope",
        "group",
        "num_trials",
        "num_successes",
        "success_rate",
        "avg_steps",
        "median_steps",
        "min_steps",
        "max_steps",
    ]
    if len(trial_df) == 0:
        return pd.DataFrame(columns=columns)

    rows = []

    def add_stats(scope, group, df):
        rows.append({
            "scope": scope,
            "group": group,
            "num_trials": len(df),
            "num_successes": int(df["success"].sum()),
            "success_rate": float(df["success"].mean()),
            "avg_steps": float(df["steps_taken"].mean()),
            "median_steps": float(df["steps_taken"].median()),
            "min_steps": int(df["steps_taken"].min()),
            "max_steps": int(df["steps_taken"].max()),
        })

    add_stats("overall", "all", trial_df)

    for category, group_df in trial_df.groupby("category", dropna=False):
        category = "Unclassified" if pd.isna(category) else category
        add_stats("category", category, group_df)

    for difficulty_level, group_df in trial_df.groupby("difficulty_level", dropna=False):
        difficulty_level = "unknown" if pd.isna(difficulty_level) else difficulty_level
        add_stats("difficulty_level", difficulty_level, group_df)

    for (task_id, task_name), group_df in trial_df.groupby(["task_id", "task_name"], dropna=False):
        add_stats("task", f"{task_id}:{task_name}", group_df)

    return pd.DataFrame(rows, columns=columns)


def get_result_table_paths(result_dir, suite_name, run_name):
    suite_result_dir = os.path.join(result_dir, suite_name)
    os.makedirs(suite_result_dir, exist_ok=True)
    result_path = os.path.join(suite_result_dir, f"{run_name}.csv")
    statistics_path = os.path.join(suite_result_dir, f"{run_name}_statistics.csv")
    return result_path, statistics_path


def initialize_csv(csv_path, columns):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, row, columns):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerow(row)


def write_statistics_table(trial_results, statistics_path):
    trial_df = pd.DataFrame(trial_results)
    create_statistics_table(trial_df).to_csv(statistics_path, index=False)


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


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = "/nfs3/jjlee/datasets/openvla/ckpt2/partial_sam_libero_spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    vla_path: str = "openvla/openvla-7b"
    center_crop: bool = True

    prompt_file: str = "original"
    task_suite_name: str = "libero_spatial"
    task_categories: Optional[str] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    sam_type: str = "SAM"
    noise_type: Optional[str] = None

    noise_ratio: float = 0.0
    seed: int = 1122
    result_dir: str = "savel_results3/results"
    log_dir: str = "savel_results3/logs"


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

    run_name = cfg.sam_type
    logger, log_path = setup_logger(cfg.log_dir, cfg.task_suite_name, run_name)

    max_steps_limit = MAX_STEPS_BY_SUITE.get(cfg.task_suite_name, 400)
    logger.info("===== Starting trial (seed=%s) =====", random_seeds)
    logger.info("Run name: %s", run_name)
    logger.info("Log path: %s", log_path)

    task_categories = parse_task_categories(cfg.task_categories)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name](task_categories=task_categories)
    num_tasks_in_suite = task_suite.n_tasks
    resize_size = get_image_resize_size(cfg)
    total_planned_episodes = num_tasks_in_suite * cfg.num_trials_per_task
    if task_categories:
        logger.info("Evaluating categories: %s", task_categories)
        logger.info("Selected %d tasks from %s", num_tasks_in_suite, cfg.task_suite_name)

    total_episodes, total_successes = 0, 0
    all_trial_results = []
    metrics_dir = f"results/metrics/{cfg.task_suite_name}"
    os.makedirs(metrics_dir, exist_ok=True)
    episode_csv_path = f"{metrics_dir}/{cfg.sam_type}_episodes.csv"
    initialize_episode_results_csv(episode_csv_path)
    result_path, statistics_path = get_result_table_paths(
        cfg.result_dir,
        cfg.task_suite_name,
        run_name,
    )
    initialize_csv(result_path, TRIAL_RESULT_COLUMNS)
    logger.info("Planned episodes: %d", total_planned_episodes)
    logger.info("Episode result table: %s", episode_csv_path)
    logger.info("Detailed result table: %s", result_path)

    if cfg.prompt_file != "original":
        with open(f"prompts/{cfg.prompt_file}.json", "r") as f:
            prompts = json.load(f)

    model.eval()
    overall_start_time = time.monotonic()
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        task_metadata = task_suite.get_task_metadata(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        if cfg.prompt_file != "original":
            if cfg.task_suite_name in prompts and task_id < len(prompts[cfg.task_suite_name]):
                modified_task_description = prompts[cfg.task_suite_name][task_id]["new_prompt"]
                if not modified_task_description:
                    logger.warning("Task %d has an empty new_prompt, using original task description.", task_id)
                else:
                    logger.info(
                        "Using modified task description: %s -> %s",
                        task_description,
                        modified_task_description,
                    )
                    task_description = modified_task_description
            else:
                logger.warning(
                    "Suite %s task %d has no prompt in %s, using original task description.",
                    cfg.task_suite_name,
                    task_id,
                    cfg.prompt_file,
                )

        task_episodes, task_successes = 0, 0
        task_category = task_metadata.get("category", "Unclassified")

        for trial_index in range(cfg.num_trials_per_task):
            real_random_seed = np.random.randint(0, 1000)

            set_seed_everywhere(real_random_seed)
            env.seed(real_random_seed)
            env.reset()
            obs = env.set_init_state(initial_states[trial_index])

            t = 0
            done = False
            episode_name = f"task{task_id + 1}_trial{trial_index + 1}"
            logger.info(
                "Episode %d/%d [%s] task_id=%d trial=%d: %s",
                total_episodes + 1,
                total_planned_episodes,
                task_category,
                task_id + 1,
                trial_index + 1,
                task_description,
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

                raw_action, _, _ = get_action(
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

                obs, _, done, _ = env.step(action)

                t += 1
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1
            success_flag = 1 if done else 0
            total_rate = total_successes / total_episodes

            result_row = {
                "task_suite": cfg.task_suite_name,
                "sam_type": cfg.sam_type,
                "noise_ratio": cfg.noise_ratio,
                "seed": random_seeds,
                "task_id": task_id + 1,
                "task_name": task.name,
                "task_description": task_description,
                "category": task_metadata.get("category"),
                "difficulty_level": task_metadata.get("difficulty_level"),
                "trial_index": trial_index + 1,
                "episode_name": episode_name,
                "success": success_flag,
                "steps_taken": t,
            }

            overall_elapsed = time.monotonic() - overall_start_time
            overall_left = estimate_time_left(overall_elapsed, total_episodes, total_planned_episodes)
            result_row.update({
                "total_elapsed_sec": overall_elapsed,
                "total_time_left_sec": overall_left,
            })
            all_trial_results.append(result_row)
            append_csv_row(result_path, result_row, TRIAL_RESULT_COLUMNS)
            append_episode_result_csv(
                episode_csv_path,
                episode_name,
                success_flag,
                1,
            )
            logger.info(
                "Result: %s | %s | Total Success/Total: %d/%d (%.1f%%) | "
                "steps=%d | Total elapsed/left: %s/%s",
                episode_name,
                "Success" if done else "Fail",
                total_successes,
                total_episodes,
                total_rate * 100,
                t,
                format_duration(overall_elapsed),
                format_duration(overall_left),
            )

        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.num_trials_per_task > 1:
            logger.info(
                "Final Task Success/Total: %d/%d | Success Rate: %.4f",
                task_successes,
                task_episodes,
                task_successes / task_episodes,
            )

    append_episode_result_csv(
        episode_csv_path,
        "Total",
        total_successes,
        total_episodes,
    )
    logger.info(
        "Final Total Success/Total: %d/%d | Success Rate: %.4f",
        total_successes,
        total_episodes,
        total_successes / total_episodes,
    )

    trial_df = pd.DataFrame(all_trial_results)
    df_success_clean = create_clean_summary(
        trial_df,
        value_col="success",
        col_col="episode_name",
    )
    summary_path = f"{metrics_dir}/{cfg.sam_type}_summary.csv"
    df_success_clean.to_csv(summary_path)

    write_statistics_table(
        all_trial_results,
        statistics_path,
    )
    logger.info("Finished. Episode result table: %s", episode_csv_path)
    logger.info("Finished. Clean summary table: %s", summary_path)
    logger.info("Finished. Overall result table: %s", result_path)
    logger.info("Finished. Statistics table: %s", statistics_path)


if __name__ == "__main__":
    main()
