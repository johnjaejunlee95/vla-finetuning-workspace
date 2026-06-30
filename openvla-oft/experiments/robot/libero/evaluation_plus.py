import csv
import gc
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["absl_logging_verbosity"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

sys.path.append("../../../LIBERO-plus")
sys.path.append("../../")

relative_new_path = "../../../LIBERO-plus"
relative_config_path = "../../../LIBERO-plus/libero/"
os.environ["PYTHONPATH"] = os.path.abspath(relative_new_path)
os.environ["LIBERO_CONFIG_PATH"] = os.path.abspath(relative_config_path)

import draccus
import numpy as np
import pandas as pd
import torch
from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


def get_task_suite_name(cfg) -> str:
    return cfg.task_suite_name.value if isinstance(cfg.task_suite_name, TaskSuite) else cfg.task_suite_name


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 240,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 450,
    TaskSuite.LIBERO_90: 400,
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TRIAL_RESULT_COLUMNS = [
    "task_suite",
    "save_tag",
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


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value  # Task suite
    prompt_file: str = "original"                    # Prompt JSON name, or "original"
    task_categories: Optional[str] = None            # Comma-separated LIBERO-plus categories to evaluate
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    noise_ratio: float = 0.0                         # Optional Gaussian action noise std

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_tag: str = "openvla_oft"                    # Folder/file tag for saved trajectories and metrics

    is_rollout: bool = False                         # Whether to save rollout videos
    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    task_suite_name = get_task_suite_name(cfg)
    assert task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = get_task_suite_name(cfg)

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def parse_task_categories(task_categories):
    if not task_categories:
        return None

    return [category.strip() for category in task_categories.split(",") if category.strip()]


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    run_id = f"EVAL-{get_task_suite_name(cfg)}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    log_message(f"Logging to local log file: {local_log_filepath}", log_file)

    return log_file


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


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
        rows.append(
            {
                "scope": scope,
                "group": group,
                "num_trials": len(df),
                "num_successes": int(df["success"].sum()),
                "success_rate": float(df["success"].mean()),
                "avg_steps": float(df["steps_taken"].mean()),
                "median_steps": float(df["steps_taken"].median()),
                "min_steps": int(df["steps_taken"].min()),
                "max_steps": int(df["steps_taken"].max()),
            }
        )

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


def get_result_table_paths(metrics_dir, run_name):
    os.makedirs(metrics_dir, exist_ok=True)
    result_path = os.path.join(metrics_dir, f"{run_name}.csv")
    statistics_path = os.path.join(metrics_dir, f"{run_name}_statistics.csv")
    return result_path, statistics_path


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
    if elapsed_seconds is None:
        return None

    if completed <= 0:
        return None

    remaining = total - completed
    if remaining <= 0:
        return 0

    return elapsed_seconds / completed * remaining


def prepare_save_dirs(cfg: GenerateConfig):
    save_root = Path("results")
    task_suite_name = get_task_suite_name(cfg)
    save_dirs = {
        "action": save_root / "trajectories" / task_suite_name / cfg.save_tag / "action_history",
        "eef": save_root / "trajectories" / task_suite_name / cfg.save_tag / "eefpos_history",
        "joint": save_root / "trajectories" / task_suite_name / cfg.save_tag / "jointpos_history",
        "action_chunks": save_root / "trajectories" / task_suite_name / cfg.save_tag / "action_chunks_history",
        "metrics": save_root / "metrics" / task_suite_name,
    }

    for dir_type, path in save_dirs.items():
        if path.is_dir() and dir_type in {"action", "eef", "joint", "action_chunks"}:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    return save_dirs


def save_results(payload, save_dirs, cfg, log_file=None):
    meta = payload["meta"]
    success = "True" if meta["success"] else "False"
    episode_name = meta["episode_name"]
    filename_base = f"{cfg.save_tag}_{get_task_suite_name(cfg)}_{episode_name}_success-{success}"

    artifact_map = {
        "action": "action",
        "eef": "eef",
        "joint": "joint",
        "action_chunks": "action_chunks",
    }
    for payload_key, dir_key in artifact_map.items():
        save_path = save_dirs[dir_key] / f"{filename_base}.pth"
        torch.save(torch.as_tensor(payload[payload_key]), save_path)


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


def create_clean_summary(df, value_col, index_col="trial_index", col_col="task_description"):
    pivot_df = df.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="first")
    pivot_df = pivot_df.sort_index(axis=1)
    pivot_df["Average"] = pivot_df.sum(axis=1).round(4)

    stats = pivot_df.agg(["sum", "std"]).round(4)
    stats.index = ["Task Average (Sum)", "Task Std"]

    return pd.concat([pivot_df, stats])


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def inject_noise2action(action, noise_std=0.05, clip_trans=2.0, clip_rot=2.0):
    action = np.asarray(action).copy()
    assert action.shape[-1] == 7, f"Expected last dim 7, got {action.shape[-1]}"

    trans = action[..., :3]
    rot = action[..., 3:6]

    action[..., :3] = np.clip(trans + np.random.randn(*trans.shape) * noise_std, -clip_trans, clip_trans)
    action[..., 3:6] = np.clip(rot + np.random.randn(*rot.shape) * noise_std, -clip_rot, clip_rot)
    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        log_message(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
            f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
            "both speed and success rate), we recommend executing the full action chunk.",
            log_file,
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    action_history = []
    action_chunk_history = []
    eef_pos_history = []
    joint_pos_history = []
    max_steps = TASK_MAX_STEPS[TaskSuite(get_task_suite_name(cfg))]

    success = False
    error = None
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                if cfg.noise_ratio > 0.0:
                    actions = inject_noise2action(actions, noise_std=cfg.noise_ratio)
                action_chunk_history.append(np.asarray(actions))
                action_queue.extend(actions)

            raw_action = np.asarray(action_queue.popleft())
            action = process_action(raw_action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())
            action_history.append(raw_action.copy())
            eef_pos_history.append(np.asarray(obs["robot0_eef_pos"]).copy())
            joint_pos_history.append(np.asarray(obs["robot0_joint_pos"]).copy())
            t += 1
            if done:
                success = True
                break

    except Exception as e:
        error = str(e)
        log_message(f"Episode error: {e}", log_file)

    return {
        "success": success,
        "video_frames": replay_images,
        "action": np.asarray(action_history),
        "eef": np.asarray(eef_pos_history),
        "joint": np.asarray(joint_pos_history),
        "action_chunks": np.asarray(action_chunk_history),
        "steps_taken": t,
        "error": error,
    }


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    num_tasks: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    total_planned_episodes=0,
    overall_start_time=None,
    save_dirs=None,
    episode_csv_path=None,
    result_path=None,
    all_trial_results=None,
    prompts=None,
    log_file=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)
    task_metadata = task_suite.get_task_metadata(task_id) if hasattr(task_suite, "get_task_metadata") else {}
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    if prompts is not None:
        task_suite_name = get_task_suite_name(cfg)
        if task_suite_name in prompts and task_id < len(prompts[task_suite_name]):
            modified_task_description = prompts[task_suite_name][task_id]["new_prompt"]
            if not modified_task_description:
                log_message(f"Task {task_id} has an empty new_prompt, using original task description.", log_file)
            else:
                log_message(
                    f"Using modified task description: {task_description} -> {modified_task_description}",
                    log_file,
                )
                task_description = modified_task_description
        else:
            log_message(
                f"Suite {task_suite_name} task {task_id} has no prompt in {cfg.prompt_file}, "
                "using original task description.",
                log_file,
            )

    task_episodes, task_successes = 0, 0
    task_category = task_metadata.get("category", "Unclassified")
    task_name = getattr(task, "name", f"task_{task_id + 1}")
    log_message(f"\nTask {task_id + 1}/{num_tasks} [{task_category}]: {task_description}", log_file)

    try:
        for episode_idx in range(cfg.num_trials_per_task):
            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[episode_idx]
            else:
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                    continue

                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

            episode_name = f"task{task_id + 1}_trial{episode_idx + 1}"
            log_message(
                f"Episode {total_episodes + 1}/{total_planned_episodes} [{task_category}] "
                f"task_id={task_id + 1} trial={episode_idx + 1}: {task_description}",
                log_file,
            )

            episode_result = run_episode(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                initial_state,
                log_file,
            )
            success = episode_result["success"]

            task_episodes += 1
            total_episodes += 1
            success_flag = 1 if success else 0
            if success:
                task_successes += 1
                total_successes += 1

            total_success_rate = total_successes / total_episodes if total_episodes else 0.0
            overall_elapsed = time.monotonic() - overall_start_time if overall_start_time is not None else None
            overall_left = estimate_time_left(overall_elapsed, total_episodes, total_planned_episodes)
            result_row = {
                "task_suite": get_task_suite_name(cfg),
                "save_tag": cfg.save_tag,
                "noise_ratio": cfg.noise_ratio,
                "seed": cfg.seed,
                "task_id": task_id + 1,
                "task_name": task_name,
                "task_description": task_description,
                "category": task_metadata.get("category"),
                "difficulty_level": task_metadata.get("difficulty_level"),
                "trial_index": episode_idx + 1,
                "episode_name": episode_name,
                "success": success_flag,
                "steps_taken": episode_result["steps_taken"],
                "total_elapsed_sec": overall_elapsed,
                "total_time_left_sec": overall_left,
                "error": episode_result["error"],
            }
            all_trial_results.append(result_row)
            append_csv_row(result_path, result_row, TRIAL_RESULT_COLUMNS)
            append_episode_result_csv(episode_csv_path, episode_name, success_flag, 1)

            log_message(
                f"Result: {episode_name} | {'Success' if success else 'Fail'} "
                f"| Total Success/Total: {total_successes}/{total_episodes} ({total_success_rate*100:.1f}%) "
                f"| steps={episode_result['steps_taken']} "
                f"| Total elapsed/left: {format_duration(overall_elapsed)}/{format_duration(overall_left)}",
                log_file,
            )

            save_results(
                {
                    "action": episode_result["action"],
                    "eef": episode_result["eef"],
                    "joint": episode_result["joint"],
                    "action_chunks": episode_result["action_chunks"],
                    "video_frames": episode_result["video_frames"],
                    "meta": {
                        "success": success,
                        "desc": task_description,
                        "episode_name": episode_name,
                    },
                },
                save_dirs,
                cfg,
                log_file,
            )
    finally:
        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0

    if cfg.num_trials_per_task > 1:
        log_message(
            f"Final Task Success/Total: {task_successes}/{task_episodes} "
            f"| Success Rate: {task_success_rate:.4f}",
            log_file,
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.eval()
    for module in (action_head, proprio_projector, noisy_action_projector):
        if module is not None:
            module.eval()

    resize_size = get_image_resize_size(cfg)
    task_suite_name = get_task_suite_name(cfg)
    run_name = cfg.save_tag
    log_file = setup_logging(cfg)
    save_dirs = prepare_save_dirs(cfg)
    episode_csv_path = save_dirs["metrics"] / f"{cfg.save_tag}_episodes.csv"
    summary_csv_path = save_dirs["metrics"] / f"{cfg.save_tag}_summary.csv"
    initialize_episode_results_csv(episode_csv_path)
    result_path, statistics_path = get_result_table_paths(save_dirs["metrics"], run_name)
    initialize_csv(result_path, TRIAL_RESULT_COLUMNS)

    task_categories = parse_task_categories(cfg.task_categories)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name](task_categories=task_categories)
    num_tasks = task_suite.n_tasks
    total_planned_episodes = num_tasks * cfg.num_trials_per_task

    log_message(f"\n===== Starting trial (seed={cfg.seed}) =====", log_file)
    log_message(f"Run name: {run_name}", log_file)
    log_message(f"Task suite: {task_suite_name}", log_file)
    if task_categories:
        log_message(f"Evaluating categories: {task_categories}", log_file)
        log_message(f"Selected {num_tasks} tasks from {task_suite_name}", log_file)
    log_message(f"Planned episodes: {total_planned_episodes}", log_file)
    log_message(f"Episode result table: {episode_csv_path}", log_file)
    log_message(f"Detailed result table: {result_path}", log_file)

    prompts = None
    if cfg.prompt_file != "original":
        with open(f"prompts/{cfg.prompt_file}.json", "r") as f:
            prompts = json.load(f)

    total_episodes, total_successes = 0, 0
    all_trial_results = []
    overall_start_time = time.monotonic()
    try:
        for task_id in range(num_tasks):
            total_episodes, total_successes = run_task(
                cfg,
                task_suite,
                task_id,
                num_tasks,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                total_planned_episodes,
                overall_start_time,
                save_dirs,
                episode_csv_path,
                result_path,
                all_trial_results,
                prompts,
                log_file,
            )
    finally:
        if log_file:
            log_file.flush()

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    append_episode_result_csv(episode_csv_path, "Total", total_successes, total_episodes)
    log_message(
        f"Final Total Success/Total: {total_successes}/{total_episodes} "
        f"| Success Rate: {final_success_rate:.4f}",
        log_file,
    )
    log_message(f"Saved episode metrics to {episode_csv_path}", log_file)

    if all_trial_results:
        trial_df = pd.DataFrame(all_trial_results)
        df_success_clean = create_clean_summary(trial_df, value_col="success", col_col="episode_name")
        df_success_clean.to_csv(summary_csv_path)
        log_message(f"Saved summary metrics to {summary_csv_path}", log_file)

    write_statistics_table(all_trial_results, statistics_path)
    log_message(f"Finished. Episode result table: {episode_csv_path}", log_file)
    log_message(f"Finished. Clean summary table: {summary_csv_path}", log_file)
    log_message(f"Finished. Overall result table: {result_path}", log_file)
    log_message(f"Finished. Statistics table: {statistics_path}", log_file)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
