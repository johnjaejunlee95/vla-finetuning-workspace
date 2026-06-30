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

sys.path.append("../../../LIBERO-original")
sys.path.append("../../")

relative_new_path = "../../../LIBERO-original"
relative_config_path = "../../../LIBERO-original/libero/"
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
    save_rollout_video,
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


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

EPISODE_RESULT_COLUMNS = ["episode_name", "success counts", "total_counts", "success_rate"]
SEED_SUMMARY_COLUMNS = [
    "seed_trial_index",
    "env_seed",
    "total_episodes",
    "total_successes",
    "total_success_rate",
]


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
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_tag: str = "openvla_oft"                    # Folder/file tag for saved trajectories and metrics
    
    is_rollout: bool = False                         # Whether to save rollout videos
    
    seed: int = 123                                    # Master seed for multi-seed environment evaluation
    trial_num: int = 3                               # Number of environment-seed trials to run

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.trial_num > 0, f"trial_num must be positive, got {cfg.trial_num}"

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


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    # Create run ID
    run_id = f"EVAL-{get_task_suite_name(cfg)}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
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


def resolve_seed_list(cfg: GenerateConfig):
    trial_num = int(cfg.trial_num)
    if trial_num <= 0:
        raise ValueError(f"trial_num must be positive, got {trial_num}")

    seed_span = 9999
    if trial_num > seed_span:
        raise ValueError(f"trial_num={trial_num} exceeds available unique env seeds (<9999): {seed_span}")

    master_seed_generator = np.random.SeedSequence(int(cfg.seed))
    trial_seed_sequences = master_seed_generator.spawn(trial_num)

    used = set()
    seed_list = []
    for seq in trial_seed_sequences:
        raw = int(seq.generate_state(1, dtype=np.uint32)[0])
        candidate = raw % seed_span
        while candidate in used:
            candidate = (candidate + 1) % seed_span
        used.add(candidate)
        seed_list.append(candidate)
    return seed_list


def prepare_save_dirs(cfg: GenerateConfig):
    save_root = Path("results")
    task_suite_name = get_task_suite_name(cfg)
    save_dirs = {
        "action": save_root / "trajectories" / task_suite_name / cfg.save_tag / "action_history",
        "eef": save_root / "trajectories" / task_suite_name / cfg.save_tag / "eefpos_history",
        "joint": save_root / "trajectories" / task_suite_name / cfg.save_tag / "jointpos_history",
        "action_chunks": save_root / "trajectories" / task_suite_name / cfg.save_tag / "action_chunks_history",
        "rollout": save_root / "rollouts" / task_suite_name / cfg.save_tag,
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

    if payload["video_frames"] and cfg.is_rollout:
        save_rollout_video(
            payload["video_frames"],
            idx=episode_name,
            success=meta["success"],
            task_description=meta["desc"],
            log_file=log_file,
            rollout_dir=save_dirs["rollout"],
        )
    else:
        log_message(f"No rollout frames captured for {episode_name}; skipping MP4 save.", log_file)

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
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def initialize_episode_results_csv(csv_path):
    initialize_csv(csv_path, EPISODE_RESULT_COLUMNS)


def initialize_seed_summary_csv(csv_path):
    initialize_csv(csv_path, SEED_SUMMARY_COLUMNS)


def append_episode_result_csv(csv_path, episode_name, success_counts, total_counts):
    success_rate = success_counts / total_counts if total_counts else 0.0
    row = {
        "episode_name": episode_name,
        "success counts": success_counts,
        "total_counts": total_counts,
        "success_rate": success_rate,
    }
    append_csv_row(csv_path, row, EPISODE_RESULT_COLUMNS)


def append_seed_summary_csv(csv_path, seed_summary):
    append_csv_row(csv_path, seed_summary, SEED_SUMMARY_COLUMNS)


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
    all_trial_results=None,
    env_seed=None,
    seed_trial_index=0,
    total_seed_trials=1,
    log_file=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    if env_seed is not None:
        env.seed(int(env_seed))

    task_episodes, task_successes = 0, 0
    log_message(
        f"\nSeed trial {seed_trial_index + 1}/{total_seed_trials} "
        f"(env_seed={env_seed}) | Task {task_id + 1}/{num_tasks}: {task_description}",
        log_file,
    )

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

            episode_name = f"seed{seed_trial_index + 1}_task{task_id + 1}_trial{episode_idx + 1}"
            log_message(
                f"Episode {total_episodes + 1}/{total_planned_episodes} ({episode_name})...",
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
            if success:
                task_successes += 1
                total_successes += 1

            all_trial_results.append(
                {
                    "master_seed": int(cfg.seed),
                    "seed_trial_index": seed_trial_index + 1,
                    "total_seed_trials": total_seed_trials,
                    "env_seed": int(env_seed) if env_seed is not None else None,
                    "task_id": task_id + 1,
                    "task_description": task_description,
                    "trial_index": seed_trial_index * cfg.num_trials_per_task + episode_idx + 1,
                    "episode_index": episode_idx + 1,
                    "success": 1 if success else 0,
                    "steps_taken": episode_result["steps_taken"],
                    "error": episode_result["error"],
                }
            )

            task_success_rate = task_successes / task_episodes if task_episodes else 0.0
            total_success_rate = total_successes / total_episodes if total_episodes else 0.0
            overall_elapsed = time.monotonic() - overall_start_time if overall_start_time is not None else None
            overall_left = estimate_time_left(overall_elapsed, total_episodes, total_planned_episodes)
            all_trial_results[-1].update(
                {
                    "total_elapsed_sec": overall_elapsed,
                    "total_time_left_sec": overall_left,
                }
            )

            log_message(
                f"Result: {episode_name} | {'Success' if success else 'Fail'} "
                f"| Task Success/Total: {task_successes}/{task_episodes} ({task_success_rate*100:.1f}%) "
                f"| Total Success/Total: {total_successes}/{total_episodes} ({total_success_rate*100:.1f}%) "
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

    episode_result_name = f"seed{seed_trial_index + 1}_env{env_seed}_{task_description}"
    append_episode_result_csv(episode_csv_path, episode_result_name, task_successes, task_episodes)

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
    seed_list = resolve_seed_list(cfg)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model.eval()
    for module in (action_head, proprio_projector, noisy_action_projector):
        if module is not None:
            module.eval()

    resize_size = get_image_resize_size(cfg)
    log_file = setup_logging(cfg)
    save_dirs = prepare_save_dirs(cfg)
    episode_csv_path = save_dirs["metrics"] / f"{cfg.save_tag}_episodes.csv"
    summary_csv_path = save_dirs["metrics"] / f"{cfg.save_tag}_summary.csv"
    seed_summary_csv_path = save_dirs["metrics"] / f"{cfg.save_tag}_seed_summary.csv"
    seed_summary_json_path = save_dirs["metrics"] / f"{cfg.save_tag}_seed_summary.json"
    initialize_episode_results_csv(episode_csv_path)
    initialize_seed_summary_csv(seed_summary_csv_path)

    task_suite_name = get_task_suite_name(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    total_planned_episodes = len(seed_list) * num_tasks * cfg.num_trials_per_task

    log_message(f"\n===== Starting evaluation (master_seed={cfg.seed}) =====", log_file)
    log_message(f"Task suite: {task_suite_name}", log_file)
    log_message(f"Seed trials: {len(seed_list)} | Env seeds: {seed_list}", log_file)
    log_message(f"Planned episodes: {total_planned_episodes}", log_file)

    total_episodes, total_successes = 0, 0
    all_trial_results = []
    seed_summaries = []
    overall_start_time = time.monotonic()
    try:
        for seed_trial_index, env_seed in enumerate(seed_list):
            seed_start_episodes = total_episodes
            seed_start_successes = total_successes
            set_seed_everywhere(int(env_seed))
            log_message(
                f"\n===== Starting seed trial {seed_trial_index + 1}/{len(seed_list)} "
                f"(env_seed={env_seed}) =====",
                log_file,
            )
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
                    all_trial_results,
                    int(env_seed),
                    seed_trial_index,
                    len(seed_list),
                    log_file,
                )

            seed_episodes = total_episodes - seed_start_episodes
            seed_successes = total_successes - seed_start_successes
            seed_success_rate = seed_successes / seed_episodes if seed_episodes else 0.0
            seed_summary = {
                "seed_trial_index": seed_trial_index + 1,
                "env_seed": int(env_seed),
                "total_episodes": seed_episodes,
                "total_successes": seed_successes,
                "total_success_rate": seed_success_rate,
            }
            seed_summaries.append(seed_summary)
            append_seed_summary_csv(seed_summary_csv_path, seed_summary)
            log_message(
                f"Final Seed Trial Success/Total: {seed_successes}/{seed_episodes} "
                f"| Success Rate: {seed_success_rate:.4f}",
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
        df_success_clean = create_clean_summary(trial_df, value_col="success")
        df_success_clean.to_csv(summary_csv_path)
        log_message(f"Saved summary metrics to {summary_csv_path}", log_file)

    seed_summary = {
        "master_seed": int(cfg.seed),
        "trial_num": int(cfg.trial_num),
        "env_seeds": [int(seed) for seed in seed_list],
        "seed_summary_csv": str(seed_summary_csv_path),
        "per_seed": seed_summaries,
    }
    with open(seed_summary_json_path, "w") as f:
        json.dump(seed_summary, f, indent=2)
    log_message(f"Saved seed summary to {seed_summary_csv_path}", log_file)
    log_message(f"Saved seed summary metadata to {seed_summary_json_path}", log_file)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
