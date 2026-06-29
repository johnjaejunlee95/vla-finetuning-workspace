import collections
import csv
import dataclasses
import json
import logging
import math
import pathlib
import shutil

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    results_out_path: str = "data/libero/results"  # Path to save rollout arrays and success metrics
    methods: str = "pi0-LIBERO-LoRA-baseline"
    seeds: int = 1234  # Master seed for multi-seed environment evaluation
    trial_num: int = 1  # Number of environment-seed trials to run


def _resolve_seed_list(args: Args):
    trial_num = int(args.trial_num)
    if trial_num <= 0:
        raise ValueError(f"trial_num must be positive, got {trial_num}")

    # Keep env seeds in [0, 9998] so every seed is strictly less than 9999.
    seed_span = 9999
    if trial_num > seed_span:
        raise ValueError(f"trial_num={trial_num} exceeds available unique env seeds (<9999): {seed_span}")

    master_seed_generator = np.random.SeedSequence(int(args.seeds))
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


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if task_suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if task_suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if task_suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if task_suite_name == "libero_90":
        return 400  # longest training demo has 373 steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _create_trial_logger(results_root: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger(f"libero_eval.trial.{results_root.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    log_path = results_root / "episode_results.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _close_trial_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _recreate_dir(path: pathlib.Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _eval_single_seed(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    eval_seed: int,
    trial_index: int,
    total_trials: int,
    base_results_root: pathlib.Path,
) -> dict:
    np.random.seed(eval_seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    trial_tag = f"trial_{trial_index + 1:03d}_seed_{int(eval_seed)}"
    results_root = base_results_root / trial_tag
    trajectory_out_path = results_root / "trajectories"
    metrics_out_path = results_root / "metrics"

    _recreate_dir(results_root)
    _recreate_dir(trajectory_out_path)
    _recreate_dir(metrics_out_path)

    arguments_path = results_root / "arguments.json"
    arguments = dataclasses.asdict(args)
    arguments["env_seed"] = int(eval_seed)
    arguments["trial_index"] = int(trial_index + 1)
    arguments["total_trials"] = int(total_trials)
    arguments["trial_tag"] = trial_tag
    with arguments_path.open("w", encoding="utf-8") as f:
        json.dump(arguments, f, indent=2)

    trial_logger = _create_trial_logger(results_root)

    max_steps = _get_max_steps(args.task_suite_name)

    total_episodes, total_successes = 0, 0
    task_rows = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, eval_seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            action_plan = collections.deque()
            episode_real_actions = []
            episode_eef = []

            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        infer_result = client.infer(element)
                        action_chunk = np.asarray(infer_result["actions"])

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    obs, _, done, _ = env.step(action.tolist())
                    episode_real_actions.append(np.asarray(action, dtype=np.float32))
                    episode_eef.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    trial_logger.exception(
                        "episode_exception | task_id=%d | task_description=%s | episode=%d | t=%d | message=%s",
                        task_id + 1,
                        task_description,
                        episode_idx + 1,
                        t,
                        str(e),
                    )
                    break

            task_episodes += 1
            total_episodes += 1
            task_success_rate = round(float(task_successes) / float(task_episodes), 4) if task_episodes else 0.0
            total_success_rate = round(float(total_successes) / float(total_episodes), 4) if total_episodes else 0.0
            trial_logger.info(
                "episode_result | task_id=%d | episode=%d | steps_taken=%d | task success=%d | total success=%d | task_success_rate=%.4f | total_success_rate=%.4f",
                task_id + 1,
                episode_idx + 1,
                int(t),
                task_successes,
                total_successes,
                float(task_success_rate),
                float(total_success_rate),
            )

            success_label = "success" if done else "failure"
            trajectory_filename = f"rollout_task{task_id + 1}_episode{episode_idx + 1}_{success_label}.pt"
            trajectory_path = trajectory_out_path / trajectory_filename
            torch.save(
                {
                    "real_actions": torch.as_tensor(np.asarray(episode_real_actions, dtype=np.float32)),
                    "eef": torch.as_tensor(np.asarray(episode_eef, dtype=np.float32)),
                    "steps_taken": int(t),
                    "task_description": task_description,
                    "env_seed": int(eval_seed),
                    "seed_trial_index": int(trial_index + 1),
                },
                trajectory_path,
            )

        current_task_success_rate = round(float(task_successes) / float(task_episodes), 4) if task_episodes else 0.0
        task_rows.append(
            {
                "task_id": task_id + 1,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": current_task_success_rate,
            }
        )
        env.close()

    task_csv_path = metrics_out_path / "task_success_rates.csv"
    with task_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "task_description", "episodes", "successes", "success_rate"],
        )
        writer.writeheader()
        writer.writerows(task_rows)
        writer.writerow(
            {
                "task_id": "overall",
                "task_description": "overall",
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": round(float(total_successes) / float(total_episodes), 4) if total_episodes else 0.0,
            }
        )

    summary_path = metrics_out_path / "summary.json"
    summary = {
        "task_suite_name": args.task_suite_name,
        "env_seed": int(eval_seed),
        "seed_trial_index": int(trial_index + 1),
        "total_seed_trials": int(total_trials),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": round(float(total_successes) / float(total_episodes), 4) if total_episodes else 0.0,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    trial_logger.info(
        "trial_summary | total_episodes=%d | total_successes=%d | total_success_rate=%.4f | summary_path=%s",
        int(total_episodes),
        int(total_successes),
        float(summary["total_success_rate"]),
        str(summary_path),
    )
    _close_trial_logger(trial_logger)

    return {
        "trial_index": int(trial_index + 1),
        "env_seed": int(eval_seed),
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": summary["total_success_rate"],
        "results_dir": str(results_root),
        "metrics_summary": str(summary_path),
    }


def eval_libero(args: Args) -> None:
    seed_list = _resolve_seed_list(args)
    base_results_root = pathlib.Path(args.results_out_path) / args.task_suite_name / args.methods
    _recreate_dir(base_results_root)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    seed_rows = []
    for trial_index, eval_seed in enumerate(seed_list):
        logging.info(
            "Running seed trial %d/%d with env_seed=%d",
            trial_index + 1,
            len(seed_list),
            int(eval_seed),
        )
        result = _eval_single_seed(
            args=args,
            client=client,
            eval_seed=eval_seed,
            trial_index=trial_index,
            total_trials=len(seed_list),
            base_results_root=base_results_root,
        )
        seed_rows.append(result)

    seed_csv_path = base_results_root / "seed_summary.csv"
    with seed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_index",
                "env_seed",
                "total_episodes",
                "total_successes",
                "total_success_rate",
                "results_dir",
                "metrics_summary",
            ],
        )
        writer.writeheader()
        writer.writerows(seed_rows)

    success_rates = np.asarray([row["total_success_rate"] for row in seed_rows], dtype=np.float32)
    aggregate = {
        "master_seed": int(args.seeds),
        "trial_num": int(args.trial_num),
        "env_seeds": [int(seed) for seed in seed_list],
        "seed_summary_csv": str(seed_csv_path),
        "mean_success_rate": round(float(np.mean(success_rates)), 4) if len(success_rates) else 0.0,
        "std_success_rate": round(float(np.std(success_rates)), 4) if len(success_rates) else 0.0,
        "per_seed": seed_rows,
    }
    seed_json_path = base_results_root / "seed_summary.json"
    with seed_json_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
