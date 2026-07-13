import collections
import csv
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
from typing import Optional

# Keep the LIBERO-plus package ahead of any installed original LIBERO package.
LIBERO_PLUS_ROOT = pathlib.Path(__file__).resolve().parents[3] / "libero-env" / "LIBERO-plus"
sys.path.insert(0, str(LIBERO_PLUS_ROOT))

os.environ["PYTHONPATH"] = str(LIBERO_PLUS_ROOT)
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_PLUS_ROOT / "libero")

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_ping_interval: Optional[float] = None
    websocket_ping_timeout: Optional[float] = None
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    prompt_file: str = "original"  # Prompt JSON name, or "original"
    task_categories: Optional[str] = None  # Comma-separated LIBERO-plus categories
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    results_out_path: str = "data/libero-plus/results"  # Path to save rollout arrays and success metrics
    methods: str = "pi0-LIBERO-plus-baseline"
    tags: Optional[str] = None # Tag to append to the results directory name


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 260
    if task_suite_name == "libero_goal":
        return 280
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _parse_task_categories(task_categories: Optional[str]):
    if not task_categories:
        return None
    return [category.strip() for category in task_categories.split(",") if category.strip()]


def _load_prompts(prompt_file: str):
    if prompt_file == "original":
        return None
    with open(f"prompts/{prompt_file}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _category_dir_name(task_categories: Optional[str]) -> str:
    categories = _parse_task_categories(task_categories)
    if not categories:
        return "All Categories"
    return ", ".join(categories)


def _resolve_save_tag(args: Args) -> str:
    if args.tags:
        return args.tags

    categories = _parse_task_categories(args.task_categories)
    if len(categories or []) == 1:
        category = categories[0]
        category_tags = {
            "Background Textures": "background",
            "Camera Viewpoints": "camera",
            "Language Instructions": "language",
            "Light Conditions": "light",
            "Objects Layout": "object",
            "Robot Initial States": "robot_states",
            "Sensor Noise": "sensor",
        }
        if category in category_tags:
            return category_tags[category]
        return category.lower().replace(" ", "_")

    return "all"


def _write_episode_summary(episode_rows, episode_csv_path: pathlib.Path) -> None:
    columns = ["episode_name", "success counts", "total_counts", "success_rate"]
    grouped_rows = []
    rows_by_description = collections.defaultdict(list)
    for row in episode_rows:
        rows_by_description[row["task_description"]].append(row)

    for task_description, records in rows_by_description.items():
        total_counts = len(records)
        success_counts = int(sum(row["success"] for row in records))
        grouped_rows.append(
            {
                "episode_name": task_description,
                "success counts": success_counts,
                "total_counts": total_counts,
                "success_rate": success_counts / total_counts if total_counts else 0.0,
            }
        )

    total_counts = len(episode_rows)
    success_counts = int(sum(row["success"] for row in episode_rows))
    grouped_rows.append(
        {
            "episode_name": "Total",
            "success counts": success_counts,
            "total_counts": total_counts,
            "success_rate": success_counts / total_counts if total_counts else 0.0,
        }
    )

    with episode_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(grouped_rows)


def _append_overall_results(
    args: Args,
    episode_rows,
    overall_results_path: pathlib.Path,
) -> None:
    columns = [
        "task_suite_name",
        "success counts",
        "total_counts",
        "success_rate",
    ]
    total_counts = len(episode_rows)
    success_counts = int(sum(row["success"] for row in episode_rows))
    row = {
        "task_suite_name": args.task_suite_name,
        "success counts": success_counts,
        "total_counts": total_counts,
        "success_rate": success_counts / total_counts if total_counts else 0.0,
    }

    should_write_header = not overall_results_path.exists() or overall_results_path.stat().st_size == 0
    with overall_results_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def _get_task_description(task_suite_name: str, task_id: int, task, prompts) -> str:
    task_description = task.language
    if prompts is None:
        return task_description

    suite_prompts = prompts.get(task_suite_name, [])
    if task_id >= len(suite_prompts):
        logging.warning(
            "Suite %s task %d has no prompt; using the original task description.",
            task_suite_name,
            task_id,
        )
        return task_description

    modified_task_description = suite_prompts[task_id].get("new_prompt", "")
    return modified_task_description or task_description


def _write_statistics_table(episode_rows, statistics_path: pathlib.Path) -> None:
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

    def add_stats(rows, scope, group, records):
        if not records:
            return
        steps = np.asarray([record["steps_taken"] for record in records], dtype=np.float32)
        successes = np.asarray([record["success"] for record in records], dtype=np.float32)
        rows.append(
            {
                "scope": scope,
                "group": group,
                "num_trials": len(records),
                "num_successes": int(successes.sum()),
                "success_rate": float(successes.mean()),
                "avg_steps": float(steps.mean()),
                "median_steps": float(np.median(steps)),
                "min_steps": int(steps.min()),
                "max_steps": int(steps.max()),
            }
        )

    rows = []
    add_stats(rows, "overall", "all", episode_rows)
    for category in sorted({row["category"] for row in episode_rows}):
        add_stats(rows, "category", category, [row for row in episode_rows if row["category"] == category])
    for difficulty in sorted({row["difficulty_level"] for row in episode_rows}):
        add_stats(
            rows,
            "difficulty_level",
            difficulty,
            [row for row in episode_rows if row["difficulty_level"] == difficulty],
        )
    for task_id, task_name in sorted({(row["task_id"], row["task_name"]) for row in episode_rows}):
        add_stats(
            rows,
            "task",
            f"{task_id}:{task_name}",
            [row for row in episode_rows if row["task_id"] == task_id and row["task_name"] == task_name],
        )

    with statistics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _eval_once(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    prompts,
) -> dict:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_categories = _parse_task_categories(args.task_categories)
    task_suite = benchmark_dict[args.task_suite_name](task_categories=task_categories)
    num_tasks_in_suite = task_suite.n_tasks
    trial_logger = logging.getLogger(__name__)
    max_steps = _get_max_steps(args.task_suite_name)
    total_episodes, total_successes = 0, 0
    episode_rows = []

    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        task_metadata = task_suite.get_task_metadata(task_id) if hasattr(task_suite, "get_task_metadata") else {}
        task_name = getattr(task, "name", f"task_{task_id + 1}")
        task_category = task_metadata.get("category") or "Unclassified"
        difficulty_level = task_metadata.get("difficulty_level") or "unknown"
        task_description = _get_task_description(args.task_suite_name, task_id, task, prompts)
        initial_states = task_suite.get_task_init_states(task_id)
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION)

        task_episodes, task_successes = 0, 0
        try:
            for episode_idx in range(args.num_trials_per_task):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                done = False
                action_plan = collections.deque()

                while t < max_steps + args.num_steps_wait:
                    try:
                        # Let objects settle before querying the policy.
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

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
                            assert len(action_chunk) >= args.replan_steps, (
                                f"We want to replan every {args.replan_steps} steps, "
                                f"but policy only predicts {len(action_chunk)} steps."
                            )
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()
                        obs, _, done, _ = env.step(action.tolist())
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
                task_success_rate = task_successes / task_episodes if task_episodes else 0.0
                total_success_rate = total_successes / total_episodes if total_episodes else 0.0
                trial_logger.info(
                    "episode_result | task_id=%d | episode=%d | steps_taken=%d | task success=%d | total success=%d | task_success_rate=%.4f | total_success_rate=%.4f",
                    task_id + 1,
                    episode_idx + 1,
                    int(t),
                    task_successes,
                    total_successes,
                    task_success_rate,
                    total_success_rate,
                )

                episode_rows.append(
                    {
                        "task_id": task_id + 1,
                        "task_name": task_name,
                        "task_description": task_description,
                        "category": task_category,
                        "difficulty_level": difficulty_level,
                        "trial_index": episode_idx + 1,
                        "success": int(done),
                        "steps_taken": int(t),
                    }
                )
        finally:
            env.close()

    total_success_rate = total_successes / total_episodes if total_episodes else 0.0
    trial_logger.info(
        "trial_summary | total_episodes=%d | total_successes=%d | total_success_rate=%.4f",
        total_episodes,
        total_successes,
        total_success_rate,
    )

    return {
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": total_success_rate,
        "episode_rows": episode_rows,
    }


def eval_libero_plus(args: Args) -> None:
    category_name = _category_dir_name(args.task_categories)
    base_results_root = pathlib.Path(args.results_out_path) / args.methods / category_name
    base_results_root.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args.prompt_file)
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        ping_interval=args.websocket_ping_interval,
        ping_timeout=args.websocket_ping_timeout,
    )

    logging.info("Running LIBERO-plus evaluation")
    result = _eval_once(args=args, client=client, prompts=prompts)
    all_episode_rows = result["episode_rows"]

    save_tag = _resolve_save_tag(args)
    episode_csv_path = base_results_root / f"episodes_{save_tag}_{args.task_suite_name}.csv"
    statistics_path = base_results_root / f"statistics_{save_tag}_{args.task_suite_name}.csv"
    _write_episode_summary(all_episode_rows, episode_csv_path)
    _write_statistics_table(all_episode_rows, statistics_path)
    overall_results_path = base_results_root / f"{category_name}-overall-results.csv"
    _append_overall_results(
        args=args,
        episode_rows=all_episode_rows,
        overall_results_path=overall_results_path,
    )

    logging.info("Finished. Episode result table: %s", episode_csv_path)
    logging.info("Finished. Statistics table: %s", statistics_path)
    logging.info("Updated category overall results: %s", overall_results_path)


def _get_libero_env(task, resolution):
    """Initializes and returns the LIBERO-plus environment and task description."""
    task_description = task.language
    task_bddl_file = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite's quaternion-to-axis-angle conversion."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_plus)
