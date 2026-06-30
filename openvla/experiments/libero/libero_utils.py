"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import torch
import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description):
    """Saves an MP4 replay of an episode."""
    # rollout_dir = f"./rollouts/{DATE}"

    # if os.path.exists(rollout_dir):
    #     print(f"Directory {rollout_dir} already exists. Removing it to save new rollouts.")
    #     shutil.rmtree(rollout_dir)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4" #{DATE_TIME}--
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
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



def inject_noise2action(
        action = None,
        noise_std = 0.05,
        clip_trans = 2.0,
        clip_rot = 2.0,
    ) -> torch.Tensor:
    assert action.shape[-1] == 7, f"Expected last dim 7, got {action.shape[-1]}"
    action = torch.from_numpy(action)

    trans = action[..., :3]
    rot = action[..., 3:6]
    grip = action[..., 6:7]
    
    trans_std = noise_std
    rot_noise_std = noise_std
    
    noise_trans = torch.randn_like(trans) * trans_std
    noise_rot = torch.randn_like(rot) * rot_noise_std

    noise_trans = torch.randn((10, *trans.shape), device=trans.device, dtype=trans.dtype).mean(dim=0) * trans_std
    noise_rot   = torch.randn((10, *rot.shape),   device=rot.device,   dtype=rot.dtype).mean(dim=0) * rot_noise_std
    
    noisy_trans = trans + noise_trans
    noisy_rot = rot + noise_rot

    noisy_trans = torch.clamp(noisy_trans, -clip_trans, clip_trans)
    noisy_rot = torch.clamp(noisy_rot, -clip_rot, clip_rot)

    return torch.cat([noisy_trans, noisy_rot, grip], dim=-1).numpy()


import pandas as pd
def create_clean_summary(df, value_col, index_col='trial_index', col_col='task_description'):
    pivot_df = df.pivot(index=index_col, columns=col_col, values=value_col)
    pivot_df = pivot_df.sort_index(axis=1)
    pivot_df['Average'] = pivot_df.sum(axis=1).round(4)

    stats = pivot_df.agg(['sum', 'std']).round(4)
    stats.index = ['Task Average (Sum)', 'Task Std']
    
    final_df = pd.concat([pivot_df, stats])
    
    return final_df
