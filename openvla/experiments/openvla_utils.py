"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
import warnings 

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers import BitsAndBytesConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, CustomOpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from peft import PeftModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # print("[*] Instantiating Pretrained VLA model")
    # print("[*] Loading in BF16 with Flash-Attention Enabled")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, CustomOpenVLAForActionPrediction)
    
    
    if 'nfs' in cfg.pretrained_checkpoint :
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            attn_implementation=cfg.attn_impl,#"flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            # trust_remote_code=False,
        )
        merged_vla = PeftModel.from_pretrained(vla, cfg.pretrained_checkpoint + '/adapter')
        vla = merged_vla.merge_and_unload()
        
    else:         
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            attn_implementation=cfg.attn_impl,#"flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            # trust_remote_code=False,
        )
   
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True, use_fast=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, current_noise_type, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        image = tf.image.convert_image_dtype(image, tf.float32)

        image = crop_and_resize(image, crop_scale, batch_size)
        
        if current_noise_type is not None:
            image = inject_visual_noise(image, noise_type=current_noise_type)

        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    
    input_ids = inputs["input_ids"]
    input_pixel_values = inputs["pixel_values"]
    action, logits = vla.predict_action(input_ids, input_pixel_values, unnorm_key=unnorm_key, do_sample=False)
    
    return action, logits, image


import tensorflow_addons as tfa
import math 

K_SHARP = tf.constant([[0., -1., 0.],
                       [-1., 5., -1.],
                       [0., -1., 0.]], dtype=tf.float32)
K_SHARP = tf.reshape(K_SHARP, [3, 3, 1, 1])

def _sharpness_kernel_conv(image_f32):
    c = tf.shape(image_f32)[-1]
    k = tf.tile(K_SHARP, [1, 1, c, 1])  # [3,3,C,1]
    x = tf.expand_dims(image_f32, 0)
    y = tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding="SAME")
    return tf.squeeze(y, 0)


def inject_visual_noise(image, noise_type=None):
    if noise_type is None:
        return image

    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    if noise_type == 'gaussian':
        sigma = 70.0 / 255.0
        noise = tf.random.normal(shape, 0.0, sigma)
        image = image + noise

    elif noise_type == 'dead_pixel':
        corruption_prob = 0.1
        H, W = shape[0], shape[1]

        mask = tf.random.uniform([H, W, 1]) < corruption_prob
        flip = tf.random.uniform([H, W, 1]) < 0.5

        val = tf.where(flip, tf.ones_like(image_f[..., :1]), tf.zeros_like(image_f[..., :1]))
        image_f = tf.where(mask, val, image_f)
        
    elif noise_type == 'blur':
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tfa.image.gaussian_filter2d(image, filter_shape=5, sigma=1.5, padding='REFLECT')
        
    elif noise_type == 'jitter':
        max_factor = 0.4
        delta = tf.random.uniform([], -max_factor, max_factor)
        contrast_factor = tf.random.uniform([], 1.0 - max_factor, 1.0 + max_factor)
        saturation_factor = tf.random.uniform([], 1.0 - max_factor, 1.0 + max_factor)
        sharpness_factor = tf.random.uniform([], 1.0 - max_factor, 1.0 + max_factor)

        image = image + delta

        mean_val = tf.reduce_mean(image)
        image = (image - mean_val) * contrast_factor + mean_val

        image = tf.clip_by_value(image, 0.0, 1.0)
        hsv = tf.image.rgb_to_hsv(image)

        h = hsv[..., 0]
        s = tf.clip_by_value(hsv[..., 1] * saturation_factor, 0.0, 1.0)
        v = hsv[..., 2]
        hsv = tf.stack([h, s, v], axis=-1)

        image = tf.image.hsv_to_rgb(hsv)

        sharpened = _sharpness_kernel_conv(image)
        w = sharpness_factor - 1.0
        image = image * (1.0 - w) + sharpened * w

    elif noise_type == 'rotation':
        max_angle = 30.0
        angle = tf.random.uniform([], 0.0, max_angle) * math.pi / 180.0
        image = tfa.image.rotate(
            image,
            angle,
            interpolation='BILINEAR',
            fill_mode='CONSTANT',
            fill_value=0.0
        )

    elif noise_type == 'shift':
        shift_fraction = 0.15
        dx = tf.random.uniform([], -shift_fraction*w, shift_fraction*w)
        dy = tf.random.uniform([], -shift_fraction*h, shift_fraction*h)
        image = tfa.image.translate(image, [dx, dy], interpolation='BILINEAR', fill_mode='REFLECT')

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
