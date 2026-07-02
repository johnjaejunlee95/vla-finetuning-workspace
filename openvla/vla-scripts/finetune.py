"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR" 
os.environ["absl_logging_verbosity"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import draccus
import random
import torch
import torch.distributed as dist
from collections import deque
import tensorflow as tf
from tqdm import tqdm
tqdm.disable = True

import torch.nn.functional as F
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

import warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "droid_wipe"
    dataset_names: list[str] = None 
    run_root_dir: Path = Path("runs")
    adapter_dir: Path = Path("adapter")
    save_path: Optional[Path] = None

    # Fine-tuning Parameters
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True
    last_ckpt: int = 100_000

    
    # LoRA Arguments
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    resume: bool = False
    resume_step: Optional[int] = 0
    checkpoint_path: Optional[str] = None
    wandb_id: Optional[str] = None
    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"
    lr_scheduler_stepsize: int = 100_000
    run_id_note: Optional[str] = None


def make_wandb_config(cfg: FinetuneConfig) -> dict:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(val) for val in value]
        return value

    return convert(asdict(cfg))


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name if cfg.dataset_names is None else 'LIBERO Full Suites'}`")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True

    save_path = cfg.save_path
    if cfg.save_path is None:
        save_path = cfg.run_root_dir
    run_dir, adapter_dir = cfg.run_root_dir, cfg.run_root_dir / cfg.adapter_dir
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        if cfg.resume and adapter_dir.exists():
            if distributed_state.is_main_process:
                print(f"Resuming LoRA weights from {adapter_dir}")
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
        else:
            if distributed_state.is_main_process:
                print("Starting LoRA fine-tuning from scratch (Gaussian init)")
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], gradient_as_bucket_view=True, static_graph=True)
    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)
    
    lr_scheduler_stepsize = cfg.lr_scheduler_stepsize
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_scheduler_stepsize,
        gamma=0.1,
    )
    
    start_step = 0
    start_step = 0
    if cfg.resume:
        checkpoint = torch.load(f"{save_path}/optimizer_state_dict_ckpt.pt", map_location="cpu")

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        fresh_lrs = []
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate
            fresh_lrs.append(group["lr"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            scheduler.base_lrs = fresh_lrs
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = fresh_lrs.copy()
            print("[Resume] Scheduler state loaded (LR overridden).")
        else:
            print("[Resume] No scheduler state found in checkpoint.")

        start_step = checkpoint["gradient_step_idx"]
        scheduler.step_size = lr_scheduler_stepsize
        
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if k == "step":
                            continue
                        state[k] = v.to(device=p.device, dtype=p.dtype, non_blocking=True)
                    
        if distributed_state.is_main_process:
            print(f"Resumed training from step {start_step}.")
            print(f"Optimizer and scheduler states loaded: LR={scheduler.get_last_lr()[0]:.3e}")
            print(f"Scheduler Step Size={scheduler.step_size}, last_epoch={scheduler.last_epoch}")
    
    
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    # if hasattr(vla_dataset, "dataset"):
    #     if distributed_state.is_main_process:
    #         print("Injecting tf.data.AUTOTUNE prefetch to RLDSDataset...")
    #     vla_dataset.dataset = vla_dataset.dataset.prefetch(tf.data.AUTOTUNE)

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, save_path)

    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,      
        pin_memory=True,
    )

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, 
                   project=cfg.wandb_project, 
                   name=f"ft+{cfg.run_id_note}", 
                   config=make_wandb_config(cfg),
                   id = cfg.wandb_id if (cfg.resume or cfg.wandb_id) else None,
                   resume="allow" if cfg.resume else None)
        
        wandb_last_step = -1
        
        if cfg.resume and cfg.wandb_id:
            try:
                api = wandb.Api()
                run_path = f"{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.wandb_id}"
                prev_run = api.run(run_path)
                
                wandb_last_step = prev_run.lastHistoryStep
                print(f"==================================================")
                print(f"[WandB] Fetched last logged step from server: {wandb_last_step}")
                print(f"==================================================")
                
            except Exception as e:
                print(f"[Warning] Failed to fetch WandB run info: {e}")
                print("Logging will proceed based on local step.")

    if distributed_state.is_main_process:
        print(f"Starting Fine-tuning Loop with {cfg.dataset_name}")

    vla.train()
    
    start_time = time.time()
    max_steps = cfg.max_steps
    overall_steps = cfg.max_steps - start_step
    
    accumulated_batches = []
    gradient_step_idx = start_step
    
    action_acc_list = deque(maxlen=100)
    action_l1_list = deque(maxlen=100)
    action_loss_list = deque(maxlen=100)

    for batch_idx, batch in enumerate(dataloader):
        accumulated_batches.append(batch)

        if len(accumulated_batches) == cfg.grad_accumulation_steps:
            for i, mb in enumerate(accumulated_batches):
        
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    input_ids = mb["input_ids"].to(device_id, non_blocking=True)
                    attention_mask = mb["attention_mask"].to(device_id, non_blocking=True)
                    pixel_values = mb["pixel_values"].to(torch.bfloat16).to(device_id, non_blocking=True)
                    labels = mb["labels"].to(device_id, non_blocking=True)

                    output: CausalLMOutputWithPast = vla(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                loss = output.loss / cfg.grad_accumulation_steps
                loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            optimizer.zero_grad()
            gradient_step_idx += 1
            accumulated_batches.clear()

            if distributed_state.is_main_process and gradient_step_idx % 5 == 0:
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = labels[:, 1:]
                mask = action_gt > action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
                continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
                action_l1_loss = F.l1_loss(continuous_actions_pred, continuous_actions_gt)
                
                action_acc_list.append(action_accuracy.item())
                action_l1_list.append(action_l1_loss.item())
                action_loss_list.append(loss.item() * cfg.grad_accumulation_steps)
                
                if (gradient_step_idx > wandb_last_step):
                    wandb.log(
                        {
                            "train_loss": loss.item() * cfg.grad_accumulation_steps,
                            "action_accuracy":action_accuracy.item(),
                            "l1_loss": action_l1_loss.item(),
                        },
                        step=gradient_step_idx,
                    )
                if gradient_step_idx % 100 == 0:
                    print(f"Step {gradient_step_idx}: train_loss={loss.item() * cfg.grad_accumulation_steps:.4f}, action_accuracy={action_accuracy.item():.4f}, l1_loss={action_l1_loss.item():.4f}")
                
            dist.barrier()
            
            if (gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0) or (gradient_step_idx - (start_step) == 100) or (gradient_step_idx == max_steps):
                if distributed_state.is_main_process:
                    elapsed_time = time.time() - start_time
                    steps_done = gradient_step_idx - start_step
                    eta_time = (elapsed_time / steps_done) * (overall_steps - steps_done) if steps_done > 0 else 0

                    eta_days = int(eta_time // 86_400)
                    eta_hours = int((eta_time % 86_400) // 3_600)
                    eta_minutes = int((eta_time % 3_600) // 60)

                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{current_time}] Step: {gradient_step_idx}/{max_steps} | Saving Model Checkpoint...")
                    
                    save_dir = save_path / cfg.adapter_dir if cfg.use_lora else save_path

                    processor.save_pretrained(save_path)
                    vla.module.save_pretrained(save_dir)
                    
                    torch.save({
                        'gradient_step_idx': gradient_step_idx,
                        'optimizer_state_dict': optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, os.path.join(save_path, "optimizer_state_dict_ckpt.pt"))
                    
                    print(f"[{current_time}]: Saved Checkpoint for Step {gradient_step_idx} (ETA: {eta_days}d {eta_hours}h {eta_minutes}m)")
                
                dist.barrier()

                if cfg.use_lora and (gradient_step_idx == max_steps) or (gradient_step_idx == 500):
                    base_vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False)
                    
                    merged_vla = PeftModel.from_pretrained(base_vla, save_path / cfg.adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        merged_vla.save_pretrained(save_path)
                        elapsed_time = time.time() - start_time
                        steps_done = gradient_step_idx - start_step
                        eta_time = (elapsed_time / steps_done) * (overall_steps - steps_done) if steps_done > 0 else 0

                        eta_days = int(eta_time // 86_400)
                        eta_hours = int((eta_time % 86_400) // 3_600)
                        eta_minutes = int((eta_time % 3_600) // 60)

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{current_time}]: Saved Model Checkpoint for Step {gradient_step_idx} at {save_path} (elapsed: {elapsed_time/3600:.2f}h, "
                            f"ETA: {eta_days} days {eta_hours} hours {eta_minutes} min left ({eta_time/3600:.2f}h))")

                dist.barrier()

            if gradient_step_idx == max_steps:
                print(f"Max step {max_steps} reached! Stopping training...")
                break
    
    print("Fine-tuning Complete!")


if __name__ == "__main__":
    finetune()
