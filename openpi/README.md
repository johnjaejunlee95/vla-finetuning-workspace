# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/). ***I didn't changed the original README.md file, but added some additional information about the repo and its contents.***

Currently, this repo contains three types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

## Fine-Tuning Base Models on Your Own Data

The rest of this README focuses on fine-tuning the $\pi_{0.5}$ base model on the [LIBERO dataset](https://libero-project.github.io/datasets), then running LIBERO evaluation against the trained checkpoint. The same flow can be adapted to other LeRobot datasets.

We will explain four steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Finetune-LIBERO
4. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). This script follows the standard LIBERO conversion flow and can be modified for your own data format. You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need configs for data processing and training. The LIBERO configs are defined in [`src/openpi/training/config.py`](src/openpi/training/config.py), and the LIBERO policy transforms are defined in [`src/openpi/policies/libero_policy.py`](src/openpi/policies/libero_policy.py):

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

The repo provides LIBERO fine-tuning configs for $\pi_0$, $\pi_0$-FAST, and $\pi_{0.5}$ in both full fine-tuning and LoRA / low-memory modes:

- `pi0_libero`
- `pi0_libero_low_mem_finetune`
- `pi0_fast_libero`
- `pi0_fast_libero_low_mem_finetune`
- `pi05_libero`
- `pi05_libero_low_mem_finetune`

Before training, compute normalization statistics for the training data. [`scripts/compute_norm_stats.py`](scripts/compute_norm_stats.py) loads the selected training config, iterates over the configured dataset, computes state and action normalization statistics, and writes `norm_stats.json` under that config's assets directory.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

I have already provided local `assets/` folders with normalization statistics for all six LIBERO config variants listed above, so you can either use the provided assets directly or recompute them after changing the dataset/config.

### 3. Finetune-LIBERO

Run full $\pi_{0.5}$ LIBERO fine-tuning with:

```bash
uv run scripts/train.py pi05_libero \
    --exp-name=pi05-LIBERO-full-baseline \
    --overwrite \
    --save-interval 10000 \
    --checkpoint-base-dir your/own/data/path \
    --num-train-steps 30000 \
    --seed 1111 \
    --batch-size 32 \
    --log-interval 100 \
    --num-workers 16 \
    --fsdp-devices 4
```

The command logs training progress to the console, writes checkpoints under `your/own/data/path/pi05-LIBERO-full-baseline`, and can also report to Weights & Biases if it is enabled in your environment.

Full fine-tuning without LoRA requires substantially more GPU memory than LoRA / low-memory fine-tuning. Because I only have 4 GPUs, I set `--fsdp-devices 4` to shard the model across all 4 devices and reduce per-GPU memory usage.

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 4. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).

To share my evaluation experience, I prepared a bash script that launches policy servers and LIBERO evaluation jobs. The example below is the setup I used in practice; adjust `CPU_LISTS`, checkpoint paths, ports, and task suites for your own environment:

```bash

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Edit these CPU ranges for your server before running.
CPU_LISTS=("0-15" "16-31" "32-47" "48-63")

CUDA_VISIBLE_DEVICES=0 taskset --cpu-list ${CPU_LISTS[0]} uv run scripts/serve_policy.py --port 1234 --env LIBERO_pi05 policy:checkpoint --policy.config pi05_libero --policy.dir /your/own/model/checkpoint/dir_root/ &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list ${CPU_LISTS[1]} uv run scripts/serve_policy.py --port 1031 --env LIBERO_pi05 policy:checkpoint --policy.config pi05_libero --policy.dir /your/own/model/checkpoint/dir_root/ &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list ${CPU_LISTS[2]} uv run scripts/serve_policy.py --port 1739 --env LIBERO_pi05 policy:checkpoint --policy.config pi05_libero --policy.dir /your/own/model/checkpoint/dir_root/ &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list ${CPU_LISTS[3]} uv run scripts/serve_policy.py --port 4930 --env LIBERO_pi05 policy:checkpoint --policy.config pi05_libero --policy.dir /your/own/model/checkpoint/dir_root/ &

sleep 10

CUDA_VISIBLE_DEVICES=0 taskset --cpu-list ${CPU_LISTS[0]} python examples/libero/main.py --args.task-suite-name libero_spatial --args.methods pi05-libero-full-baseline --args.seeds 42 --args.port 1739 --args.trial_num 3 &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list ${CPU_LISTS[1]} python examples/libero/main.py --args.task-suite-name libero_object --args.methods pi05-libero-full-baseline --args.seeds 42 --args.port 4930 --args.trial_num 3 &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list ${CPU_LISTS[2]} python examples/libero/main.py --args.task-suite-name libero_goal --args.methods pi05-libero-full-baseline --args.seeds 42 --args.port 1031 --args.trial_num 3 &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list ${CPU_LISTS[3]} python examples/libero/main.py --args.task-suite-name libero_10 --args.methods pi05-libero-full-baseline --args.seeds 42 --args.port 1234 --args.trial_num 3 &
```

### More Examples

I remove addtional examples from this README for brevity. For more examples, see the [examples](examples) folder, and I will add more examples in the future.

## PyTorch Support

I excluded the PyTorch experiments from this README. In my runs, the PyTorch version produced substantially different accuracies compared with the JAX version, so I only report and describe the JAX-based LIBERO fine-tuning and evaluation workflow above.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |
