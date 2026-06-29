# VLA Archive


## Description

This repository is a local archive and baseline workspace for vision-language-action (VLA) robotics codebases. It is designed to facilitate the comparison of various VLA approaches and to iterate on model-specific configurations over time.

Since I currently do not have access to real robot hardware, the main focus of this archive is simulation-based evaluation, especially LIBERO task benchmark. I also maintain additional evaluation benchmarks such as LIBERO-Plus, which is intended for testing model performance under more complex and robust scenarios. A local copy of the ALOHA robot codebase is also included for reference, although it has not yet been tested in my environment.


## Refined Configurations

The main purpose of this archive is to support future tuning, comparison, and revision of optimized settings for OpenVLA, OpenVLA-OFT, and the $\pi$-series models from OpenPI.

The included environment files, setup scripts, and project snapshots should be treated as starting points rather than fixed configurations. As better model-specific training, fine-tuning, and evaluation settings are identified, these files may be updated accordingly.

## Contents

- `openpi/`: Physical Intelligence's `openpi` repository, including code, examples, and documentation for $\pi_0$, $\pi_0$-FAST, and $\pi_{0.5}$ models.
- `openvla/`: OpenVLA codebase for training, fine-tuning, and evaluating VLA models, including LIBERO-related evaluation files.
- `openvla-oft/`: OpenVLA-OFT codebase for parameter-efficient OpenVLA fine-tuning, including LIBERO and ALOHA setup notes, local LIBERO copies, and a Conda-based installation script.

## Usage

Each project should be used from its own subdirectory. Since each repository has different dependency requirements, it is highly recommended to keep their environments separate.

```bash
cd openpi
# See openpi/README.md

cd ../openvla
# See openvla/README.md

cd ../openvla-oft
# See openvla-oft/README.md, SETUP.md, and LIBERO.md
```

For environment setup, check the project-specific setup files:

- `openvla/installation.sh`
- `openvla-oft/installation.sh`
- `openpi/README.md` (`uv`-based environment instructions)

## Notes

This repository is intended to serve as a working snapshot for vision-language action research. Each subdirectory should be treated as an independent project with its own dependencies, installation process, and usage assumptions. For further details, follow the instructions in each each subdirectory. Moreover, please note the following:

- For any code or installation issues specific to an upstream project, please refer to the original repository first.
- For questions about this archive, its local setup, or its usage, please open an `Issues` in this repository. I will try to respond ASAP.
- **Future Updates:** I will continue adding other approaches, simulation benchmarks, and configuration notes as time allows.

## Reference Repositories

This archive is based on or related to the following public repositories:

- OpenPI: https://github.com/Physical-Intelligence/openpi/
- OpenVLA: https://github.com/openvla/openvla/
- OpenVLA-OFT: https://github.com/moojink/openvla-oft/
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus
