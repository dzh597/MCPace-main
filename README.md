# NativE + MCPace: Multi-modal Knowledge Graph Completion in the Wild

![version](https://img.shields.io/badge/version-1.1.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![SIGIR 2024](https://img.shields.io/badge/SIGIR-2024-%23bd9f65?labelColor=%2377BBDD&color=3388bb)

This repository contains the implementation of **NativE: Multi-modal Knowledge Graph Completion in the Wild** and an extended implementation of **MCPace**, a gradient-space modality coordination module for multi-modal knowledge graph completion (MMKGC).

- Paper: [NativE: Multi-modal Knowledge Graph Completion in the Wild (ACM Digital Library)](https://dl.acm.org/doi/abs/10.1145/3626772.3657800)
- Preprint: [NativE: Multi-modal Knowledge Graph Completion in the Wild (arXiv)](https://arxiv.org/abs/2406.17605)

> **Scope of this codebase.** The original NativE pipeline supports multiple WildKGC datasets. The MCPace extension currently supports the confirmed modalities `structure`, `visual`, and `textual`, and has been integrated into the DB15K, MKG-W, and MKG-Y training paths.

---

## 1. Method Overview

NativE addresses modality diversity and imbalance in MMKGC through relation-aware adaptive fusion and adversarial modality augmentation. This codebase further adds MCPace, which coordinates modalities in gradient space.

MCPace includes the following components:

1. **Modal Gradient Extraction**: extracts gradients of `structure`, `visual`, and `textual` modal representations from the final KGC training loss.
2. **RACE**: maintains relation-aware accumulated contribution energy for each modality.
3. **MCP Plane**: measures pairwise modality balance and gradient consistency.
4. **Energy-aware Rebalancing**: compensates under-optimized modalities based on accumulated energy.
5. **Cooperative Enhanced Projection**: enhances aligned modal gradients through block-wise projection.
6. **Conflict Orthogonal Projection**: reduces destructive gradient conflict with minimal perturbation.
7. **Consensus Update**: aggregates pairwise corrections into one coordinated modal gradient.

The current MCPace implementation modifies only the KGC/discriminator update. The generator loss is kept unchanged.

---

## 2. Repository Structure

```text
.
├── args.py                         # Command-line arguments
├── run_adv_wgan_gp.py              # Main entry for MKG-W / MKG-Y
├── run_adv_wgan_gp_3modal.py       # Main entry for DB15K
├── run_adv_wgan_gp_4modal.py       # Main entry for Kuai16K / TIVA
├── scripts/                        # Reproducible training scripts
│   ├── run_db15k.sh
│   ├── run_mkgw.sh
│   ├── run_mkgy.sh
│   ├── run_kvc16k.sh
│   └── run_tiva.sh
├── benchmarks/                     # Dataset triples and id mappings
├── embeddings/                     # Pre-trained multi-modal embeddings
├── checkpoint/                     # Saved model checkpoints
├── log/                            # Optional log directory
└── mmkgc/
    ├── adv/                        # Adversarial generators and auxiliary modules
    ├── base/                       # C++ data-sampling backend source
    ├── config/                     # Trainers and evaluator
    ├── data/                       # Train/test data loaders
    ├── module/
    │   ├── loss/                   # Loss functions
    │   ├── model/                  # KGE/MMKGC model definitions
    │   ├── strategy/               # Negative-sampling strategies
    │   └── mcpace.py               # MCPace gradient coordinator
    └── release/Base.so             # Compiled OpenKE-style sampler
```

---

## 3. Environment Setup

### 3.1 Recommended Environment

The original project was developed with:

```text
Python 3.8
PyTorch 1.9.1
NumPy 1.23.3
scikit-learn 1.1.2
tqdm 4.64.1
```

A typical installation is:

```bash
conda create -n native python=3.8 -y
conda activate native
pip install -r requirements.txt
```

If you already have a prepared environment, activate it before running experiments. For example, in our local verification environment:

```bash
conda activate lor
```

### 3.2 C++ Sampler

The dataloaders rely on the compiled sampler:

```text
mmkgc/release/Base.so
```

If `Base.so` is missing or incompatible with your system, rebuild it from the `mmkgc` directory:

```bash
cd mmkgc
bash make.sh
cd ..
```

---

## 4. Data and Embeddings

### 4.1 Dataset Files

Each dataset under `benchmarks/` follows the OpenKE-style format:

```text
entity2id.txt
relation2id.txt
train2id.txt
valid2id.txt
test2id.txt
test2id_all.txt
type_constrain.txt
```

Supported datasets in this repository include:

```text
DB15K
MKG-W
MKG-Y
Kuai16K
TIVA
```

### 4.2 Pre-trained Multi-modal Embeddings

Download the pre-trained multi-modal embeddings from the original project release and place them under:

```text
embeddings/
```

Expected examples:

```text
DB15K-visual.pth
DB15K-textual.pth
DB15K-numeric.pth
MKG-W-visual.pth
MKG-W-textual.pth
MKG-Y-visual.pth
MKG-Y-textual.pth
Kuai16K-visual.pth
Kuai16K-textual.pth
Kuai16K-audio.pth
Kuai16K-video.pth
TIVA-visual.pth
TIVA-textual.pth
TIVA-audio.pth
TIVA-video.pth
```

---

## 5. Training and Evaluation

All scripts should be launched from the repository root.

### 5.1 DB15K with MCPace

```bash
bash scripts/run_db15k.sh
```

This script uses:

```text
run_adv_wgan_gp_3modal.py
MCPaceRotatEDB15K
WCGTrainerDB15KGP
MCPaceCoordinator
```

MCPace coordinates:

```text
structure, visual, textual
```

`numeric` remains part of the DB15K NativE fusion model, but it is not coordinated by MCPace in the current implementation.

### 5.2 MKG-W with MCPace

```bash
bash scripts/run_mkgw.sh
```

This script uses:

```text
run_adv_wgan_gp.py
MCPaceRotatE
WCGTrainerGP
MCPaceCoordinator
```

### 5.3 MKG-Y with MCPace

```bash
bash scripts/run_mkgy.sh
```

This script uses the same 2-modal training path as MKG-W.

### 5.4 Kuai16K and TIVA

The original training scripts are:

```bash
bash scripts/run_kvc16k.sh
bash scripts/run_tiva.sh
```

MCPace is not yet enabled for these 4-modal paths by default. Their current implementation uses `run_adv_wgan_gp_4modal.py` and the corresponding 4-modal model/trainer logic.

---

## 6. Important Arguments

General training arguments are defined in `args.py`.

| Argument | Description |
|---|---|
| `-dataset` | Dataset name, e.g., `DB15K`, `MKG-W`, `MKG-Y` |
| `-batch_size` | Training batch size |
| `-neg_num` | Number of negative samples per positive triple |
| `-dim` | Base embedding dimension |
| `-margin` | RotatE-style margin |
| `-epoch` | Number of training epochs |
| `-learning_rate` | KGC/discriminator learning rate |
| `-lrg` | Generator learning rate |
| `-mu` | Weight for adversarial/fake-modality loss |
| `-save` | Checkpoint path |

MCPace-specific arguments:

| Argument | Default | Description |
|---|---:|---|
| `-use_mcpace` | `0` | Whether to enable MCPace |
| `-mcpace_mu` | `0.1` | Consensus coordination strength |
| `-mcpace_blocks` | `4` | Number of blocks for cooperative projection |
| `-mcpace_lambda_alpha` | `1.0` | Regularization for block-wise projection coefficients |
| `-mcpace_eps` | `1e-8` | Numerical stability constant |
| `-mcpace_min_rebalance` | `0.2` | Lower bound for energy-aware rebalancing |
| `-mcpace_max_rebalance` | `5.0` | Upper bound for energy-aware rebalancing |
| `-mcpace_log_interval` | `0` | Print MCPace statistics every N steps; `0` disables logging |

---

## 7. Reproducibility Notes

- Random seeds are set in the main scripts via `torch.manual_seed(args.seed)` and `torch.cuda.manual_seed_all(args.seed)`.
- The training dataloader uses the OpenKE-style C++ sampler in `mmkgc/release/Base.so`.
- Checkpoints are saved to the path specified by `-save`.
- Training logs are redirected by the shell scripts to dataset-specific `.txt` files.
- MCPace computes additional gradients through `torch.autograd.grad`; therefore, enabling MCPace increases memory usage and training time.

---

## 8. Implementation Details for MCPace

The MCPace integration follows this gradient correction workflow:

```text
1. Forward pass and final KGC/discriminator loss computation
2. torch.autograd.grad extracts modal gradients
3. RACE and MCP statistics are updated from full batch_r
4. MCPace computes modified modal gradients
5. loss.backward() preserves ordinary parameter gradients
6. delta backward applies g_mod - g_orig to modal tensors
7. optimizer.step() updates the model
```

The implementation files are:

```text
mmkgc/module/mcpace.py
mmkgc/module/model/MCPaceRotatE.py
mmkgc/module/model/MCPaceRotatEDB15K.py
mmkgc/config/WCGTrainerGP.py
mmkgc/config/WCGTrainerDB15KGP.py
run_adv_wgan_gp.py
run_adv_wgan_gp_3modal.py
```

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Activate the correct environment before running:

```bash
conda activate <your-env-name>
```

### `OSError: cannot open shared object file: Base.so`

Rebuild the C++ sampler:

```bash
cd mmkgc
bash make.sh
cd ..
```

### CUDA out-of-memory when MCPace is enabled

Try reducing:

```text
-batch_size
-neg_num
-mcpace_blocks
```

or disable MCPace with:

```text
-use_mcpace=0
```

---

## 10. Citation

If you use this codebase, please cite the original NativE paper:

```bibtex
@inproceedings{DBLP:conf/sigir/ZhangCGXHLZC24,
  author    = {Yichi Zhang and Zhuo Chen and Lingbing Guo and Yajing Xu and
               Binbin Hu and Ziqi Liu and Wen Zhang and Huajun Chen},
  title     = {NativE: Multi-modal Knowledge Graph Completion in the Wild},
  booktitle = {SIGIR},
  pages     = {91--101},
  publisher = {ACM},
  year      = {2024}
}
```

---

## 11. Acknowledgements

This codebase is built on the OpenKE-style KGC training pipeline. We thank the authors of NativE and the OpenKE community for their contributions to MMKGC and knowledge graph representation learning.
