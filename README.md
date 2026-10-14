<div align="center">

# GASST

### Spatial–Spectral Feature Modeling for Hyperspectral Image Classification

**Adaptive Receptive Fields · Global–Local Interaction · Selective Token Fusion**

![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Get_Started-0F766E?style=flat-square)](#-quick-demo)

[Overview](#-overview) • [Installation](#a-installation) • [Data](#b-data-preparation) • [Demo](#-quick-demo) • [Experiments](#-training--evaluation) • [中文 Demo 指南](docs/DEMO.md)

</div>

## 📖 Overview

**GASST** is a hyperspectral image classification model organized around three modules: **GARFB**, **GLITB**, and **TSFAB**. It learns patch-level representations using multi-scale receptive fields and selective spatial–spectral feature processing, then predicts the class of the center pixel.

<div align="center">
  <img src="figures/GASST.png" alt="GASST conceptual architecture supplied with the project" width="100%">
  <br>
  <sub>Conceptual architecture supplied with the project. See the implementation note below for the executable structure.</sub>
</div>

| Module | Full name | Role in the released implementation |
| :--- | :--- | :--- |
| **GARFB** | Geometrically Adaptive Receptive Field Block | Combines depthwise convolutions with different receptive fields and spatial/channel-dependent fusion weights. |
| **GLITB** | Global-Local Interaction Transformer Block | Integrates a GARFB-based attention branch with normalization, residual connections, and a convolutional MLP. |
| **TSFAB** | Token Selective Fusion Attention Block | Selects Top-k attention connections over grouped features and applies a convolutional feed-forward update. |

<details>
<summary><strong>Architecture and implementation note</strong></summary>

The figure illustrates the supplied design; this release preserves the existing numerical implementation rather than rebuilding it from the diagram. The executable model is defined in [`models/GASST.py`](models/GASST.py).

With the default `depth=6`, its actual sequence is:

```text
HSI patch → 3×3 Conv → Patch Embedding
          → GLITB → TSFAB × 3 → GLITB → TSFAB × 3
          → Global Average Pooling → LayerNorm → Dropout → Linear
```

GARFB is nested within GLITB, and both GLITB calls share parameters. The diagram's separate GARFB stage and illustrated global/local QKV interaction are not a literal representation of this preserved implementation. Module names alone do not establish architectural equivalence with the conceptual figure.

</details>

## 🚀 Let's Get Started!

### A. Installation

Download the repository and open a terminal in the project root. The original project specifies **Python 3.8** and **PyTorch 1.13.1**; its dependency file is preserved.

```bash
conda create -n gasst python=3.8 -y
conda activate gasst
pip install -r requirements.txt
```

The badges describe the declared environment, not a passing CI build. A clean installation of all pinned dependencies has not yet been validated for this release.

### B. Data Preparation

Prepare the datasets separately and place them under `datasets/`. This repository does not bundle datasets or pretrained weights.

```text
datasets/
├── pu/                         # Pavia University
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── ip/                         # Indian Pines
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── houston13/                   # Houston 2013
│   ├── GRSS2013.mat
│   └── GRSS2013_gt.mat
└── whuhh/                      # WHU-Hi-HongHu
    ├── WHU_Hi_HongHu.mat
    └── WHU_Hi_HongHu_gt.mat
```

| Dataset | Identifier | `dataID` | Classes |
| :--- | :---: | :---: | :---: |
| Pavia University | `pu` | `1` | 9 |
| Indian Pines | `ip` | `4` | 16 |
| Houston 2013 | `houston13` | `3` | 15 |
| WHU-Hi-HongHu | `whuhh` | `7` | 22 |

The input convolution expects **30 channels**. Keep the default PCA configuration. Other dataset names accepted by the legacy command-line parser are not necessarily supported by the GASST model factory.

## ⚡ Quick Demo

### 1. Check the model without downloading data

```bash
python demo.py
```

This uses two synthetic patches of shape `[2, 1, 30, 10, 10]` to check model construction, forward propagation, cross-entropy loss, and finite backward gradients. The output shape is `[2, 9]`.

**This is an execution check, not an accuracy benchmark.** A successful run writes `smoke_result.json`, `manifest.json`, and `console.log` to a new directory under `outputs/`.

### 2. Inspect the experiment configuration

```bash
python demo.py --mode experiment --dataset pu --dry-run
```

This prints the command arguments and seed settings without importing PyTorch, opening data files, or starting training.

### 3. Run a real-data pipeline check

After preparing Pavia University data:

```bash
python demo.py --mode experiment --dataset pu --epochs 1 --runs 1 --output outputs/pu_pipeline_check
```

This invokes the existing `main.py` with one training epoch, followed by **full-scene inference and evaluation**. It is a pipeline check with intentionally short training; CPU inference may still take substantial time. Its scores must not be reported as fully trained GASST results.

### 4. Run the full configured experiment

```bash
python demo.py --mode experiment --dataset pu --epochs 500 --runs 1 --output outputs/pu_experiment
```

Use `--data-dir /path/to/datasets` for an external dataset directory. Each output directory must be new or empty, so previous experiments are not overwritten.

| Saved file | Contents |
| :--- | :--- |
| `manifest.json` | Actual arguments, initial seed, split seeds, package versions, source/data SHA-256 hashes, timestamps, and completion/failure status |
| `console.log` | Training/evaluation output and any error traceback |
| `metrics.json` | Per-run OA, AA, Kappa, class accuracies, confusion matrices, and timings |
| `last_run_split.npz` | Training/test masks from the last run; `-1` denotes unselected pixels |
| `checkpoints/GASST/...` | Checkpoints saved by the original training function |
| `Exp_*/` and `Log_*/` | Classification maps and experiment logs produced by the original entry point |

The wrapper adds recording and initial seeding without editing the model, preprocessing, optimizer, or training loop. See the [中文 Demo 指南](docs/DEMO.md) for details and troubleshooting.

## 🧪 Training & Evaluation

The following commands use the current defaults: **30 training samples per class**, **Top-k ratio 1/2**, and **500 maximum epochs**. They are reproducible configuration examples, not verified paper-specific optimal settings.

```bash
python demo.py --mode experiment --dataset pu        --epochs 500 --runs 1
python demo.py --mode experiment --dataset ip        --epochs 500 --runs 1
python demo.py --mode experiment --dataset houston13 --epochs 500 --runs 1
python demo.py --mode experiment --dataset whuhh     --epochs 500 --runs 1
```

For direct access to the original entry point:

```bash
python main.py --model GASST --dataset_name pu --dataID 1 --dataset_dir ./datasets --num_run 1 --epoch 500
```

<details>
<summary><strong>Default model and training parameters</strong></summary>

| Parameter | Value | Description |
| :--- | :---: | :--- |
| `patch_size` / `ps` | `10` / `2` | Input patch / embedding patch size |
| `pc` | `30` | PCA components |
| `emb_dim` | `128` | Embedding dimension |
| `depth` | `6` | Number of TSFAB blocks |
| `num_heads` / `group_num` | `8` / `4` | Attention heads / feature groups |
| `k` | `1/2` | Top-k selection ratio |
| `bs` | `256` | Training batch size |
| `drop_rate` | `0.1` | Dropout probability |
| Optimizer | AdamW | Initial learning rate `0.0005`, weight decay `0.0001` |
| Scheduler | Cosine annealing | Minimum learning rate `1e-6` |
| Device | CPU | Preserved from the existing entry point |

The demo defaults to one run; `main.py` alone defaults to three. The training loop retains its original early stopping. Passing `--device` to `main.py` does not switch its hardcoded CPU device to CUDA.

</details>

### Reproducibility status

The release has passed model-renaming checks: parameter keys and initial values match, strict weight loading succeeds, and fixed-input forward outputs and gradients are identical in an isolated CPU comparison. See the [validation record](docs/rename_validation.json).

**A full GASST benchmark has not been rerun. Verified target OA/AA/Kappa values and an associated checkpoint/configuration bundle are not yet included.** The demo makes experiments easier to run and audit; it does not promise to reproduce unspecified historical scores.

Important protocol details are documented in [docs/DEMO.md](docs/DEMO.md): the original training-set validation, final-model evaluation, separately fitted PCA transforms, and repeated-run behavior are preserved. Initial seeding improves traceability but does not guarantee identical results across environments or reproduce past runs whose Torch seed was not recorded.

> **Metric units:** the original metric function reports OA, AA, per-class accuracy, and **Kappa × 100**. Divide the stored Kappa by 100 when reporting it on the usual coefficient scale.

## 🗂️ Repository Structure

```text
GASST/
├── demo.py                     # Smoke check and recorded experiment launcher
├── main.py                     # Existing training/evaluation entry point
├── train.py                    # Training, validation, and inference
├── models/
│   ├── GASST.py                # GASST, GARFB, GLITB, and TSFAB
│   └── get_model.py            # Model factory
├── utils/                      # Data, metrics, optimization, and visualization
├── figures/GASST.png           # Supplied conceptual architecture
├── datasets/                   # User-provided data
├── docs/
│   ├── DEMO.md                 # Chinese demo guide and reproducibility notes
│   ├── RELEASE_NOTES.md        # Naming and packaging changes
│   ├── rename_validation.json  # Numerical equivalence record
│   └── UPSTREAM_README.md      # Upstream attribution and references
├── requirements.txt
└── LICENSE
```

## 📜 Reference

GASST publication metadata has not yet been supplied for this release, so a GASST BibTeX entry is not fabricated here. The upstream DSFormer reference is preserved in [the original project documentation](docs/UPSTREAM_README.md); it is attribution for the source implementation, not a GASST paper citation.

## 🙋 Q & A

**Can I reproduce the published scores immediately?**  
Not yet: confirmed GASST benchmark settings, target scores, and matching pretrained artifacts still need to be supplied. Start with the demo to validate the environment and record a new experiment.

**Do I need a GPU?**  
The current entry point runs on CPU. No GPU configuration is required by the demo.

**Can I load older weights?**  
Compatible `state_dict` weights can be loaded with the same model configuration. Parameter attribute names are preserved. Full-object pickle checkpoint compatibility is not guaranteed.

**How should I report a problem?**  
Include the command, Python/PyTorch versions, and the generated `console.log` and `manifest.json`. Remove local paths you do not wish to share.

## 💖 Acknowledgments & License

This release is organized from the existing DSFormer implementation. Original authorship, source acknowledgments, and references are retained in [UPSTREAM_README.md](docs/UPSTREAM_README.md).

The code retains the original [MIT License](LICENSE) and copyright notice. Please preserve these notices when redistributing the source.

---

<div align="center">
<sub>GASST · Hyperspectral Image Classification</sub>
</div>
