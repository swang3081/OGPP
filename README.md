# [TOG (SIGGRAPH 2026)] Generative Modeling with Orbit-Space Particle Flow Matching

by [Sinan Wang](https://sinanw.com/)\*, [Jinjin He](https://jinjinhe2001.github.io/)\*, Shenyifan Lu, [Ruicheng Wang](https://wrc042.github.io/), [Greg Turk](https://faculty.cc.gatech.edu/~turk/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/)

\* indicates equal contribution (co-first authors).

Our paper and video results can be found at our [project website](https://ogpp.sinanw.com/).


| Domain | Train | Inference |
|---|---|---|
| **Shape** (3D point cloud / mesh) | `scripts/run_shape.py` | `scripts/eval_shape.py` |
| **DLA** (2D diffusion-limited aggregation) | `scripts/run_dla.py` | `scripts/eval_dla.py` |
| **Thomson** (points on a sphere) | `scripts/run_thomson_multi.py` | `scripts/eval_thomson_multi.py` |
| **Minimal surface** | `scripts/run_shape.py` (mesh mode) | `scripts/eval_minimal_surface*.py` |

## Setup

```bash
conda create -n flow_matching python=3.10
conda activate flow_matching
pip install -r requirements.txt
# Install a CUDA build of PyTorch matching your system, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`environment.yml` records the exact conda environment used for the paper (Windows / CUDA).

## Repository layout

```
flow_lab/      core library: probability paths, models, dynamics, trainers, datasets, utils
scripts/       train / inference entry points
```

`flow_lab/` module map:
- `paths.py` — conditional probability paths (linear, quadratic/cubic Hermite, GenVP)
- `models.py` / `models_conditional.py` — transformer & PVCNN velocity models
- `dynamics.py` — ODE/SDE samplers
- `trainers.py` — training loops (incl. async CPU x0/x1 prefetch for mesh)
- `datasets.py` — mesh nearest-point, Poisson-sphere, OT / mini-batch-OT datasets
- `distributions.py`, `voronoi.py`, `sort_numba.py`, `io.py`, `utils.py`

## Data
The datasets are available here:
[Google Drive](https://drive.google.com/drive/folders/13kfWJqhXrnrUBi1H5-ZSewGNlHNzrb_H?usp=sharing)

## Train

```bash
# 3D shape (single GPU)
python scripts/run_shape.py --exp_name shape_linear --batch_size 256 --n_points 2048

# 3D shape (multi-GPU)
python scripts/run_uniGBN_1024_10k_linear_uniform_multipleGPU.py --exp_name shape_mgpu --batch_size 256

# DLA (2D)
python scripts/run_dla.py --exp_name dla_ours

# Thomson
python scripts/run_thomson_multi.py --exp_name thomson_ours
```

## Inference

Each domain has a matching `eval_*.py` entry point that loads a trained
checkpoint and samples over a sweep of sampling steps:

```bash
python scripts/eval_dla.py             # DLA
python scripts/eval_shape.py           # 3D shape
python scripts/eval_thomson_multi.py   # Thomson
python scripts/eval_minimal_surface.py # minimal surface
python scripts/eval_minimal_surface_variable.py # minimal surface （variable anchors)
```

## Baselines

The third-party baselines (LION, DiT-3D, PVD, PSF) used for comparison are
**not** vendored here. See [`BASELINES.md`](BASELINES.md) for upstream links and
the modifications we applied.

## License

See [`LICENSE`](LICENSE).
