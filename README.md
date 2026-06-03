# [TOG (SIGGRAPH 2026)] Generative Modeling with Orbit-Space Particle Flow Matching (OGPP)

by [Sinan Wang](https://sinanw.com/)\*, [Jinjin He](https://jinjinhe2001.github.io/)\*, Shenyifan Lu, [Ruicheng Wang](https://wrc042.github.io/), [Greg Turk](https://faculty.cc.gatech.edu/~turk/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/)

\* indicates equal contribution (co-first authors).

Our paper and video results can be found at our [project website](https://ogpp.sinanw.com/).


| Domain | Train | Inference |
|---|---|---|
| **Blue noise** (2D blue-noise point sets) | `scripts/run_blue_noise.py` | `scripts/eval_blue_noise.py` |
| **Shape** (3D point cloud / mesh) | `scripts/run_shape.py` | `scripts/eval_shape.py` |
| **DLA** (2D diffusion-limited aggregation) | `scripts/run_dla.py` | `scripts/eval_dla.py` |
| **Thomson** (points on a sphere) | `scripts/run_thomson_multi.py` | `scripts/eval_thomson_multi.py` |
| **Minimal surface** | `scripts/run_minimalsurface_*.py` | `scripts/eval_minimal_surface*.py` |

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

Each script defines its own arguments — run `python scripts/<script>.py -h` to
list them and add flags (e.g. `--exp_name`, `--batch_size`) as needed.

```bash
# blue noise (2D point sets, single GPU, see run_blue_noise_multiGPU.py for the multiple GPU script)
python scripts/run_blue_noise.py

# DLA
python scripts/run_dla.py

# Thomson
python scripts/run_thomson_multi.py

# 3D shape
python scripts/run_shape.py --n_points 2048

# minimal surface — fixed 3 anchors (--mode default|minibatch_ot|eqotfm)
python scripts/run_minimalsurface_multimode.py --data_path <3_anchors.npz>

# minimal surface — variable 3-8 anchors
python scripts/run_minimalsurface_variable.py --data_path <variable_anchors.npz>
```

## Inference

Each domain has a matching `eval_*.py` entry point that loads a trained
checkpoint and samples over a sweep of sampling steps:

```bash
python scripts/eval_blue_noise.py --ckpt <path/to/checkpoint.pt>  # blue noise (2D point sets)
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

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{wang2026generative,
  title={Generative Modeling with Orbit-Space Particle Flow Matching},
  author={Wang, Sinan and He, Jinjin and Lu, Shenyifan and Wang, Ruicheng and Turk, Greg and Zhu, Bo},
  journal={ACM Transactions on Graphics (TOG)},
  volume={45},
  number={4},
  pages={1--27},
  year={2026},
  publisher={ACM New York, NY, USA}
}
```

## Acknowledgments

Parts of our flow-matching / diffusion training and sampling code are adapted
from the starter code of MIT course 6.S184 *Introduction to Flow Matching and
Diffusion Models* ([course website](https://diffusion.csail.mit.edu/2026/index.html)).
We thank the course staff for making these materials publicly available.

## License

See [`LICENSE`](LICENSE).
