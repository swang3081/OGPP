# [TOG (SIGGRAPH 2026)] Generative Modeling with Orbit-Space Particle Flow Matching (OGPP)

by [Sinan Wang](https://sinanw.com/)\*, [Jinjin He](https://jinjinhe2001.github.io/)\*, Shenyifan Lu, [Ruicheng Wang](https://wrc042.github.io/), [Greg Turk](https://faculty.cc.gatech.edu/~turk/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/)

\* indicates equal contribution (co-first authors).

Our paper and video results can be found at our [project website](https://ogpp.sinanw.com/).


| Domain | Train | Inference | Metrics |
|---|---|---|---|
| **Blue noise** (2D blue-noise point sets) | `scripts/run_blue_noise.py` | `scripts/eval_blue_noise.py` | GBN code (external) |
| **Shape** (3D point cloud / mesh) | `scripts/run_shape.py` | `scripts/eval_shape.py` | LION CD/EMD (external) |
| **DLA** (2D diffusion-limited aggregation) | `scripts/run_dla.py` | `scripts/eval_dla.py` | `scripts/eval_dla_metrics.py`, `scripts/evaluate_dla_folder.py` |
| **Thomson** (points on a sphere) | `scripts/run_thomson_multi.py` | `scripts/eval_thomson_multi.py` | `scripts/eval_thomson_metrics.py`, `scripts/evaluate_thomson_folder.py` |
| **Minimal surface** | `scripts/run_minimalsurface_*.py` | `scripts/eval_minimal_surface*.py` | `scripts/eval_minimal_surface_metrics.py`, `scripts/evaluate_minimal_surface_folder.py` |

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
- `paths.py`: conditional probability paths (linear, quadratic/cubic Hermite, GenVP)
- `models.py` / `models_conditional.py`: transformer & PVCNN velocity models
- `dynamics.py`: ODE/SDE samplers
- `trainers.py`: training loops (incl. async CPU x0/x1 prefetch for mesh)
- `datasets.py`: mesh nearest-point, Poisson-sphere, OT / mini-batch-OT datasets
- `distributions.py`, `voronoi.py`, `sort_numba.py`, `io.py`, `utils.py`

## Data
The datasets are available here:
[Google Drive](https://drive.google.com/drive/folders/13kfWJqhXrnrUBi1H5-ZSewGNlHNzrb_H?usp=sharing)

## Train example

Each script defines its own arguments: run `python scripts/<script>.py -h` to
list them and add flags (e.g. `--data_path`, `--exp_name`, `--batch_size`) as needed.

```bash
# blue noise (2D point sets, single GPU, see run_blue_noise_multiGPU.py for the multiple GPU script)
python scripts/run_blue_noise.py --data_path <blue_noise_npz_path> --n_points 1024 --epochs 1000
```

## Inference example

Each domain has a matching `eval_*.py` entry point that loads a trained
checkpoint and samples over a sweep of sampling steps:

```bash
python scripts/eval_blue_noise.py --ckpt <path/to/checkpoint.pt>  --n_points 1024 --n_point_set 32 --sample_steps 200
```

## Quantitative metrics

Three domains ship with quantitative-evaluation scripts (see the **Metrics**
column above). Each domain provides a **core metrics module** (the per-sample
metric definitions, also runnable on a single file) and a **folder evaluator**
that averages the metrics over all samples produced by the matching `eval_*.py`
run and writes a `metrics.txt`:

```bash
# DLA: fractal dimension, lacunarity, angular uniformity, ...
python scripts/evaluate_dla_folder.py --folder <eval_output>/step_20

# Thomson: Coulomb / spring energy, dimensionless E*, tangent force, spacing CV
python scripts/evaluate_thomson_folder.py --folder <eval_output>/step_20

# Minimal surface: area fraction, angle smoothness, curvature, uniformity
python scripts/evaluate_minimal_surface_folder.py --folder <eval_output>/step_20
```

The folder evaluator expects the sample sub-directory layout written by the
corresponding `eval_*.py` script (`ply/` for DLA / Thomson, `npz/` for minimal
surface). Run `python scripts/<script>.py -h` for the full flag list.

The remaining two domains are scored with **external** code rather than scripts
vendored here:

- **Shape (3D point cloud).** We use the Chamfer Distance / EMD metric kernels
  from [LION](https://github.com/nv-tlabs/LION) (`third_party/ChamferDistancePytorch/chamfer3D`
  and `third_party/PyTorchEMD`). See [`BASELINES.md`](BASELINES.md) for how to
  build them and make them importable.
- **Blue noise.** We use the spectral / radial blue-noise metrics from the
  **GBN (Gaussian Blue Noise)** reference implementation
  ([project page](https://abdallagafar.com/publications/gbn/)).

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
