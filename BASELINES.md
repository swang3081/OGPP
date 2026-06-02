# Baselines

For the comparisons in the paper we evaluated against the following point-cloud
generative models. They are **not** included in this repository — clone them
from their upstream sources and follow their own setup instructions.

| Baseline | Upstream |
|---|---|
| LION | https://github.com/nv-tlabs/LION |
| DiT-3D | https://github.com/DiT-3D/DiT-3D |
| PVD | https://github.com/alexzhou907/PVD |
| PSF (Point Straight Flow) | https://github.com/Lakonik/PSF |

## Metric kernels (Chamfer Distance / EMD)

Our metric scripts (e.g. `tests/eval_mesh_metrics.py`) import the CUDA
implementations of Chamfer Distance and Earth Mover's Distance shipped inside
LION's `third_party/` directory:

- `third_party/ChamferDistancePytorch/chamfer3D`
- `third_party/PyTorchEMD`

To run those metrics, clone LION, build those extensions for your PyTorch/CUDA
version, and make them importable (add LION's root to `PYTHONPATH`). The scripts
originally expected the LION checkout at `external/LION-main`; adjust the
`sys.path` insertion near the top of each metric script to point at your clone.

## Modifications we made

To build LION's CD/EMD extensions against modern PyTorch (2.x) and recent GPUs
we patched a few files (removed deprecated `THC` headers, added modern CUDA
macros, preferred prebuilt extensions). These patches are not redistributed
here; reproduce them against the upstream LION source if needed.
