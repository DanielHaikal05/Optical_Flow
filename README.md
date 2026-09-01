# Optical Flow Workspace

This workspace collects experiments and integrations for normal flow, event-camera motion segmentation, depth-assisted flow, and DSO-based evaluation.

## Core Work

- `Camera_NF.py`, `patch_normal_flow.py`, `dso_weighted_patch_normal_flow.py` - normal-flow implementations and compatibility entry points.
- `evaluate_dso_patch_nf.py`, `evaluate_raft_small_dso_patch_protocol.py`, `visualize_dso_patch_nf.py` - DSO patch normal-flow evaluation and visualization scripts.
- `dso_pixel_selector/` - Python utilities for DSO-style pixel selection.
- `tests/` - focused sanity tests for the DSO-weighted patch normal-flow code.
- `tools/` - local orchestration and evaluation utilities.
- `run_scripts/` - experiment launch scripts used for remote/DGX runs.

## Integrated Projects

These directories are upstream or adapted project checkouts with local changes:

- `dso/` - Direct Sparse Odometry changes for headless/normal-flow output.
- `NFlowNet/` - normal-flow network training, evaluation, and metrics scripts.
- `E-MoFlow/` - event motion-flow training/evaluation changes.
- `EvMotionSeg/` - event motion segmentation experiments and standalone tooling.
- `VecKM_flow/` - VecKM flow/egomotion experiments.
- `depthanyevent/` - depth-assisted event-camera inference scripts.
- `event_suppression/` - event suppression model and evaluation code.
- `event-cam-prop-tracker/` - small event-camera propagation tracker checkout.

## Data And Results

Large data and generated research outputs are intentionally separated from source:

- `Datasets/` - local datasets. This is very large and should usually be omitted when sharing code.
- `results/`, `res_d/`, `analysis/`, `Results_Drone/`, `Results_MVSEC/`, `NFlowNet/Results/` - selected lightweight reports, plots, metrics, summaries, and handoff artifacts are tracked.
- `E-MoFlow/outputs/`, `EvMotionSeg/data/`, `VecKM_flow/outputs/` - generated outputs inside embedded project checkouts remain local-only unless those checkouts are flattened or handled as separate repositories.
- `event_suppression/checkpoints/`, `depthanyevent/weights/`, `VecKM_flow/train/checkpoints/`, `VecKM_flow/train/model_checkpoints/` - model weights/checkpoints.

## Sharing Notes

The workspace has been cleaned of local caches, Python bytecode, a local virtualenv, CMake build output, and log files. For a lightweight code handoff, share the source directories, tracked result summaries/plots, and omit `Datasets/`, frame outputs, raw arrays, videos, large output folders, and checkpoints unless the recipient specifically needs them.

Run the root tests with:

```bash
pytest tests
```
