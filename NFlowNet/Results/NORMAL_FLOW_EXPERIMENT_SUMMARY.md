# Normal-Flow Experiment Summary

Date: 2026-08-28

## Scope

We evaluated normal flow on the TartanAir `Office` sequence. The main comparison now focuses on five methods:

- RAFT-small as the learned optical-flow reference.
- The original deterministic normal-flow baseline.
- The best DSO-inspired deterministic normal-flow variant.
- DSO-weighted patch normal flow.
- Actual DSO output converted into sparse normal flow.

RMSE is scalar normal-flow error in pixels/frame. AEPE is endpoint error between predicted normal-flow vectors and GT normal-flow vectors, also in pixels/frame.

## Methods

| Method | Brief description |
|---|---|
| RAFT-small | TorchVision RAFT-small predicts dense full optical flow. We project its full flow onto the GT image-gradient direction before evaluating it as normal flow. |
| NF baseline | The original deterministic method in `Camera_NF.py`: Scharr spatial gradients plus temporal brightness difference, using `v_n = -It / (Ix^2 + Iy^2) * [Ix, Iy]`. |
| DSO-derived NF | Patch-based 1-D photometric search along the image-gradient normal, with affine brightness compensation. This does not run full DSO. |
| DSO-weighted patch NF | Patch-based 1-D photometric search where patch residuals are weighted by a DSO-style frontend confidence map from sparse high-gradient pixel selection. It does not use DSO pose/depth. |
| DSO | Actual DSO run on the Office sequence. DSO estimates sparse map points and poses, so we project each DSO point into the next frame and evaluate normal flow only at those sparse DSO-selected locations. |

## Results

| Method | RMSE px/frame | AEPE px/frame | Compute time |
|---|---:|---:|---:|
| RAFT-small | 2.634 | 0.757 | 70.70 ms/pair |
| NF baseline | 11.295 | 6.610 | 13.36 ms/pair |
| DSO-derived NF | 11.217 | 5.625 | 79.28 ms/pair |
| DSO-weighted patch NF | 15.382 | 9.927 | 577.35 ms/frame |
| DSO | 3.590 | 0.979 | 49.43 ms/frame |

## Interpretation

RAFT-small is the strongest dense/grid-evaluated method. The DSO-inspired deterministic variant improves over the baseline, mostly in AEPE, but only modestly: it does not close the gap to RAFT-small.

The DSO-weighted patch experiment was run on a smaller same-protocol subset using full TartanAir optical-flow GT projected onto the image-gradient normal. Its end-to-end compute time is 577.35 ms/frame: 25.80 ms/frame for the weighted patch estimator plus 551.55 ms/frame for the Python DSO-style selector and confidence map. It gave only a tiny improvement over ordinary patch NF on that subset, so the current evidence does not support DSO-style confidence weighting as a major boost.

The actual DSO result is strong, but it is not directly dense-comparable to the other three rows. DSO chooses sparse high-information points, then we derive normal flow from DSO's estimated point depths and camera poses. That means the DSO metric answers: "How accurate is normal flow at DSO's selected points?", not "How accurate is a full-frame normal-flow field?"

## Artifacts

| Artifact | Path |
|---|---|
| Focused summary | `NFlowNet/Results/NORMAL_FLOW_EXPERIMENT_SUMMARY.md` |
| Camera_NF / RAFT metrics | `NFlowNet/Results/camera_nf_variants_vs_raft_small_office_metrics.json` |
| DSO metrics | `NFlowNet/Results/dso_office_metrics.json` |
| DSO sparse evaluation report | `NFlowNet/Results/DSO_NORMAL_FLOW_EVALUATION.md` |
| DSO sparse normal-flow points | `NFlowNet/Results/dso_office_sparse_normal_flow_points.csv` |
| DSO-weighted patch report | `results/dso_weighted_patch_nf/DSO_WEIGHTED_PATCH_NF_REPORT.md` |
