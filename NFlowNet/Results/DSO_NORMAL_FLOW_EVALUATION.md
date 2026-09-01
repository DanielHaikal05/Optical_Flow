# DSO Sparse Normal-Flow Evaluation on TartanAir Office

## Setup

DSO does not output a dense optical-flow or normal-flow image. I added a headless DSO runner and an output wrapper that exports:

- `poses.csv`: tracked camera poses for each input frame.
- `keyframe_points.csv`: DSO sparse point snapshots from `publishKeyframes`, split into `active`, `marginalized`, `outlier`, and `immature` point groups.

The evaluator projects each DSO point from host frame `i` into frame `i+1`, computes the full pixel displacement, projects that displacement onto the TartanAir GT gradient direction, and compares it with the GT normal-flow scalar/vector sampled at the same sparse point location.

Run command:

```bash
python3 NFlowNet/evaluate_dso_normal_flow_tartanair.py --run-dso
```

Camera assumption:

| Parameter | Value |
|---|---:|
| Resolution | 640 x 480 |
| fx | 320.0 |
| fy | 320.0 |
| cx | 320.0 |
| cy | 240.0 |

TartanAir calibration was not present in the local Office folder, so this uses the standard 640x480, 90-degree-FOV pinhole assumption.

## Timing

| Stage | Time |
|---|---:|
| DSO run | 30.251 s |
| DSO time per input frame | 49.430 ms/frame |
| Sparse projection + GT evaluation | 139.089 s |
| Evaluation time per evaluated sparse point | 0.0886 ms/point |

DSO processed 612 input frames and evaluated 1,570,588 sparse point snapshot rows against 611 GT frame pairs.

## Snapshot Metrics

Snapshot metrics count every exported DSO point snapshot. Non-final rows can include repeated evolving estimates of the same physical point.

| DSO output | Points | RMSE px/frame | AEPE px/frame |
|---|---:|---:|---:|
| active | 340,614 | 4.223 | 1.194 |
| marginalized | 389,717 | 4.476 | 1.268 |
| outlier | 105,322 | 6.865 | 2.435 |
| immature | 734,935 | 11.234 | 6.285 |
| all final | 125,009 | 3.995 | 1.102 |
| all non-final | 1,445,579 | 8.708 | 3.901 |
| all | 1,570,588 | 8.430 | 3.678 |

## Latest Unique Metrics

Latest-unique metrics keep the latest estimate for each `(output label, host frame, rounded pixel)` so repeated snapshots do not dominate the aggregate.

| DSO output | Points | RMSE px/frame | AEPE px/frame |
|---|---:|---:|---:|
| active latest unique | 117,712 | 4.139 | 1.162 |
| marginalized latest unique | 102,084 | 3.590 | 0.979 |
| outlier latest unique | 19,433 | 6.147 | 2.012 |
| immature latest unique | 243,096 | 11.152 | 6.191 |
| all final latest unique | 125,009 | 3.995 | 1.102 |
| all latest unique | 354,738 | 9.545 | 4.634 |

## Notes

- RMSE is scalar normal-flow RMSE in pixels/frame.
- AEPE is endpoint error between predicted sparse normal-flow vectors and GT normal-flow vectors, sampled at DSO point locations. Since both vectors lie along the same sampled GT gradient direction, AEPE equals the scalar absolute error for this projection-based evaluation.
- The sparse DSO numbers are not directly comparable to dense/grid-sampled NF and RAFT results because DSO chooses its own high-information point locations.
- The most meaningful DSO outputs are `active` and `marginalized`; `immature` uses the midpoint of DSO's inverse-depth bracket, and `outlier` is included for completeness rather than as a desired output.

## Artifacts

| Artifact | Path |
|---|---|
| Aggregate JSON | `NFlowNet/Results/dso_office_metrics.json` |
| Aggregate CSV | `NFlowNet/Results/dso_office_metrics.csv` |
| Per-frame metrics CSV | `NFlowNet/Results/dso_office_frame_metrics.csv` |
| Sparse normal-flow point CSV | `NFlowNet/Results/dso_office_sparse_normal_flow_points.csv` |
| DSO pose export | `NFlowNet/Results/dso_office/poses.csv` |
| DSO point export | `NFlowNet/Results/dso_office/keyframe_points.csv` |
| DSO log | `NFlowNet/Results/dso_office/dso_stdout.log` |
