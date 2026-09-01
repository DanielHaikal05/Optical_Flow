# DSO-Weighted Direct Patch Normal Flow Report

## Protocol

- Sequence: TartanAir Office
- Development pairs: 5
- Evaluation pairs: 80
- Grid step: 80
- Patch radius: 3
- DSO sigma: 4.0
- DSO lambda: 2.0
- Orientation gamma: 2.0

The DSO selector here is a Python DSO-style frontend selector: adaptive local gradient thresholds, deterministic block selection, and directional diversity. It does not use DSO pose, depth, tracking, or bundle adjustment.

## Ablation

| Method | Patch | DSO selector | DSO spatial weighting | Orientation gating |
|---|---:|---:|---:|---:|
| Classical NF | No | No | No | No |
| Patch NF | Yes | No | No | No |
| DSO Patch lambda=0 | Yes | Yes | No | No |
| DSO Patch | Yes | Yes | Yes | No |
| DSO Oriented Patch | Yes | Yes | Yes | Yes |

## Evaluation Results

| Method | Valid points | Mean error | Median error | P90 error | RMSE | AEPE | Bad >1 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| classical_nf | 1,222 | 10.3321 | 5.6603 | 30.1240 | 15.4250 | 10.3321 | 0.808 |
| patch_nf | 1,222 | 9.9397 | 4.5743 | 30.3703 | 15.7438 | 9.9397 | 0.702 |
| dso_patch_lambda0 | 1,222 | 9.9397 | 4.5743 | 30.3703 | 15.7438 | 9.9397 | 0.702 |
| dso_patch_nf | 1,222 | 9.9243 | 4.5743 | 30.4421 | 15.7239 | 9.9243 | 0.703 |
| dso_oriented_patch_nf | 1,222 | 9.9908 | 4.6034 | 30.4455 | 15.8307 | 9.9908 | 0.703 |

## Runtime

- DSO selector: 531.143 ms/frame
- Confidence map: 2.219 ms/frame
- classical_nf: 3.809 ms/frame
- patch_nf: 25.161 ms/frame
- dso_patch_lambda0: 25.234 ms/frame
- dso_patch_nf: 25.608 ms/frame
- dso_oriented_patch_nf: 26.600 ms/frame

## Sanity Checks

- `dso_patch_lambda0` max scalar difference from ordinary `patch_nf`: 0

## Research Questions

- RQ1: Patch NF vs classical NF: patch mean error is 9.9397, classical is 10.3321.
- RQ2: DSO weighting vs patch NF: mean-error delta `patch - dso_patch` is 0.0154. Positive means DSO weighting helped.
- RQ3: Nearby non-DSO pixels: delta `patch - dso_patch` is 0.0021.
- RQ4: See `plots/error_vs_distance_to_dso.png` and `plots/dso_improvement_vs_distance.png`.
- RQ5: Orientation gating delta `dso_patch - dso_oriented` is -0.0665. Positive means orientation gating helped.
- RQ6: DSO weighting adds selector/confidence overhead plus the weighted patch estimator cost shown above.

## Parameter Sweep

Saved 0 one-parameter development-sweep rows to `summary_sweep.csv`.

## Outputs

- `raw_results.csv`: one row per evaluated point and method.
- `summary.csv`: aggregate, distance-bin, and DSO-region metrics.
- `summary.json`: full machine-readable result.
- `debug/dso_selected_points.png`: selector overlay on the exact NF image coordinates.
- `debug/dso_confidence_heatmap.png`: DSO confidence field.
- `plots/`: summary plots.
