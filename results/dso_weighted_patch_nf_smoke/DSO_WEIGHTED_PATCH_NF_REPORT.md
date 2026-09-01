# DSO-Weighted Direct Patch Normal Flow Report

## Protocol

- Sequence: TartanAir Office
- Development pairs: 2
- Evaluation pairs: 3
- Grid step: 80
- Patch radius: 2
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
| classical_nf | 24 | 4.7621 | 2.2068 | 11.9531 | 6.9493 | 4.7621 | 0.833 |
| patch_nf | 23 | 2.3328 | 1.3684 | 6.0503 | 3.2492 | 2.3328 | 0.609 |
| dso_patch_lambda0 | 23 | 2.3328 | 1.3684 | 6.0503 | 3.2492 | 2.3328 | 0.609 |
| dso_patch_nf | 23 | 2.3547 | 1.2684 | 6.0819 | 3.2828 | 2.3547 | 0.609 |
| dso_oriented_patch_nf | 23 | 2.3851 | 1.2684 | 6.0819 | 3.3170 | 2.3851 | 0.609 |

## Runtime

- DSO selector: 517.516 ms/frame
- Confidence map: 2.019 ms/frame
- classical_nf: 4.220 ms/frame
- patch_nf: 14.483 ms/frame
- dso_patch_lambda0: 14.243 ms/frame
- dso_patch_nf: 14.242 ms/frame
- dso_oriented_patch_nf: 14.987 ms/frame

## Sanity Checks

- `dso_patch_lambda0` max scalar difference from ordinary `patch_nf`: 0

## Research Questions

- RQ1: Patch NF vs classical NF: patch mean error is 2.3328, classical is 4.7621.
- RQ2: DSO weighting vs patch NF: mean-error delta `patch - dso_patch` is -0.0219. Positive means DSO weighting helped.
- RQ3: Nearby non-DSO pixels: delta `patch - dso_patch` is 0.0074.
- RQ4: See `plots/error_vs_distance_to_dso.png` and `plots/dso_improvement_vs_distance.png`.
- RQ5: Orientation gating delta `dso_patch - dso_oriented` is -0.0304. Positive means orientation gating helped.
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
