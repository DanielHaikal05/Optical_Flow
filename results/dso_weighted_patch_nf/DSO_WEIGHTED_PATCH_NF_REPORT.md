# DSO-Weighted Direct Patch Normal Flow Report

## Protocol

- Sequence: TartanAir Office
- Development pairs: 5
- Evaluation pairs: 80
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

| Method | RMSE px/frame | AEPE px/frame | Compute time |
|---|---:|---:|---:|
| raft_small | 1.6925 | 0.9176 | 89.864 ms/frame |
| classical_nf | 15.4250 | 10.3321 | 3.861 ms/frame |
| patch_nf | 15.4585 | 9.9772 | 24.835 ms/frame |
| dso_patch_nf | 15.3821 | 9.9271 | 577.347 ms/frame |
| dso_oriented_patch_nf | 15.4221 | 9.9501 | 578.177 ms/frame |

RAFT-small is included as a same-protocol learned baseline: same Office eval split, same grid-selected points, and the same projection of dense optical flow onto the image-gradient normal direction.

## Runtime

- DSO selector: 549.342 ms/frame
- Confidence map: 2.205 ms/frame
- classical_nf: 3.861 ms/frame
- patch_nf: 24.835 ms/frame
- dso_patch_lambda0: 25.234 ms/frame
- dso_patch_nf: 577.347 ms/frame end-to-end
- dso_oriented_patch_nf: 578.177 ms/frame end-to-end
- raft_small: 89.864 ms/frame total, 75.176 ms/frame model-forward only

## Sanity Checks

- `dso_patch_lambda0` max scalar difference from ordinary `patch_nf`: 0

## Research Questions

- RQ1: Patch NF vs classical NF: patch mean error is 9.9772, classical is 10.3321. Patch matching improves mean error and bad-pixel rate, though scalar RMSE is nearly unchanged.
- RQ2: DSO weighting vs patch NF: mean-error delta `patch - dso_patch` is 0.0501 px. This is positive but very small, so the fixed-parameter experiment does not provide strong evidence of a meaningful global improvement.
- RQ3: Nearby non-DSO pixels: delta `patch - dso_patch` is 0.0037 px. This is essentially negligible, so the desired propagation effect to nearby non-DSO pixels is not supported by this run.
- RQ4: See `plots/error_vs_distance_to_dso.png` and `plots/dso_improvement_vs_distance.png`.
- RQ5: Orientation gating delta `dso_patch - dso_oriented` is -0.0230 px. Negative means orientation gating slightly hurt globally at the default settings.
- RQ6: DSO weighting adds selector/confidence overhead plus the weighted patch estimator cost shown above.

RAFT-small remains much stronger than the deterministic methods on this protocol: RMSE 1.6925 and AEPE 0.9176, versus the best DSO-weighted deterministic RMSE 15.3821 and AEPE 9.9271.

## Region Analysis

| Region | Patch mean error | DSO patch mean error | Oriented DSO patch mean error | Patch - DSO |
|---|---:|---:|---:|---:|
| Exact DSO-selected points | 7.5018 | 7.4969 | 7.4987 | 0.0049 |
| Nearby non-DSO points | 7.4315 | 7.4278 | 7.4181 | 0.0037 |
| Distant points | 18.3576 | 18.1548 | 18.2753 | 0.2028 |

The most important region is nearby non-DSO points. The observed gain there is only 0.0037 px, so the main hypothesis that DSO anchors improve nearby non-DSO patch estimates is not supported in this first pass.

## Patch-Size Check

The development sweep suggested that patch size mattered more than DSO weighting, so I also ran the same evaluation split with a 7x7 patch (`patch_radius=3`).

| Method | Mean error | Median error | P90 error | RMSE | AEPE | Bad >1 px |
|---|---:|---:|---:|---:|---:|---:|
| classical_nf | 10.3321 | 5.6603 | 30.1240 | 15.4250 | 10.3321 | 0.808 |
| patch_nf, 7x7 | 9.9397 | 4.5743 | 30.3703 | 15.7438 | 9.9397 | 0.702 |
| dso_patch_nf, 7x7 | 9.9243 | 4.5743 | 30.4421 | 15.7239 | 9.9243 | 0.703 |
| dso_oriented_patch_nf, 7x7 | 9.9908 | 4.6034 | 30.4455 | 15.8307 | 9.9908 | 0.703 |

With 7x7 patches, DSO weighting again gives only a tiny mean-error gain (`0.0154 px`) and orientation gating hurts globally.

## Parameter Sweep

Saved 51 one-parameter development-sweep rows to `summary_sweep.csv`.

The development sweep did not justify increasing DSO weighting: the best lambda row was `lambda=0`, which is equivalent to ordinary patch NF. The sweep did suggest that larger patches can improve median/mean behavior, but that is a patch-estimator effect, not a DSO-weighting effect.

## Outputs

- `raw_results.csv`: one row per evaluated point and method.
- `summary.csv`: aggregate, distance-bin, and DSO-region metrics.
- `summary.json`: full machine-readable result.
- `raft_small_summary.json`: same-protocol RAFT-small aggregate metrics.
- `raft_small_raw_results.csv`: one row per evaluated RAFT-small point.
- `debug/dso_selected_points.png`: selector overlay on the exact NF image coordinates.
- `debug/dso_confidence_heatmap.png`: DSO confidence field.
- `debug/dso_patch_nf_debug_panel.png`: RGB/selector/confidence/arrow diagnostic panel.
- `debug/dso_patch_nf_error_points.png`: DSO-patch point error overlay.
- `plots/`: summary plots.

Additional plots include error vs DSO confidence, image-gradient magnitude, GT normal displacement, lambda, sigma, patch radius, orientation gamma, and runtime by stage.

## Commands

```bash
python3 -m py_compile Camera_NF.py dso_pixel_selector/selector.py patch_normal_flow.py dso_weighted_patch_normal_flow.py evaluate_dso_patch_nf.py visualize_dso_patch_nf.py
python3 - <<'PY'
from tests.test_dso_weighted_patch_nf import test_dso_lambda_zero_matches_patch, test_synthetic_translation_recovers_normal_displacement
test_dso_lambda_zero_matches_patch()
test_synthetic_translation_recovers_normal_displacement()
print('sanity tests passed')
PY
python3 evaluate_dso_patch_nf.py --dev-pairs 5 --eval-pairs 80 --output-root results/dso_weighted_patch_nf
python3 visualize_dso_patch_nf.py --results-root results/dso_weighted_patch_nf
python3 evaluate_raft_small_dso_patch_protocol.py --dev-pairs 5 --eval-pairs 80 --output-root results/dso_weighted_patch_nf
python3 evaluate_dso_patch_nf.py --dev-pairs 5 --eval-pairs 80 --skip-sweep --patch-radius 3 --output-root results/dso_weighted_patch_nf_patch7
python3 visualize_dso_patch_nf.py --results-root results/dso_weighted_patch_nf_patch7
```
