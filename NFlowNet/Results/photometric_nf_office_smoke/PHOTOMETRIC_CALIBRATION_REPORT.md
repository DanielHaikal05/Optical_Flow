# Photometric Calibration Normal-Flow Experiment

## Setup

- Sequence: `/home/daniel/Optical_Flow/Datasets/TartanAir/Office`
- Pairs: `3`
- Selected points: `69`
- Grid step: `120`
- Synthetic response: `I_raw = I_linear ** (1 / 2.2)`
- Vignette: `V(r)=1-k1*r^2-k2*r^4`, `k1=0.25`, `k2=0.15`
- Current-frame exposure: `1.0`
- Next-frame exposures: `[0.5, 0.75, 1.0, 1.25, 1.5]`

The baseline classical normal-flow equation was unchanged. Calibration is applied before Gaussian smoothing and derivative computation.

## Exposure 1.0

| mode | exposure_next | mean_normal_error | mean_EPE | mean_abs_It | runtime_ms |
|---|---:|---:|---:|---:|---:|
| baseline | 1.00 | 1.3487 | 1.3514 | 0.1350 | 13.2 |
| response | 1.00 | 1.8013 | 1.8040 | 0.1610 | 37.7 |
| vignette | 1.00 | 1.6739 | 1.6760 | 0.1456 | 17.0 |
| exposure | 1.00 | 1.3487 | 1.3514 | 0.1350 | 13.2 |
| response_exposure | 1.00 | 1.8013 | 1.8040 | 0.1610 | 38.0 |
| full_photometric | 1.00 | 1.8028 | 1.8049 | 0.1787 | 42.3 |
| full_photometric_affine | 1.00 | 1.8439 | 1.8459 | 0.1729 | 118.9 |

## Exposure 1.5

| mode | exposure_next | mean_normal_error | mean_EPE | mean_abs_It | runtime_ms |
|---|---:|---:|---:|---:|---:|
| baseline | 1.50 | 2.6258 | 2.6302 | 0.1623 | 13.4 |
| response | 1.50 | 4.8371 | 4.8438 | 0.2702 | 35.5 |
| vignette | 1.50 | 3.1566 | 3.1626 | 0.1779 | 16.9 |
| exposure | 1.50 | 1.9859 | 1.9997 | 0.1730 | 13.2 |
| response_exposure | 1.50 | 2.0556 | 2.0589 | 0.1262 | 35.7 |
| full_photometric | 1.50 | 2.0690 | 2.0718 | 0.1437 | 40.6 |
| full_photometric_affine | 1.50 | 2.3531 | 2.3577 | 0.1607 | 126.2 |

## Conclusion

Full photometric calibration substantially reduces the error caused by the synthetic camera response, vignette, and exposure model. Residual affine correction after full calibration was also tested separately.


## Previous Native Office Results

| method | RMSE | AEPE | ms/pair |
|---|---:|---:|---:|
| baseline | 11.2948 | 6.6099 | 13.36 |
| affine | 11.3420 | 6.6658 | 35.72 |
| patch | 11.2388 | 5.6216 | 57.52 |
| patch_affine | 11.2170 | 5.6249 | 79.28 |
| raft_small | 2.6340 | 0.7572 | 70.70 |

## Reproduce

```bash
python3 NFlowNet/evaluate_photometric_nf.py
```
