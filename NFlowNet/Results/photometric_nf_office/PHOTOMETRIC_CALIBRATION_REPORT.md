# Photometric Calibration Normal-Flow Experiment

## Setup

- Sequence: `/home/daniel/Optical_Flow/Datasets/TartanAir/Office`
- Pairs: `611`
- Selected points: `17843`
- Grid step: `80`
- Synthetic response: `I_raw = I_linear ** (1 / 2.2)`
- Vignette: `V(r)=1-k1*r^2-k2*r^4`, `k1=0.25`, `k2=0.15`
- Current-frame exposure: `1.0`
- Next-frame exposures: `[0.5, 0.75, 1.0, 1.25, 1.5]`

The baseline classical normal-flow equation was unchanged. Calibration is applied before Gaussian smoothing and derivative computation.

## Exposure 1.0

| mode | exposure_next | mean_normal_error | mean_EPE | mean_abs_It | runtime_ms |
|---|---:|---:|---:|---:|---:|
| baseline | 1.00 | 6.5578 | 6.5645 | 0.1318 | 2304.4 |
| response | 1.00 | 6.4737 | 6.4803 | 0.1400 | 8792.5 |
| vignette | 1.00 | 6.5937 | 6.6002 | 0.1431 | 3441.5 |
| exposure | 1.00 | 6.5578 | 6.5645 | 0.1318 | 2681.5 |
| response_exposure | 1.00 | 6.4737 | 6.4803 | 0.1400 | 8863.8 |
| full_photometric | 1.00 | 6.6035 | 6.6099 | 0.1492 | 9740.8 |
| full_photometric_affine | 1.00 | 6.6596 | 6.6658 | 0.1421 | 23705.9 |

## Exposure 1.5

| mode | exposure_next | mean_normal_error | mean_EPE | mean_abs_It | runtime_ms |
|---|---:|---:|---:|---:|---:|
| baseline | 1.50 | 6.9767 | 6.9928 | 0.1962 | 2299.7 |
| response | 1.50 | 7.2291 | 7.2522 | 0.2617 | 8659.3 |
| vignette | 1.50 | 7.1025 | 7.1202 | 0.2177 | 3435.4 |
| exposure | 1.50 | 7.4050 | 7.4160 | 0.1416 | 2677.6 |
| response_exposure | 1.50 | 6.7682 | 6.7747 | 0.1202 | 8743.8 |
| full_photometric | 1.50 | 6.8986 | 6.9050 | 0.1292 | 9625.0 |
| full_photometric_affine | 1.50 | 6.8508 | 6.8573 | 0.1400 | 24341.1 |

## Conclusion

Photometric calibration is useful here primarily as a recovery mechanism for known synthetic corruption. At exposure `1.0`, full calibration recovers the previous clean baseline almost exactly. At low exposure, response/exposure/full calibration reduce error clearly. At high exposure, saturation limits recovery; response+exposure is slightly better than full calibration, while full calibration plus residual affine helps somewhat at `1.5`.

This does not make the classical estimator competitive with RAFT-small on this sequence. It improves robustness to controlled photometric corruption, not the aperture/differential-motion limitations of the underlying normal-flow equation.


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
