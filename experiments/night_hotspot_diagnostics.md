# Hotspot diagnostics: peak-normalized protocol

CPU, same first 16 saved selection quadruples per group: 64 quadruples / 256 reconstructions per model. Not a fresh holdout; no test images were opened. All quantities use quantized phase masks; displayed metrics use reconstruction PNG roundtrips.

| Run | SSIM | Pearson | Raw amplitude cosine | Correct-wrong margin | Raw peak | Top 17 pixels energy fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r31_peak_fourier | 0.260593 | 0.514926 | 0.752201 | 0.037412 | 1.1580 | 0.006642 |
| r35_peak_transport | 0.374852 | 0.441937 | 0.746002 | 0.023542 | 4.4256 | 0.063628 |
| r37_peak_transport_random | 0.380531 | 0.315063 | 0.708479 | 0.008108 | 5.0904 | 0.046729 |

## MNIST contrast audit

| Run | Target FG mean | Predicted FG mean | Predicted BG mean | Raw peak | Peak in FG | Top 17 pixels energy fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r31_peak_fourier | 0.636422 | 0.392332 | 0.247347 | 1.513342 | 0.890625 | 0.010615 |
| r35_peak_transport | 0.636422 | 0.075698 | 0.020392 | 8.597211 | 0.921875 | 0.136118 |
| r37_peak_transport_random | 0.636422 | 0.063208 | 0.014126 | 9.255433 | 0.609375 | 0.090070 |

Foreground is target amplitude > 0.1, background its complement. Energy is raw amplitude squared; top 0.1% uses ceil(16384 * 0.001) = 17 pixels. Raw peaks share identical saved scales.

The large MNIST SSIM improvement coincides with much darker foreground and background, far larger raw peaks, and weaker correct-vs-wrong depth correlation margins. This supports a peak-normalization/contrast failure mode, not improved depth separation. R35 also shifts more raw energy into target foreground, so not every change is harmful; its total cosine is close to R31. R37 loses Pearson/cosine and has its MNIST maximum outside foreground in about 39% of cases.

A cosine auxiliary loss alone may not repair the displayed contrast: MNIST cosine improves from 0.5833 (R31) to 0.6079 (R35) despite severe foreground dimming. Balanced normalized foreground/background MSE directly penalizes this failure and should be evaluated against both SSIM and Pearson, without assuming improvement.

Sources: `night_hotspot_diagnostics.json`, per-image `night_hotspot_diagnostics.npz`; executable `src/hotspot_diagnostics.py`.
