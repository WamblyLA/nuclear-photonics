# Common-image diagnostic

This is an abstract metric reference, **not a DOE, a physically valid baseline,
a submission candidate, or a quality upper bound**. It produces no phase mask
and uses no propagation, neural inference, optimization, or official test data.

For each of the previously used 256 selection quartets: independently peak-normalize
the four targets, take their arithmetic mean, peak-normalize that mean, save/read
an actual uint8 grayscale PNG, then use that identical image in all four planes.
Targets and metrics use the existing v2 protocol. Source IDs and manifest SHA256
are embedded in `common_image_diagnostic.json`; all paths are repository-relative.

R49 is its saved best-selection result at additional step 4000.
R56 is the latest completed validation row available at this diagnostic snapshot,
step 34000; it is not claimed to be the final or selected checkpoint.
No network was re-evaluated. Historical model metrics were computed on GPU;
the common reference used CPU Torch 2.6.0+cpu. Small numerical
backend differences are possible, so this is not a strict paired significance test.

| Group | Common SSIM | R49 SSIM | R56 SSIM | Common Pearson | R49 Pearson | R56 Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aggregate | 0.463232 | 0.295942 | 0.323258 | 0.619896 | 0.543870 | 0.567883 |
| CIFAR10 | 0.552920 | 0.465158 | 0.475348 | 0.502329 | 0.343277 | 0.393771 |
| MNIST | 0.571629 | 0.220233 | 0.277376 | 0.700104 | 0.664859 | 0.668126 |
| FashionMNIST | 0.394388 | 0.222847 | 0.248787 | 0.714945 | 0.667373 | 0.687158 |
| mixed | 0.333990 | 0.275530 | 0.291523 | 0.562206 | 0.499970 | 0.522479 |

The identical reconstruction has exactly zero mean diagonal-minus-off-diagonal
depth margin, both for SSIM and Pearson: every reconstruction row has the same
four target scores. Finite-precision maximum absolute observed margins are
5.96e-08 / 5.96e-08.

The common reference shows how much aggregate similarity can exist with no depth
separation. It does not quantify which fraction of a model's gain came from
separation, and surpassing or falling below it does not establish physical
feasibility or optimality. Assess actual depth separation with the model's
correct-versus-wrong-plane metric matrix and visual reconstructions alongside
the ordinary scores. The current comparison alone cannot justify calling a
higher SSIM a better separation of four objects.

Reproduce this snapshot's comparison row:

```powershell
.venv\Scripts\python.exe -m scripts.diagnose_common_image --history-step 34000
```

The full history-file SHA may change while training continues; the selected
validation row has its own SHA and line number. The target-source manifest is
fixed, and no checkpoint file is opened.
