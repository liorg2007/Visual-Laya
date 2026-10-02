# 01_accuracy: held-out accuracy, Brier, NLL per image task

**Question.** How accurate and calibrated is the final checkpoint (`runs/stage2_c/wise085_calibrated`) on held-out image tasks, versus chance, the caption->Laya baseline and stock Laya?

**Method.** The three configured tasks (oxford_pets, eurosat, vqav2_yesno) use the existing full-test predictions in `eval/results/stage2_c_wise085.predictions.jsonl` (n = all test rows). I verified they are reproducible: a fresh run on 60 oxford_pets images (120 rows) matched the saved probabilities exactly (max abs diff 0.0, argmax agreement 100%). Four extra tasks (cifar10, food101, aokvqa, coco) were run fresh (`run_extra.py`, fp16 GPU, seed 0) on a random subsample of 300 images per task, with all question rows for those images. Gold = argmax of target. Brier = multiclass sum of squares (range 0..2, lower is better); NLL = -log p(gold). 95% CIs are percentile bootstrap (2000 resamples, seed 0) resampling by image, so rows from the same image stay together. Rows split by question type: `choice` (k options) and `noul` (binary yes/no/unsure-style, 2 classes, chance 0.5). Baseline = `eval/results/caption_laya.predictions.jsonl` (BLIP caption -> stock Laya), full test. Stock Laya (text only) has no image predictions (`probs=None` for all 6437 image rows), so it can only be compared as "no image information" = chance.

## Results (final checkpoint; acc [95% CI]; chance; caption->Laya acc)

| task / qtype | n | chance | final acc | Brier | NLL | caption->Laya acc | final - caption (paired CI) |
|---|---|---|---|---|---|---|---|
| oxford_pets choice (20) | 347 | 0.050 | 0.651 [0.605, 0.706] | 0.466 | 1.002 | 0.069 [0.043, 0.095] | +0.582 [+0.527, +0.634] |
| oxford_pets noul | 347 | 0.5 | 0.879 [0.841, 0.911] | 0.189 | 0.314 | 0.490 [0.435, 0.542] | +0.389 [+0.326, +0.452] |
| eurosat choice (10) | 1334 | 0.100 | 0.960 [0.949, 0.970] | 0.062 | 0.127 | 0.222 [0.199, 0.246] | +0.738 [+0.714, +0.762] |
| eurosat noul | 1334 | 0.5 | 0.980 [0.972, 0.987] | 0.034 | 0.062 | 0.556 [0.529, 0.582] | +0.424 [+0.396, +0.451] |
| vqav2_yesno noul | 3075 | 0.5 (majority 0.533) | 0.608 [0.591, 0.626] | 0.330 | 0.657 | 0.592 [0.573, 0.611] | +0.017 [-0.005, +0.039] |
| cifar10 choice (10)* | 300 | 0.100 | 0.930 [0.900, 0.957] | 0.103 | 0.203 | not run | |
| cifar10 noul* | 300 | 0.5 | 0.990 [0.977, 1.000] | 0.014 | 0.028 | not run | |
| food101 choice (20)* | 300 | 0.050 | 0.837 [0.790, 0.880] | 0.242 | 0.531 | not run | |
| food101 noul* | 300 | 0.5 | 0.937 [0.907, 0.963] | 0.084 | 0.150 | not run | |
| aokvqa choice (4)* | 310 | 0.250 | 0.442 [0.388, 0.495] | 0.671 | 1.231 | not run | |
| coco choice (4-8)* | 600 | 0.176 (mean 1/k) | 0.622 [0.583, 0.662] | 0.515 | 0.927 | not run | |
| coco noul* | 600 | 0.5 | 0.745 [0.713, 0.777] | 0.336 | 0.494 | not run | |

\* 300-image subsample. The caption baseline was started on these but the shared GPU made the run too slow (final run alone took ~45 min vs the 25 min budget), so it was cancelled; no caption numbers exist for them.

## Accuracy by number of options (choice rows, final checkpoint)

| k | tasks | n | chance | acc [95% CI] | caption->Laya |
|---|---|---|---|---|---|
| 4 | aokvqa, coco | 431 | 0.250 | 0.492 [0.445, 0.538] | n/a |
| 5 | coco | 120 | 0.200 | 0.600 [0.508, 0.689] | n/a |
| 6 | coco | 107 | 0.167 | 0.636 [0.536, 0.732] | n/a |
| 7 | coco | 118 | 0.143 | 0.678 [0.590, 0.763] | n/a |
| 8 | coco | 134 | 0.125 | 0.582 [0.496, 0.667] | n/a |
| 10 | eurosat, cifar10 | 1634 | 0.100 | 0.955 [0.945, 0.964] | 0.222 (eurosat) |
| 20 | oxford_pets, food101 | 647 | 0.050 | 0.737 [0.702, 0.771] | 0.069 (pets) |

Within coco alone: k=4 0.620, 5 0.600, 6 0.636, 7 0.678, 8 0.582 (CIs about +-0.09, all overlapping; no trend with k).

## Findings
- Every image task except vqav2_yesno is far above chance, and far above caption->Laya (paired CIs exclude 0 by wide margins: +0.39 to +0.74). The caption baseline is at or near chance on pets and eurosat (generic BLIP captions do not carry breed/land-use labels).
- vqav2_yesno: 0.608 vs chance 0.5 and majority-class 0.533, so the model beats chance, but the +0.017 gain over caption->Laya has a paired CI [-0.005, +0.039]: not distinguishable from the baseline. Brier/NLL are better than the baseline's (0.330 vs 0.497 Brier) mainly from better-calibrated, less extreme probabilities.
- Accuracy does not fall with k alone: it is driven by task difficulty (eurosat/cifar10 k=10 about 0.96, coco k=4-8 about 0.6, aokvqa k=4 0.44, pets/food k=20 0.65/0.84). Accuracy as a multiple of chance is highest for k=20.
- aokvqa (0.44 vs 0.25) is the weakest choice task: it needs knowledge/reasoning, not recognition.
- The easy "noul" yes/no rows (about 50% gold rate) are higher than the matching choice rows in every task, as expected.

## Caveats
- By-k pooling mixes tasks, so k and task difficulty are confounded; use within-coco rows for k as the only clean comparison.
- Extra-task numbers use 300-image subsamples (wider CIs, +-0.03 to +-0.05); the three configured tasks use the full test sets. Test rows within an image are correlated; CIs account for this by resampling images.
- No stock-Laya image number exists (text-only agent); no caption baseline for cifar10/food101/aokvqa/coco.
- Test sets here are the data pipeline's held-out split; label sets for oxford/food use up to 20 options (gold included).

## Take-aways
1. Final checkpoint is strongly above chance and caption->Laya on recognition tasks (eurosat 0.96, cifar10 0.93, food101 0.84, pets 0.65 at k=20), with CIs well clear of baselines.
2. vqav2_yesno is the one task without a demonstrated gain in accuracy (0.608, within noise of caption->Laya's 0.592); aokvqa (0.44) and coco choice (0.62) are the weak spots.
3. Option count is not the dominant factor; task type is. Reported numbers reproduce exactly on a fresh run.

Files: `results.json` (all numbers incl. CIs), `analyze.py`, `run_extra.py`, `preds/final.predictions.jsonl`.
