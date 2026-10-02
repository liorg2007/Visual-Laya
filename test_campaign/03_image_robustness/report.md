# 03 Image robustness of Laya-Vision (stage2_c/wise085_calibrated)

**Question.** How much does accuracy (and confidence) degrade when test images are corrupted?

**Method.** Two image tasks from the held-out test sets: EuroSAT (10 classes, chance 0.10) and Oxford Pets (20 options, chance 0.05). Fixed subsample: the first n=150 records of the seed-0 shuffled test file per task (the same images in every condition). 22 conditions per task, defined in `run.py`: clean; rot90/rot180/hflip/vflip/grayscale; and for 8 families a mild and a severe level (Gaussian blur r=2/6, noise sigma=20/70, JPEG q=30/5, brightness x0.6/x0.25, contrast x0.5/x0.2, centre-crop keeping 70%/40%, down-then-up-scale x4/x12, black occlusion of 15%/40% of area; random positions and noise seeded per image). Metric: argmax of the answer probabilities vs gold; Wilson 95% CIs; also mean confidence (max prob) and agreement of the prediction with the clean prediction. Single-process, fp16 autocast on a shared GPU, results checkpointed per condition (`ckpt_n150.json`). Plot: `curves.png`; full table: `results.json` (`plot.py` prints it).

**Numbers (accuracy, 95% CI; mean confidence in brackets).** Clean: EuroSAT 0.980 (0.94-0.99) [0.98]; Pets 0.747 (0.67-0.81) [0.73].

| condition | EuroSAT mild | EuroSAT severe | Pets mild | Pets severe |
|---|---|---|---|---|
| Gaussian blur | 0.487 [0.91] | 0.340 [0.98] | 0.640 [0.71] | 0.453 [0.66] |
| Gaussian noise | 0.487 [0.92] | 0.360 [0.96] | 0.733 [0.73] | 0.547 [0.66] |
| JPEG | 0.620 [0.84] | 0.480 [0.85] | 0.713 [0.72] | 0.600 [0.69] |
| Brightness | 0.960 [0.96] | 0.720 [0.90] | 0.727 [0.72] | 0.727 [0.73] |
| Contrast | 0.913 [0.95] | 0.613 [0.90] | 0.760 [0.72] | 0.667 [0.71] |
| Centre crop | 0.880 [0.95] | 0.673 [0.89] | 0.740 [0.73] | 0.680 [0.71] |
| Down/up-scale | 0.447 [0.90] | 0.347 [0.97] | 0.680 [0.71] | 0.453 [0.66] |
| Occlusion | 0.820 [0.91] | 0.487 [0.88] | 0.747 [0.71] | 0.560 [0.64] |

Geometry/colour (EuroSAT / Pets): rot90 0.980 / 0.660; rot180 0.980 / 0.640; hflip 0.973 / 0.767; vflip 0.987 / 0.673; grayscale 0.907 / 0.760.

**Findings.**
- Robust: brightness 0.6, contrast 0.5, mild crop, hflip, and (EuroSAT) all rotations/flips are within or near CI of clean; EuroSAT is rotation-invariant by nature, so that is expected. Grayscale costs EuroSAT 7 points (0.91, CI just excludes clean) but not Pets.
- Fragile: anything that destroys high-frequency detail. Even mild blur (r=2), noise (sigma=20) and 4x downscale cut EuroSAT from 0.98 to 0.45-0.49 (still ~4-5x chance), and severe versions reach 0.34-0.36. JPEG q=30 gives 0.62. On Pets, mild versions of the same are mostly within CI of clean (0.64-0.73 vs 0.75), severe blur/downscale drop to 0.45 and occlusion 40% to 0.56. Pets is also hurt by rot90/rot180/vflip (about -8 to -11 points; CIs overlap clean but agreement with clean predictions is only 0.67-0.74), while hflip is harmless.
- Calibration degrades badly on EuroSAT: confidence stays 0.84-0.98 while accuracy falls to 0.34-0.49 (e.g. blur r=6: conf 0.98 / acc 0.34; wrong-answer confidence 0.97). Predictions also collapse toward one class (top-class share 0.37 -> 0.53). On Pets confidence drops somewhat with accuracy (0.73 -> 0.64-0.66), so it is closer to calibrated there, but conf(wrong) stays about 0.5.

**Caveats.** n=150 per cell gives CIs of roughly +/-7-10 points, so differences of <8 points (most Pets mild conditions, brightness on Pets) are within noise. Only two tasks, one checkpoint, no comparison against other checkpoints/baselines (no baseline for the corrupted condition; clean accuracy and chance are the references). The EuroSAT images are small (64x64) so blur/downscale/noise at fixed pixel strengths are far more destructive there than on Pets photos; severity is not comparable across tasks. Corruptions apply to the raw image before the model's own preprocessing. Run took ~35 min on a shared GPU (over the 25 min target) after an earlier run was killed by a reboot.

**Take-aways.**
1. The model tolerates photometric changes (brightness/contrast/grayscale-ish) and mild cropping, but is highly sensitive to blur, noise, JPEG and low resolution, especially on small satellite tiles.
2. On corrupted EuroSAT it is confidently wrong (confidence >= 0.84 at accuracy 0.34-0.62): confidence cannot be used to detect corruption there.
3. Pets orientation matters (rot/vflip lower accuracy and agreement); mild corruptions on Pets are within noise of clean.
