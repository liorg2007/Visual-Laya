# 05 Modality grounding: does Laya-Vision use the image?

**Checkpoint:** `runs/stage2_c/wise085_calibrated`. **Question:** is accuracy driven by the image, or by text/state priors?

## Method
Held-out tests (`data/*.test.jsonl`), 8 tasks, up to 3 questions per image, seed 0. Each image/question set was re-run under 8 conditions through `agent.predict`:
`real`; `black`, `white`, `noise` (uninformative image, same size); `patch4x4` (image cut into 4x4 tiles and shuffled, keeps colour/texture but destroys layout); `shuffle_within` (another image of the same task, a different true label); `swap_cross_task` (image from another task); `text_only` (no image, state = the text or "No image is available.").
Prediction = argmax of `answer_probs`. CIs: bootstrap over images (1000-2000 resamples). "Macro" = mean of the 8 per-task accuracies (bootstrap CI in the table). Baselines: chance and a majority-class oracle (most frequent label per question-type/option-count stratum within the evaluated sample).

**Completeness:** the earlier run was killed by a reboot (checkpoint held only 49 images, and `results_final.json` was stale, n=4/task). `run.py` was changed to resume from complete image groups and checkpoint after every image (atomic write). It was resumed twice and stopped at a time budget, so the sample is **partial**: 45 images/task (44 scienceqa, 176 vqav2) = **902 questions x 8 conditions = 7216 predictions**, out of 590 planned images. The order was interleaved across tasks, so the sample is balanced; only fully completed (all 8 conditions) images are used. Noise/patch randomness is seeded per image.

## Numbers (accuracy, n = questions)
| task (n q) | real | black | white | noise | patch4x4 | shuf-within | swap-task | text-only | majority |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 (90) | .956 | .311 | .311 | .256 | .500 | .244 | .278 | .300 | .356 |
| eurosat (90) | .922 | .344 | .344 | .367 | .833 | .322 | .356 | .356 | .389 |
| food101 (90) | .889 | .278 | .267 | .278 | .789 | .289 | .278 | .289 | .311 |
| oxford_pets (90) | .700 | .244 | .244 | .244 | .644 | .289 | .256 | .222 | .356 |
| coco (135) | .622 | .274 | .274 | .304 | .593 | .311 | .370 | .444 | .400 |
| aokvqa (48) | .521 | .208 | .229 | .208 | .521 | .292 | .354 | .396 | .333 |
| scienceqa (56) | .661 | .571 | .571 | .589 | .714 | .589 | .607 | .625 | .482 |
| vqav2 yes/no (303) | .611 | .492 | .465 | .502 | .568 | .465 | .459 | .475 | .525 |
| **macro (image-level)** | **.741** [.707,.773] | .339 [.305,.372] | .335 [.300,.369] | .344 [.309,.379] | .648 [.611,.683] | .345 [.311,.379] | .371 [.338,.405] | .382 [.346,.418] | ~.39 |
| drop vs real, 95% CI | 0 | -.40 [.36,.45] | -.41 [.36,.45] | -.40 [.35,.44] | -.09 [.06,.13] | -.40 [.35,.44] | -.37 [.32,.41] | -.36 [.31,.40] | |

Per-task bootstrap CIs are in `results.json` (e.g. cifar10 real .956; wide for aokvqa/scienceqa at n<=56).
VQAv2 yes/no (n=303; always-yes .525, always-no .475): real predicts yes 44% of the time (acc .55 on gold-yes, .68 on gold-no). With any non-informative image it collapses to "no": black/white/noise/mismatched predict yes only 7-12%, text-only 0%. Accuracy stays ~.46-.50, i.e. at the always-no level.

## Findings
- **The image is used, strongly, for classification/captioning-style tasks.** Replacing the image with black/white/noise or a different image drops macro accuracy by about 0.40 (CI excludes 0), to roughly the majority level. On cifar10/eurosat/food101/oxford_pets the "choice" questions go from .53-.93 to .02-.13, i.e. below chance when the image is wrong: the model follows the image, not a text prior. A wrong-but-real image yields the wrong label (shuffle_within ~ black ~ text_only), so it is image content, not merely "any image present".
- **Not a layout-free texture/colour classifier alone:** patch shuffling costs little on eurosat/food101/coco/aokvqa (0.00 to -0.10) but a lot on cifar10 (.96 to .50) and little on scienceqa. So much of the signal is global colour/texture statistics that survive tile shuffling; spatial structure matters mainly for low-resolution objects.
- **Weakly grounded tasks:** scienceqa (.66 real vs .57-.61 without the image, ~ text-prior/majority, 0.48) and vqav2 yes/no (.61 vs .47-.50) show a small/uncertain image effect. The vqav2 real-vs-blank gap (~+.11 to +.15) is borderline: real CI [.56,.67] vs text-only [.43,.53] and black [.44,.54] barely overlap at n=303. The aokvqa/coco text_only values are above chance (.40/.44) because the question options carry priors; text-only coco .44 ~ majority .40, so no evidence of text-only skill beyond priors.
- **Calibration issue (not accuracy):** mean answer confidence barely drops with no usable image (macro conf: real .76, black .71, noise .68, shuffle .74, text-only .59). The model is over-confident when the image is uninformative (e.g. cifar10 black: wrong answers with high confidence), so confidence is a poor "image missing" signal. On blank images the model still outputs a fixed default (e.g. cifar10 -> the last class; vqav2 -> "no").

## Caveats
- Partial sample (45 images/task; scienceqa/aokvqa n=56/48 questions, wide CIs); the planned 50-240 images were not all completed within the time budget.
- Majority baseline is an oracle computed from the evaluated sample; with few questions per stratum it is noisy and slightly optimistic.
- Questions of the same image share a bootstrap unit; the 4x4 patch shuffle is one random permutation per image. Non-image states (`text_only`) use a different state format ("No image is available." when no text), so it also tests state-format shift.
- Contended GPU; no effect on results, only on speed.

## Take-aways
1. Yes, the model uses the image: removing/mismatching it costs ~0.40 macro accuracy (CI .35-.45) and drives the classification-style tasks to or below chance/majority.
2. Grounding is weak on scienceqa and vqav2 yes/no, where image-free accuracy is near the majority baseline and the real-image gain is small (+.05 to +.15).
3. Confidence does not fall when the image is uninformative, so the output probabilities cannot flag missing/irrelevant images; and part of the visual signal is colour/texture, which survives patch shuffling.

Files: `run.py` (resumable), `analyze.py`, `raw_final.json`, `results.json` (= `results_final.json`), `grounding.png`, logs `run.log`, `run2.log`, `run3.log`.
