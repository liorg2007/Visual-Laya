# 02 Calibration: Laya-Vision `runs/stage2_c/wise085_calibrated`

**Question.** Are the confidences (max p) trustworthy, do they rank errors, and does calibration degrade on tasks away from the monitored in-distribution set?

**Method.** (a) Existing full-test predictions `eval/results/stage2_c_wise085.predictions.jsonl` (oxford_pets, vqav2_yesno, eurosat = image; ag_news, boolq, typed_decisions = text; 16.7k rows; probs rounded to 4 dp). (b) New run of the same checkpoint (fp16 autocast, seed 0, 150 random image states per task) on cifar10, food101, aokvqa, scienceqa, coco (n rows below). Confidence = max p; correct = argmax equals gold. Metrics: ECE 10/15 equal-width, adaptive (equal-mass) ECE 10/15, NLL, Brier, AUROC of confidence and of entropy confidence (1 - H/log K) vs correctness, selective accuracy at coverage 100..20%, 95% CIs (Wilson for accuracy; row-level bootstrap, 300 resamples, for ECE/AUROC). Temperature: post-hoc power rescaling p^(1/tau) (equivalent to logit temperature), tau fit by NLL. Scripts: `run_new.py`, `analyze.py`; data: `results.json`, `pred_cal.jsonl`; plots: `reliability_by_task.png`, `selective.png`.

**Scope note.** cifar10/food101/aokvqa/scienceqa are *training and calibration-fit tasks* (their test splits are held out, but the task distribution was seen), so they are not truly OOD. Only **coco** (never in train/calib files) is genuinely unseen; boolq/ag_news are text tasks not trained in stage 2. I report all and label them honestly.

## Numbers (ECE10 / adaptive ECE10 / AUROC-conf / acc, with 95% CI where useful)
| task | n | acc [CI] | chance | ECE10 | ECE15 | aECE10 | AUROC conf | AUROC entropy | sel. acc @80/50/20% |
|---|---|---|---|---|---|---|---|---|---|
| oxford_pets | 694 | .765 [.73,.80] | .27 | .023 | .035 | .043 | .81 | .71 | .85/.93/.99 |
| vqav2_yesno | 3075 | .608 [.59,.63] | .50 | .022 | .023 | .037 | .59 | .59 | .63/.68/.72 |
| eurosat | 2668 | .970 [.96,.98] | .30 | .009 | .012 | .010 | .94 | .92 | 1.0/1.0/1.0 |
| cifar10 | 300 | .980 [.96,.99] | .30 | .014 | .019 | .017 | .86 | .82 | 1.0/.99/1.0 |
| food101 | 300 | .927 [.89,.95] | .28 | .026 | .029 | .029 | .89 | .82 | .98/1.0/1.0 |
| aokvqa | 154 | .442 [.37,.52] | .25 | .031 | .078 | .069 | .63 | .62 | .46/.53/.71 |
| scienceqa | 171 | .620 [.55,.69] | .37 | .068 | .085 | .089 | .78 | .78 | .69/.81/.97 |
| **coco (unseen)** | 600 | .680 [.64,.72] | .34 | **.052** | .046 | .050 | **.74** | **.64** | .75/.82/.94 |
| ag_news (text) | 5000 | .924 | .25 | .029 | .030 | .029 | .86 | .85 | .97/.99/1.0 |
| boolq (text) | 3270 | .747 | .50 | .071 | .072 | .071 | .70 | .70 | .80/.85/.92 |
| typed_decisions (text) | 2000 | .715 | .32 | **.118** | .118 | .118 | .75 | .68 | .78/.87/.97 |

Groups: old image tasks (n=6437) ECE10 .014 [.010,.025], AUROC .815; cifar10+food101+aokvqa (n=754) ECE10 .019, AUROC .927; coco ECE10 .052 [.04,.09], AUROC .738; text overall ECE10 .042 [.036,.049]. Note bootstrap ECE CIs can sit above the point estimate (ECE is upward-biased at small n), and rows from one image are correlated, so CIs are optimistic.

**Per option count (image rows, all tasks pooled; mixes tasks so K is confounded with task):** K=2 ECE10 .017 (n=5417, AUROC .77); K=10 .010 (n=1484, acc .96); K=20 .026 (n=497); K=4 .053 (n=259); K=3/5/6/8: .137/.137/.142/.163 (n=54-71 each, CIs ~±.1, mostly coco and scienceqa; overconfident in 5, 6, 8: conf > acc by .10-.12). Text: K=2 .051, K=4 .041, K=5 .138 (n=300, typed_decisions, *under*confident: conf .58 vs acc .71).

**Temperature.** The checkpoint already carries fitted temperatures (image ~1.25-1.36, by option bucket). Re-fitting a single extra tau on the old image tasks gives tau=0.95 (i.e. essentially no change), and applying it elsewhere changes ECE by <=.01 except aokvqa (.031 to .077, worse). Per-task oracle tau (fit on the test rows themselves, so optimistic) shows where the miscalibration is systematic: typed_decisions tau=.52 (ECE .118 to .014, underconfident), boolq tau=1.48 (.071 to .009, overconfident), ag_news 1.23, coco 1.16 (.052 to .038), aokvqa 1.22 (.031 to .040, noise-level). Image tasks that matter (eurosat, cifar, food, pets, vqav2) are already at tau~0.9-1.15. I could not run the *un*calibrated checkpoint (`wise085`) on the new tasks: the shared GPU made the 750-state run take 26 min, so the pre-calibration comparison is not available; the tau analysis above is the substitute.

## Findings
1. Image confidence is well calibrated on in-distribution tasks (ECE10 .01-.03, passes the .10 acceptance bar easily); text is worse (ECE .042 overall), driven by typed_decisions (.118, underconfident) and boolq (.071, overconfident). The text head was refit by calibration but not on these test distributions.
2. Shift: unseen coco has ECE .052 (about 3-4x the in-distribution image .014) and the ranking signal drops: AUROC .74 vs .82-.93; entropy confidence is clearly worse than max-p there (.64 vs .74). Calibration degrades gracefully, not catastrophically. The seen-task test splits (cifar/food) show no degradation; scienceqa (.068) and aokvqa (small n, ECE CI [.04,.13]) are the weak spots.
3. Confidence is a useful selective-prediction signal: acc at 50% coverage rises to .96 for in-dist image (from .775), .82 on coco (from .68), but it is a weak signal where the model is near chance: vqav2_yesno AUROC .59, acc .61 to .68 at 50% coverage, boolq AUROC .70. Entropy confidence is never better than max-p (equal for binary).
4. Calibration is poor for mid-size option counts (K=3-8, ECE .13-.16, overconfident), but those cells are small (n 54-71) and task-confounded.

## Caveats
Subsampled new tasks (150 states, ECE noisy for n<=300); fp16 autocast whereas the checkpoint config says bf16 (may differ slightly from the existing bf16/other-run predictions); old predictions rounded to 4 dp; tau re-fit via probability power scaling, not original logits; no uncalibrated-checkpoint comparison.

## Take-aways
- In-distribution image calibration is good (ECE10 ~.014); do not touch it.
- Text calibration is the weaker part (typed_decisions .118 under, boolq .071 over); a per-task or per-type text temperature refit would likely fix it (oracle ECE <.015).
- Under real shift (coco) ECE roughly triples to .05 and error ranking drops (AUROC .74); use max-p, not entropy, for abstention, and expect only modest gains at near-chance tasks.
