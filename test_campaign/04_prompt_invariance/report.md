# 04 Prompt invariance — Laya-Vision `stage2_c/wise085_calibrated` (after the all-caps fix)

**Question.** Does the prediction change when only the prompt's surface form changes (option order, choice labels, paraphrase, casing, punctuation, option length, number of options), with the image and correct option held fixed? Does the all-caps fix (`laya_vision/textnorm.py` + `agent.py`) remove the `case_upper` collapse without changing anything else?

**Method.** 50 images per task × 6 tasks (ScienceQA, A-OKVQA, CIFAR-10, Food-101, Oxford Pets, EuroSAT) from `data/*.test.jsonl`, one choice question per image, n=300. 20–21 variants per image in one `predict()` call. Fixed seeds (`Random(1)` subset, `Random(0)` draws); clean single-process GPU run (78 s, 0 errors). Metrics: accuracy, paired Δ vs base, flip rate (prediction ≠ base prediction). 95% CIs: bootstrap over records. "Before/after" pairs each image with the pre-fix run (`raw_main_prefix.json`). Scripts: `run.py`, `analyze.py`, `compare.py`, `label_check.py`, `drift_check.py`, `residual_check.py`. Pre-fix outputs are kept as `*_prefix.*`.

## Post-fix results (pooled, n=300 unless noted; mean chance 1/k = 0.15)

| variant | acc [95% CI] | Δ vs base | flip |
|---|---|---|---|
| **base** | **0.760 [0.710, 0.807]** | — | — |
| option order perm0–3 | 0.733–0.763 | −0.027…+0.003 (all CIs ∋ 0) | 0.09–0.11 |
| relabel upper / number / "Option X" | 0.760 / 0.730 / 0.730 | 0.000 / −0.030 [−.060, .000] / −0.030 [−.063, .003] | 0.04 / 0.08 / 0.11 |
| relabel answer-text-as-key (letter tasks, n=100) | 0.610 | +0.060 [.000, .130] | 0.16 |
| paraphrase in-dist / 3 OOD | 0.763 / 0.750–0.760 | +0.003 / −0.010…0.000 | 0.04 / 0.01–0.03 |
| case_lower / **case_upper** | 0.740 / **0.750 [.700, .797]** | −0.020 [−.043, .003] / **−0.010 [−.033, .017]** | 0.057 / 0.077 |
| punct_strip | 0.757 | −0.003 [−.010, .000] | 0.007 |
| opt_long (verbose option text) | 0.687 | **−0.073 [−.117, −.030]** | 0.19 |
| add_dist (letter tasks, k 3.6→7.6, n=100) | 0.380 | **−0.170 [−.250, −.090]** | 0.36 |
| drop_half (k 11.2→6.0, n=292) | 0.767 | +0.003 [−.027, .034] | 0.10 |
| k = 2 / 5 / 10 / 20 (chance .50/.20/.10/.05) | 0.883 / 0.770 / 0.905 (n=200) / 0.780 (n=100) | | |

Order consistency: all 5 orders agree on 0.813 [0.767, 0.857] of images (pairwise flip 0.098); EuroSAT .96, CIFAR .94, Food .90, Pets .72, ScienceQA .68, A-OKVQA .68. Accuracy by gold position (k≥3): first .70, middle .78, last .63 — confounded by task mix.

## Fix verification: all-caps (pre vs post, paired by image, n=50/task)

| task | base | upper pre | upper post | post − pre | post: upper − base |
|---|---|---|---|---|---|
| ScienceQA | 0.60 | 0.52 | 0.48 | −0.04 [−.20, .12] | −0.12 [−.24, −.02] |
| A-OKVQA | 0.50 | 0.42 | 0.56 | +0.14 [−.04, .30] | +0.06 [.00, .14] |
| CIFAR-10 | 0.92 | 0.64 | 0.92 | +0.28 [.14, .42] | 0.00 |
| Food-101 | 0.82 | 0.06 | 0.82 | **+0.76 [.64, .86]** | 0.00 |
| Oxford Pets | 0.76 | 0.22 | 0.80 | +0.58 [.44, .72] | +0.04 [.00, .10] |
| EuroSAT | 0.96 | 0.66 | 0.92 | +0.26 [.14, .40] | −0.04 [−.10, .00] |
| **Pooled** | 0.760 | 0.420 [.363, .477] | 0.750 [.700, .797] | **+0.330 [.267, .393]** | −0.010 [−.033, .017] |

- **case_upper is back at base.** The remaining ScienceQA gap (−0.12) is the lower-casing itself: post-fix case_upper matches case_lower on 50/50 ScienceQA items, and case_lower is −0.12 in both runs. Pooled, case_upper and case_lower agree on 294/300.
- **No other variant changed because of the fix** (none contains an all-caps string, so inputs are byte-identical). The 12 deterministic variants changed 0 predictions except one near-tie (ScienceQA, conf .400 vs .399) traced to non-determinism in the pre-fix run's resumed half; re-running 5 such batches with the fix monkeypatched off reproduced the post-fix values in 4. One record's confidence (+0.014, same answer) is attributable to the fix, most likely via batch padding. Random-draw variants changed 0 predictions on the first 72 records where draws are identical (the pre-fix run re-seeded after a reboot).
- **Labels preserved:** all 12 spot-checked case_upper items return the caller's upper-case keys ("SHIP", "PERMANENT CROP", "MUSSELS", "A"–"D"); all 300 map back through the caller's key dict.

## Caveats
- n=50/task → per-task CIs ±0.10–0.15; one checkpoint; choice questions only.
- **Short all-caps strings are not rewritten** by design (needs a word of ≥4 letters): "DOG"/"CAT" (all 50 CIFAR records), "PUG" (29 Pets), "PHO"/"HOT DOG" (16 Food), "TO EAT". No loss visible at this n but untested at scale. Mixed-case shouting ("Which is BIGGER?") and the text-only path are unchanged.
- k=10/20 points include only classification tasks, so the k curve mixes tasks.

## Take-aways
1. **The fix works:** all-caps prompts 0.420 → 0.750 pooled vs base 0.760 (Food-101 0.06 → 0.82, Pets 0.22 → 0.80), and the caller still gets their own labels back.
2. **The fix is surgical:** 0 changed predictions wherever inputs and draws are identical; one 0.014 confidence change from batch padding.
3. **Remaining prompt sensitivity is option content, not wording:** order, relabelling, paraphrase and punctuation stay within ~0.03 (orders still flip ~10% of answers, ~17% on ScienceQA/A-OKVQA/Pets); verbose options (−0.07) and extra distractors (−0.17) are the real costs, plus a ScienceQA lower-casing effect (−0.12).
