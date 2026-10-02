# 12 - Question types and mixed states (Laya-Vision, stage2_c/wise085_calibrated)

**Question.** How does the model behave across the question types the API exposes (`choice`, `noul` yes/no probability, `score` ordinal) and across mixed input states (image+text, text-only, contradicting text, multi-question calls)? (Free-form / numeric question types do not exist in the API and were not tested.)

**Method.** `run.py` (resumable, per-section checkpoint to `results_final.json`; A was finished before the reboot, B-E after). Held-out `data/*.test.jsonl`, n=50 items per task (seeded), fp16 autocast, Wilson 95% CIs. Final numbers: `results.json`. Small n: CIs are wide (about +/-13 pts).

## Numbers
| Probe | Result (95% CI) | Baseline |
|---|---|---|
| A choice, oxford_pets (20-way) | 0.74 [.60,.84] | chance .05 |
| A choice, coco | 0.54 [.40,.67] | .25 |
| A choice, aokvqa | 0.40 [.28,.54] | .25 |
| A choice, scienceqa | 0.60 [.46,.72] | .25 |
| B noul matched vs counterfactual label, AUC: pets / food101 / cifar10 | 0.954 / 0.986 / 0.998 (acc@.5 .88/.87/.96, n=100) | AUC .5 |
| B noul coco caption true vs swapped | AUC 0.845, acc .65 [.55,.74] | .5 |
| B noul VQAv2 yes/no (n=100) | AUC 0.658, acc .64 [.54,.73] | majority .60 (within noise) |
| C score, synthetic blur (12 imgs x 4 levels) | Spearman +0.58; mean score 1.26,1.24,1.30,1.59 | expect monotone up |
| C score darkness / low contrast / desaturation | rho 0.02 / 0.07 / 0.06 (flat) | |
| C score noise / dot count | rho -0.33 / -0.31 (wrong sign) | |
| C text-only sentiment score (4 texts) | 0.04, 0.91, 2.00, 2.58 (monotone) | |
| D pets: image only / image+true text | 0.80 [.67,.89] / 1.00 [.93,1.0] | |
| D pets: image+WRONG text picks image label / follows text | 0.42 [.29,.56] / 0.48 [.35,.62] | |
| D pets: text only (true / wrong label) | 1.00 / 1.00 (just reads the label) | |
| D pets: gray image + true text | 1.00 | |
| D scienceqa (rows with text): image+text / image only / text only | 0.48 / 0.56 / 0.58 (all within noise, n=50) | .25 |
| D OCR 8-way rendered word (n=24): image only | 0.33 [.18,.53] | chance .125 |
| D OCR: +agreeing text / +contradicting text follows text | 0.92 / 0.79 (follows image: 0.125 = chance) | |
| E 5-question call vs single calls, max abs prob diff | mean .009, max .034 | |
| E reversed question order, max abs diff | 0.0 (exact) | |
| E choice acc inside 5-q call (n=15) | 0.73 [.48,.89] | |

## Findings
- `choice` and `noul` on natural images work well above chance. Matched-vs-counterfactual `noul` discriminates strongly for class labels (AUC .95-.998) and moderately for captions (.85), but is near-useless on VQAv2 yes/no (.64 vs .60 majority).
- `score` on synthetic image degradations is largely not tracking the property: only blur is mildly ordinal; darkness/contrast/saturation are flat, noise and dot count have the wrong sign. The text-path score is clean and monotone, so ordinal behaviour is a text-trained skill that does not transfer to image probes.
- Text dominates image in mixed states: with a contradicting caption the model follows the text about as often as it says the image label (48% vs 42%), and on OCR it follows the text 79% vs image 12.5%; a gray image plus true text is 100%. Text-only and image+text are not worse than image-only for pets, but for ScienceQA added text did not help (within noise).
- Multi-question calls are consistent: order-invariant exactly; batched vs single-call probabilities differ by <=0.034 (fp16 noise level).

## Caveats
n=50 (24 for OCR, 12 images for C, 15 for E) so most differences between conditions are within noise; synthetic score probes use my own criteria wordings (prompt choice may matter); the A section used a different sampling stream than B-E (resume after reboot); GPU shared with other agents.

## Take-aways
1. Choice/noul are reliable for class-like content; yes/no VQA and image-quality `score` probes are not.
2. Text overrides vision when both are given, even when contradicting the image (a grounding risk).
3. Batching multiple questions per call is safe (order-invariant, near-identical to single calls).
