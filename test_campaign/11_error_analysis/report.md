# 11_error_analysis: failure modes of stage2_c/wise085_calibrated

**Question.** Where does the final checkpoint fail? Source: `eval/results/stage2_c_wise085.predictions.jsonl` joined to `data/*.test.jsonl` (no fresh GPU runs needed). Script: `analyze.py`; all numbers in `results.json`. Accuracies are on the **choice** question for Oxford Pets / EuroSAT (the headline table mixes in the easier binary "verify class" question), Wilson 95% CIs.

## Numbers
| slice | n | acc [95% CI] | reference |
|---|---|---|---|
| Oxford Pets 20-way choice | 347 | 0.651 [0.600, 0.700] | chance 0.05 |
| Oxford Pets binary verify | 347 | 0.879 [0.840, 0.909] | 0.5 |
| EuroSAT 10-way choice | 1334 | 0.960 [0.948, 0.969] | chance 0.10 |
| EuroSAT binary verify | 1334 | 0.980 [0.971, 0.986] | 0.5 |
| VQAv2 yes/no | 3075 | 0.609 [0.591, 0.626] | majority "no" 0.533 |

**Oxford Pets confusions** (n small, 1-5 each): samoyed -> great pyrenees (5), british shorthair -> russian blue (5), persian -> ragdoll (4), pomeranian -> havanese (3), leonberger -> japanese chin / great pyrenees (3+3), english setter -> english cocker (3), basset -> beagle (3), american bulldog -> boxer (3), staffordshire -> pit bull (2). Worst classes: american bulldog 1/9, leonberger 2/14, english cocker 1/5, miniature pinscher 2/8, persian 3/9, saint bernard 3/9. Errors are visually plausible look-alikes (white fluffy dogs, long-haired white cats, bull-type breeds), i.e. fine-grained breed discrimination, not random noise. Confusion matrix: `confusion_oxford_pets.png`.

**EuroSAT confusions:** Permanent Crop -> Annual Crop (8) / Herbaceous Veg (5) / Pasture (3); River <-> Highway (4+2); Highway -> Annual Crop (4); SeaLake -> River (3). Per-class: Permanent Crop 0.857 (n=126), River 0.933, Highway 0.938; all others >= 0.96. These are the known EuroSAT ambiguities (linear features, vegetation types). `confusion_eurosat.png`.

**VQAv2 yes/no.** Model is biased to "no": predicts yes 38.9% vs 46.7% true; yes-recall 0.498 vs no-recall 0.706. By first words (CIs wide, most n<300): "is this" 0.653 (n=608), "is the" 0.605 (691), "is there" 0.569 (288), "does the" 0.519 (108), "are there" 0.614, "can/can you" ~0.5 (n~25). Only "is this" vs "is there" is near-significant; the rest overlap. Keyword slices: colour 0.46 (n=26), counting-ish words (any/many/two...) 0.542 (131), people/animals 0.625, sky/weather 0.677. Note "is there" questions are where yes-prediction rate is lowest (0.36 vs 0.51 true).

**Confidence.** Calibrated, so high-confidence errors are rare but not absent:
- EuroSAT: 8 errors at conf >= 0.95 (of 1160, 0.7%); 13 at >= 0.9. Top wrong cases (highway->annual crop 1.00, herbaceous->pasture 1.00) are probably borderline/label-noise tiles (see sheet).
- Oxford Pets: 0 errors at conf >= 0.95 (n=32); 4/58 wrong at >= 0.9 (all look-alike breeds, e.g. samoyed->great pyrenees at 0.94, persian->ragdoll 0.94).
- VQAv2: confidence is narrow (>=0.8 for only 192 items, 39 wrong = 20%); 0 wrong at >=0.95 but n=3. Many "wrong" high-confidence VQA items are subjective/ambiguous ("Is this a desert scene?", "Does he look angry?").
- Low-confidence correct: Oxford conf<0.5: 52/126 correct (41%, matches stated confidence); EuroSAT conf<0.6: 12/26 (46%); VQA conf<0.6: 911/1691 (54%, near chance, so those items are effectively coin flips).
- Reliability tables are in `results.json` (confidence.*). Oxford is slightly under-confident at 0.3-0.6; EuroSAT 0.7-0.9 slightly over-confident (n small).

**Option count / position.** Option count is fixed per task (20, 10, 4, 2) so only a cross-task comparison exists; accuracy falls with option count within the same data type (EuroSAT 10-way 0.96 vs Pets 20-way 0.65), but that is confounded with task difficulty. Position (choices are shuffled): EuroSAT flat (0.94-0.98 by position, predicted-position histogram near uniform). Oxford by true-position quartile Q1-Q4 = 0.71/0.64/0.72/0.55 (Q4 vs Q3 CIs overlap; per-position n is 7-26, so mostly noise; weak hint of a last-quartile dip). ag_news Q3 0.867 vs Q2 0.980 and typed_decisions Q1 0.57 vs Q2/Q3 0.74 differ more, but those are confounded with class identity (position correlates with which class), so I cannot attribute them to position bias. No evidence of a strong systematic position bias in the vision tasks.

**Image properties.** Oxford (122 distinct sizes) and VQA (257): no relation of correctness to area, brightness, contrast, or aspect ratio (all |Spearman| < 0.07, p > 0.2). EuroSAT is all 64x64; brightness shows a small negative trend (brightest quartile 0.931 vs 0.976 darkest, rho = -0.076, p = 0.006; likely bright arid/bare tiles resembling other classes; not corrected for multiple tests, so treat as weak).

**Worst errors** (`worst_errors_contact_sheet.png`: 8 most-confident wrong each for Pets and EuroSAT, 8 for VQA). Pets: look-alike breeds, some arguably hard (samoyed vs great pyrenees, persian vs ragdoll). EuroSAT: tiles that look like the predicted class to a human (plowed fields labelled Highway/Permanent Crop; a dark water/river tile). VQA: subjective questions (angry, injured, desert climate) and one count/visibility question.

## Caveats
Single checkpoint; test sets small for Pets (347 images, 37 breeds listed 20 per question, per-class n 5-14 so confusion pairs are anecdotal); VQA question-type slices small; VQA labels are soft consensus and argmax of target used as truth; the position/brightness p-values are uncorrected. Contact-sheet picks are selected by confidence, not random, so they overstate how "wrong" the model looks.

## Take-aways
1. Weakness is fine-grained discrimination (Pets look-alike breeds 65%; EuroSAT Permanent Crop / River / Highway) rather than size/brightness/position artifacts, none of which matter materially.
2. VQAv2 yes/no (60.9%, only +7.6 pts over majority) is the weakest area, with a "no" bias (yes-recall 0.50) and near-coin-flip behavior at conf < 0.6; subjective questions make much of the error irreducible.
3. Calibration makes confidence trustworthy on vision tasks: confident errors are rare (0 at >=0.95 for Pets, 0.7% EuroSAT) and mostly look-alike/ambiguous labels, so confidence thresholding for abstention is effective.
