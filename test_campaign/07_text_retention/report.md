# 07 Text retention: does the vision checkpoint keep stock Laya's text-only ability?

**Question.** Does Laya-Vision (final = `stage2_c/wise085_calibrated`) still do what stock Laya does on text-only inputs? Compared: stock Laya (pinned rev 55cf4c4), stage2_a/calibrated, stage2_b/calibrated, stage2_b/wise080_calibrated, stage2_c/wise085_calibrated.

**Method.** Held-out `data/{ag_news,boolq,typed_decisions}.test.jsonl`, text only, fixed seed 0, n=400 per task (1200 items, same items for every model). `run_preds.py` runs the batched agent forward and stores per-item answer probabilities (`preds/*.jsonl`); `analyze.py` computes accuracy (Wilson 95% CI), paired difference vs stock (bootstrap 95% CI, exact McNemar), top-1 agreement with stock, ECE(10), NLL, Brier. Each model was run one at a time on GPU (fp16 default of the loader). Chance: ag_news 0.25, boolq 0.5, typed_decisions ~0.31 (mix of choice/score/noul questions).
Important: ag_news and boolq are tasks Laya was never fine-tuned on here (pure retention). typed_decisions is the stage-2 *text-replay* distribution (25% of the training mix), so gains there reflect training on it, not retention.

**Results (accuracy [95% CI]; diff = model minus stock, 95% bootstrap CI)**

| model | ag_news | boolq | typed_decisions | ALL |
|---|---|---|---|---|
| stock | .907 | .757 | .362 | .676 |
| stage2_a | .922 [.892,.945] d=+.015 | .728 [.682,.769] d=-.030 | .752 [.708,.792] d=+.390 | .801 |
| stage2_b | .917 [.886,.941] d=+.010 | .733 [.687,.774] d=-.025 | .760 [.716,.799] d=+.398 | .803 |
| wise080 | .925 [.895,.947] d=+.018 | .752 [.708,.792] d=-.005 [-.040,.030] | .698 [.651,.740] d=+.335 | .792 |
| **wise085 (final)** | .920 [.889,.943] d=+.013 [-.005,.030] | .752 [.708,.792] d=-.005 [-.043,.033] | .713 [.666,.755] d=+.350 [.295,.407] | .795 |

Stock accuracy CIs: ag_news [.875,.932], boolq [.713,.797]. Full details in results.json (CIs, McNemar p, ECE, NLL, Brier).
Top-1 agreement with stock (final): ag_news .963, boolq .850, typed_decisions .445. Calibration of final vs stock: ECE .030 vs .028 (ag_news), .068 vs .064 (boolq), .116 vs .186 (typed). NLL: .239 vs .250, .531 vs .529, .696 vs 1.308.
Identity check: loading the final checkpoint with `laya.Agent` vs `laya_vision.load` gives max |prob diff| = 0.0 on all 1200 items (text path is bit-identical, so the "vision" wrapper adds nothing for text).
Plot: `accuracy.png`.

**Findings.**
1. On the two untrained held-out text tasks the final model is statistically indistinguishable from stock: ag_news +1.3 pt (CI includes 0, McNemar p=.27), boolq -0.5 pt (CI [-4.3,+3.3], p=.90). Calibration and NLL also match stock.
2. Text ability is not bit-for-bit preserved: boolq top-1 agreement with stock is only 85% (31 items stock-right/ft-wrong vs 29 the reverse), i.e. the weights moved but errors cancel. stage2_a / stage2_b lose a bit more on boolq (-3.0 / -2.5 pt) but within noise (p=.15, .30); WiSE interpolation (0.80, 0.85) recovers it to stock level, consistent with its purpose. With n=400 we can rule out a boolq loss larger than ~4 pt, not a 1-2 pt one.
3. typed_decisions improves massively (.36 -> .71, p<1e-27) because it is a training-distribution task; stock was near chance (the repo's own 2000-item stock result is 0.3605, ours 0.362 matches). Interpolation toward stock costs ~4-6 pt here (from .752/.760 for a/b to .698 (wise080) and .713 (wise085)), a trade-off of WiSE: retention on boolq vs in-distribution gain.

**Caveats.** n=400/task, so CIs are about +/-4 pt; single seed of sampling; ag_news/boolq are only two retention probes (no generation/open-ended text, only Laya's probability-readout tasks); sampled subset not the full test sets (stock on full test: ag_news .926, boolq .756, consistent with ours). The earlier `c_sweep` alpha-sweep was not run (optional; not needed for the question). Stock load warns that `choice:11+` temperature is invalid (clamped to 0.5), which affects only 11+-option choice items.

**Take-aways.**
1. Final checkpoint retains stock text ability on held-out untrained text tasks within noise (ag_news +1.3, boolq -0.5 pt).
2. Raw stage-2 checkpoints show a small non-significant boolq dip (-2.5 to -3 pt) that WiSE (0.80/0.85) removes.
3. The large typed_decisions gain is task adaptation, not retention; do not read it as text improvement in general.
