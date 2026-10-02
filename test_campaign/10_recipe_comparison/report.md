# 10 Recipe comparison: accuracy vs text retention

**Question.** What did each training change buy (guarded projector, grounding metric, recipe B, WiSE-FT, continuation), and what does it cost in text accuracy?

**Method.** The stored predictions in `eval/results/*.predictions.jsonl` cover the identical 16,707 held-out rows for stage2_a, stage2_b, wise080 (b), stage2_c wise085, caption->Laya and stock Laya. I paired them by row id on the **full test sets** (n: EuroSAT 2668, Pets 694, VQAv2 y/n 3075; AG News 5000, BoolQ 3270, typed_decisions 2000). This is a stronger paired comparison than a subsample, so I used it. Accuracy is the argmax match used by `metrics.row_stats`. CIs and differences come from a paired cluster bootstrap (2000 resamples, seed 0, clusters = image/state, stratified by task, macro over tasks). Stage-1 checkpoints and SigLIP zero-shot exist only on the COCO caption-matching test (choice rows, n=750, paired). Training curves come from `runs/*/metrics.jsonl` (`train_curves.json`). Script: `analyze.py`. Plot: `tradeoff.png`.

**A fresh re-run on a common subsample FAILED**: with 11 other agents on the shared GPU/CPU (load avg ~17), one `run_eval` produced no progress line in 25 min and I killed it. Nothing was re-run, and every number below is from existing results. The pairing is still exact because all models share the same row ids.

## Numbers (accuracy, 95% CI)

| model | EuroSAT | Pets | VQAv2 y/n | **image macro** | AG News | BoolQ | **text macro (AG+BoolQ)** | typed_decisions |
|---|---|---|---|---|---|---|---|---|
| chance / baseline: caption->Laya | .389 | .280 | .592 | .420 [.406,.434] | | | | .361 |
| stock Laya (text only) | | | | | .926 | .756 | .841 [.833,.849] | .360 |
| stage2_a (recipe A) | .929 | .403 | .527 | .620 [.609,.631] | .925 | .719 | .822 [.813,.831] | .745 |
| stage2_b | .970 | .810 | .592 | **.791** [.779,.803] | .924 | .726 | .825 [.816,.834] | .744 |
| stage2_b + WiSE .80 | .967 | .725 | .591 | .761 [.747,.774] | .926 | .750 | .838 [.830,.847] | .705 |
| stage2_c + WiSE .85 (final) | .970 | .765 | .608 | .781 [.769,.793] | .924 | .747 | .836 [.827,.844] | .715 |

Stage 1 on COCO caption matching (choice rows, n=750): SigLIP zero-shot **.823** [.793,.853], stage1_b **.731** [.695,.764], stage1_a **.385** [.352,.420]. Differences: SigLIP − stage1_b = +.092 [.056,.131]; stage1_b − stage1_a = +.345 [.297,.391].

Key paired differences (macro, 95% CI):
- image, b − a: +.171 [.155,.187]
- image, c85 − caption: +.361 [.344,.379]
- image, b − c85: +.009 [−.001,.021], so within noise
- image, b − wise080: +.030 [.018,.042]
- text, stock − a: +.019 [.012,.027]
- text, stock − b: +.016 [.008,.023]
- text, stock − wise080: +.003 [−.003,.009]
- text, stock − c85: +.006 [−.001,.012]
- text, c85 − b: +.011 [.006,.015]
- typed_decisions, c85 − b: −.029 [−.046,−.012]

## Findings

1. **Guarded projector (recipe B stage 1)** is the biggest single change. Recipe A's stage 1 collapsed (COCO .385, .44 below SigLIP; image-token cosine .9995). Stage 1b reaches .731 with grounding (accuracy minus accuracy with the image swapped) rising 0.005 to 0.337 and token cosine falling to 0.05. It is still .09 below SigLIP zero-shot.
2. **The grounding metric** did not change accuracy itself. It diagnosed that A's .446 "image accuracy" was text priors. Recipe A has no grounding value in its logs, and its Pets .403 and VQAv2 .527 (below the caption baseline .592) show it was still ungrounded after stage 2.
3. **Recipe B stage 2** (LoRA, easier negatives): image macro .620 -> .791, mostly Pets (.403 -> .810) and EuroSAT (.929 -> .970). VQAv2 y/n is only .592, equal to caption->Laya (CI [.575,.611] vs [.573,.610]) and far from a strong VQA result. The cost is BoolQ −3 points (.726 vs stock .756; stock − b = +.016 on the text macro, CI excludes 0).
4. **WiSE-FT** trades image for text almost linearly. Interpolating b at .80 recovers BoolQ (.750, text macro within noise of stock) but costs Pets −.085 and image macro −.030 (CI excludes 0). It also lowers typed_decisions (.744 -> .705): the decision skill is partly lost with the interpolation.
5. **VQAv2-train continuation + WiSE .85** moves along the same frontier at a better point: VQAv2 .608 (+.016 over b, CI of difference not computed per task), Pets .765, image macro .781 (−.009 vs b, within noise), text macro .836 (+.011 vs b, only .006 below stock, CI [−.001,.012], i.e. not distinguishable from stock). Of the three models near the frontier, c85 is the only one whose text gap to stock includes 0 while image macro stays within ~1 point of the best (b). The training-side numbers match: c adds only +.005 image accuracy in its own eval (.776 -> .782).
6. The alpha sweep on the held-out calibration splits (`runs/pipeline_c/sweep.log`, not test) shows Pets falling .851 -> .805 from alpha 1.0 to .85, while the BoolQ proxy rises .784 -> .812: the Pets-vs-BoolQ trade is the main price of WiSE.

## Caveats
- No fresh subsample re-run (see above), so the "consistent paired comparison" rests on the stored predictions. The stage-1 checkpoints were not scored on the stage-2 tasks, so they are compared only on COCO, and SigLIP zero-shot only there as well (it cannot answer yes/no rows).
- Cluster bootstrap treats clusters as independent; Pets has only 694 rows, so Pets differences carry wide CIs (about ±.03).
- Typed_decisions is a trained-on task family, not a retention test: it rises from stock .36 for all fine-tuned models. I used AG News and BoolQ for retention. The existing evals use fp16 and the calibrated checkpoints; calibration temperatures were refit separately for each checkpoint, so accuracy is unaffected but ECE is not compared here.
- Recipe A used full fine-tuning and B/C LoRA, so "recipe B bought X" bundles several changes (guarded projector, LR, LoRA, easier negatives, text rows). I cannot separate them without ablations.

## Take-aways
1. The guarded projector with LoRA encoder (recipe B) caused essentially all of the image gain (+.17 macro over A); the grounding metric is what exposed A as text-prior-only.
2. Text retention is bought with WiSE-FT at a price of 1-3 image points, mostly Pets: final c85 has text within noise of stock (−.006) and image macro .781 (b: .791, difference within noise).
3. VQAv2 y/n is still at the caption->Laya level (.608 vs .592); continuation moved it by about +.016 only, so VQA-style reasoning is the unsolved part.
