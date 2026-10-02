# 06 Generalization: OOD options, formats, vocabulary, questions

**Question.** How well does Laya-Vision (`runs/stage2_c/wise085_calibrated`) generalize beyond the exact task formats it was trained on?

**Method.** For 7 held-out test sets (cifar10, food101, oxford_pets, aokvqa, scienceqa, eurosat, coco), 80 images each (fixed seed 0, choice questions only). Each image is re-asked in perturbed forms: (a) unseen option counts (2/3/6/8/10/12/30), (b) novel key formats (letters A-J, digits, w/x/y/z), (c) novel vocabulary (synonym labels, "a photo of a X", "homemade X", synonym EuroSAT descriptions), (d) novel questions never in training (animal-vs-vehicle, living-vs-machine, cat-vs-dog derived from labels/filenames). Accuracy = argmax of `answer_probs` vs gold; Wilson 95% CIs; paired bootstrap (2000 resamples) for delta vs the same image's original question. Full table: `table.md`, data: `results.json`, plot: `generalization.png`. n = 80 per cell (2,960 questions, 548 image groups). The first run was killed by reboot at ~200/548 groups; `run.py` resumes from `partial.json` and finished with identical item construction (seed fixed; 2960 items), no OOMs.

**Key numbers (acc, 95% CI; chance in brackets)**

| variant | acc | delta vs orig |
|---|---|---|
| cifar10 orig 10-opt | 0.963 [0.90,0.99] (0.10) | |
| cifar10 synonym labels / "a photo of a X" | 0.975 / 0.963 | +0.013 / 0.000 (n.s.) |
| cifar10 letter keys / digit keys | 0.963 / 0.963 | 0.000 / 0.000 (identical predictions) |
| cifar10 animal-vs-vehicle / living-vs-machine | 0.988 / 0.912 [0.83,0.96] (0.50) | +0.025 / -0.050 (n.s.) |
| food101 orig 20-opt | 0.850 [0.76,0.91] (0.05) | |
| food101 30-opt (>20, unseen) | 0.875 | +0.025 (n.s.) |
| food101 letter MCQ 20 opt / "homemade X" 8 opt | 0.850 / 0.900 | 0.000 / +0.050 (n.s.) |
| oxford_pets orig 20-opt / cat-vs-dog (new q) | 0.650 / 0.988 (0.50) | +0.338 |
| aokvqa orig / shuffled / keys wxyz | 0.550 / 0.525 / 0.537 (0.25) | -0.025 / -0.013 (n.s.) |
| scienceqa orig / keys wxyz | 0.512 / 0.550 (0.38) | +0.037 (n.s.) |
| eurosat orig / synonym descriptions | 0.950 / 0.925 (0.10) | -0.025 [-0.06,0.00] |
| coco (stage-1 only) orig 4-8 opt / 10 opt / 2 opt | 0.637 / 0.625 / 0.887 (0.17/0.10/0.50) | -0.013 (n.s.) / +0.25 |

Fewer options raise accuracy mechanically (higher chance); chance-normalized scores `(acc-ch)/(1-ch)` are in `results.json`/`table.md`. Starred deltas in the table (food 3-opt, pets 3/10-opt, aokvqa 2-opt, coco 2/3-opt) are all improvements consistent with easier tasks, not generalization failures.

**Findings**
- Format and key-symbol changes are essentially free: letter/digit/wxyz keys give identical or within-noise accuracy on all tasks (cifar10 A-J and digits: zero change).
- Option-count extrapolation works in both directions, including 30 food options (> the 20 seen in training; 0.875 vs 0.85 at 20). No degradation vs original anywhere; accuracy tracks chance-adjusted difficulty.
- Novel vocabulary costs little: synonyms ("aircraft", "lorry", "kitten") retain 0.975 on CIFAR; EuroSAT synonym descriptions lose at most ~2.5 pts (CI touches 0). Largest observed drop is cifar10 "living vs machine" (0.912, confidence falls 0.97 -> 0.77, delta -0.05 n.s.): decent but least confident, showing label-semantics reasoning is weaker than label matching.
- Unseen questions (2-way coarse categories) are answered at 0.91-0.99, far above chance, though these are easy reductions of fine labels the model already handles; they do not test genuinely new reasoning.
- COCO caption-choice, absent from the stage-2 mix, holds up (0.64 at 4-8 options, 0.89 at 2) and the 10-option padded version matches the original (distractors are random other-image captions, so easy).
- Hard-reasoning sets (aokvqa 0.55, scienceqa 0.51 vs 0.38 chance) are limited by capability, not format: perturbations change nothing.

**Caveats.** n=80/cell gives CIs of roughly +/-8-10 pts; deltas under ~7 pts are not resolvable. Perturbed questions reuse the same images/label sets (labels are CIFAR/Food/Pets classes the model was trained on), so this tests format/vocabulary generalization, not new visual domains. No truly new datasets or visual domains were available among held-out sets (only coco is outside the stage-2 mix). cat-vs-dog derived from filename capitalization (Oxford-Pets convention). Run was GPU-shared; no baseline model comparison was done (chance is the stated baseline).

**Take-aways**
1. The model is robust to answer-key format, option count (2-30) and option order/wording: no significant drop on any perturbation.
2. Vocabulary/semantic shifts cause small, mostly non-significant drops; the weakest case is abstract category words (living vs machine, 0.91 with lower confidence).
3. Residual errors are about task difficulty (aokvqa, scienceqa, fine-grained pets), not distribution shift; truly new visual domains remain untested.
