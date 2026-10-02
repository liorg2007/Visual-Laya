# Test campaign brief (shared by all test agents)

Model under test: Laya-Vision, final checkpoint `runs/stage2_c/wise085_calibrated`
(load: `import laya_vision; agent = laya_vision.load(path, device="cuda")`; `agent.predict(state, {"q": {...}})`).
Other checkpoints for comparison: runs/stage2_a/calibrated, runs/stage2_b/calibrated,
runs/stage2_b/wise080_calibrated, runs/stage1_* (see `ls runs`). Stock text Laya: `laya.Agent("convaiinnovations/laya")`.
Docs: ARCHITECTURE.md, TRAINING.md, docs/interfaces.md, docs/handoff/eval.md, laya_vision/agent.py, laya_vision/eval/.
Held-out data: data/*.test.jsonl (images in data/images), existing results in eval/results/ (do not overwrite).
Use `.venv` python if present (else system python that imports laya_vision).

Rules
- Write ONLY inside your own directory under /home/zera/laya-vision/test_campaign/<your class>/. Do not edit repo source, configs, or other classes' dirs. No training. Put scripts there too.
- GPU is an RTX 4060 Ti 8 GB with ~4 GB already used and 11 other agents sharing it: use fp16/autocast, small batches, load ONE model at a time, free memory (del + torch.cuda.empty_cache()) and fall back to CPU if you hit OOM. Keep total runtime under ~25 min; subsample test sets (fixed seed, e.g. n<=500 per task) where needed.
- Be rigorous: fixed seeds, report n, give 95% CIs (bootstrap or Wilson) for accuracies, compare against a stated baseline/chance level, and say honestly when differences are within noise.
- Deliverables in your dir: `results.json` (machine-readable), `report.md` (<=1.5 pages: question, method, numbers table, findings, caveats, 3 take-aways), plus scripts and any plots (png). Do not fabricate numbers; if something failed, report it.
- Final reply to the orchestrator: <=10 lines with headline numbers and the path to report.md.
