# Handoff: training pipeline — DONE

Status as of 2026-09-28: all owned files are done and tested on CPU, including 2-process DDP on gloo. **Not yet run on a GPU or with the real Laya/SigLIP weights.**

| File | Status |
|---|---|
| laya_vision/train/__init__.py | done |
| laya_vision/train/rlcd.py | done: `rlcd_loss`, `sigma_at` (exact notebook loss) |
| laya_vision/train/loop.py | done: config, trainer, sampler, eval, save/resume, synthetic toy data |
| laya_vision/train/stage1.py, stage2.py | done (thin CLIs over `loop.cli`) |
| laya_vision/train/calibrate.py | done |
| laya_vision/train/lora.py | done (`inject_adapter_in_model`, merged copy on save) |
| configs/stage1_a.yaml, stage2_a.yaml, calibrate.yaml, smoke.yaml | done |
| scripts/setup_train_env.sh, scripts/smoke_train.sh | done |
| TRAINING.md | done (the runbook) |
| tests/test_rlcd.py, tests/test_train_smoke.py | done |

## Commands
```
python -m laya_vision.train.stage1 --config configs/stage1_a.yaml [-o key=value ...]
python -m laya_vision.train.stage2 --config configs/stage2_a.yaml
python -m laya_vision.train.calibrate --config configs/calibrate.yaml
torchrun --standalone --nproc_per_node=N -m laya_vision.train.stage2 --config ... -o train.grad_accum=...
bash scripts/smoke_train.sh      # ~20 s on CPU, ends with "SMOKE OK"
```

## Tests
- `.venv/bin/python -m pytest tests/test_rlcd.py tests/test_train_smoke.py -q`: **11 passed** in ~7 s.
- Full suite: 142 passed, 2 deselected (slow).
- ruff (E9,F63,F7,F82,F401,F811): clean.

## Design notes and gotchas
- **Data sources.** `data.train` takes the data agent's `{name: {files, ratio}}` mixture block, or a list of files.
- **Sampler.** The loop uses its own `EpochSampler` rather than `data.dataset.MixtureSampler`. It has the same semantics, and it adds a mid-epoch resume offset and truncates to equal per-rank lengths.
- **Mixture epoch size.** The default is one pass over the source that is largest relative to its ratio.
- **Skipped rows.** `DecisionDataset` returns None for rows that don't fit, and the collate can return a None batch.
  - The loop skips None batches. Under DDP the ranks agree on the skip via an all_reduce.
  - Eval sets are built with `prescan=True`, on rank 0 only.
- **Loss scaling.** Gradient accumulation divides by the actual number of micro-batches in the group, and the loss adds `0*act.sum()`. The notebook always divided by GRAD_ACCUM, so the last partial group of an epoch now gets the correct weight.
- **σ schedule.** σ is linear over optimizer-step progress, not per epoch as in the notebook.
- **LR schedule.** Warmup (`warmup_ratio`) is followed by cosine decay to `min_lr_ratio`. The notebook used cosine with no warmup and eta_min 1e-6.
- **Calibration clamp.** Calibration clamps to [0.5, 5.0], where the notebook used [0.1, 10], and logs the pre-clamp values.
  - At least 10 items are needed for a per-type T and `min_bucket` (100) for a per-bucket T.
  - With `--refit-text`, the bucket dict is replaced entirely.
  - The "before" temperatures follow VisionAgent's fallback rule.
- **Resume.** It restores optimizer, scheduler, scaler, step and epoch position, the best metric and the text baseline. LoRA resume restarts the adapters from the merged weights and resets the optimizer.
- **Vision tower.** When `lr.vision > 0`, the loop sets `vcfg.tower_trained=True`, so the tower is saved with the checkpoint.
- **Smoke run limits.** The tiny random Laya gives nearly flat logits, so the smoke runs learn nothing and the fitted T stays at 1.0. The smoke tests the plumbing only.

## Unfinished / depends on others
- Real-weight run: nothing is verified on GPU or with ModernBERT-large + SigLIP-base. Memory figures in TRAINING.md are estimates.
- The configs reference data files produced by `laya_vision.data.prepare` (`data/{task}.{split}.jsonl`, images under `data/images`). Keep them in sync if the data configs change.
