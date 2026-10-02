# Training runbook

This runbook trains Laya-Vision on a Linux machine with an NVIDIA GPU. Run every command from the repo root. The stages are:

```
setup → smoke test → data → stage 1 (align) → stage 2 (decide) → stage 3 (calibrate) → eval
```

The design is in `plan.md` §4 (Phases 4–5) and `ARCHITECTURE.md` §5. The optimisation recipe comes from Laya's fine-tuning notebook (`docs/reference/laya_finetune_notebook.py`):

- RLCD with G=4 noise samples, σ going linearly from 0.4 to 0.1, reward = log + 0.75·spherical − 1.0·RPS, group-centred and std-normalised advantage, plus soft CE with weight 1.0
- AdamW, cosine schedule, gradient clip 1.0

## Recommended recipe (B): commands and results

Recipe A (`stage1_a.yaml` → `stage2_a.yaml`, sections 3–4 below) fails acceptance. Its stage 1 (projector only, LR 1e-3) **collapses**: every image looks alike to Laya (cosine 0.9995 between different images' tokens after the embedding LayerNorm), and its caption accuracy comes from text priors (0.37 with the right image, 0.42 with a wrong one). Yes/no image questions then stay at chance through stage 2. Recipe B fixes this:

- guarded projector: `vision.proj_in_norm` and `vision.proj_standardize` (running mean/std on the projector output), plus `train.freeze_modality_emb`
- projector LR 1e-4, and LoRA on the encoder from stage 1 on, so Laya learns to read image tokens
- easier stage-1 negatives (`configs/data_stage1_easy.yaml`) and 10% text rows in stage 1
- a VQAv2 train-split continuation (`stage2_c.yaml`)
- weight interpolation toward stock Laya (`train.interpolate`, WiSE-FT) to undo text regression

Every trainer eval now also reports `image_grounding`, which is accuracy minus accuracy with each row's image swapped for another one in the batch, and `image_token_cos`. Grounding near 0, or token cosine near 1, means the model is not using the image.

```bash
python -m laya_vision.data.prepare --config configs/data_stage1_easy.yaml    # data_easy/coco.*.jsonl (reuses data/images/coco)
python -m laya_vision.data.prepare --config configs/data_stage2.yaml
python -m laya_vision.data.prepare --config configs/data_vqa_train.yaml      # data_vqa_train/vqav2_yesno.train.jsonl (160k)
python -m laya_vision.train.stage1 --config configs/stage1_b.yaml            # ~4.5 h on an 8 GB RTX 4060 Ti
python -m laya_vision.train.stage2 --config configs/stage2_b.yaml            # ~14 h
python -m laya_vision.train.stage2 --config configs/stage2_c.yaml            # ~4 h
python -m laya_vision.train.interpolate --checkpoint runs/stage2_c/best --alpha 0.85 --out runs/stage2_c/wise085
python -m laya_vision.train.calibrate --config configs/calibrate_c.yaml \
    --checkpoint runs/stage2_c/wise085 --out runs/stage2_c/wise085_calibrated
# then section 6 with --checkpoint runs/stage2_c/wise085_calibrated
```

The interpolation weight `alpha` was chosen on held-out data, never on the eval sets:

- a 3,000-item BoolQ **train**-split slice (`data/proxy/boolq_train3k.jsonl`) for text
- the image calib splits

The rule, fixed before evaluation, was: an estimated BoolQ test drop of at most 1.5 points, then the best mean image calib accuracy. On recipe B the train-split proxy overstated the test drop by about 1.8×.

Results (`eval/results/stage2_c_wise085.json`), **Acceptance: PASS**. KonIQ was not downloaded, so 3 image tasks are scored:

| | Recipe A | **Recipe B (2c, α=0.85)** | caption→Laya | stock Laya |
|---|---|---|---|---|
| EuroSAT | 0.929 | **0.970** | 0.389 | |
| Oxford Pets | 0.403 | **0.765** | 0.280 | |
| VQAv2 yes/no | 0.527 | **0.608** | 0.592 | |
| BoolQ | 0.719 | 0.747 | | 0.756 |
| AG News | 0.925 | 0.924 | | 0.926 |
| image ECE@10 | 0.010 | 0.014 | | |
| p50 latency vs text-only Laya | 1.69× | 1.68× | | |

On an 8 GB GPU, full fine-tuning of stage 2 runs out of memory, which is why every recipe-B stage uses LoRA.

## 0. Hardware and memory

These are estimates. Confirm them with `nvidia-smi` during the first 50 steps.

| Stage | Trainable | Weights + grads + AdamW | Suggested micro-batch (max_len 512, grad checkpointing) | Fits on |
|---|---|---|---|---|
| 1 | projector (1.8M), with grads flowing through frozen Laya | ~2 GB (fp32 weights, ~509M params) | 32 | 24 GB GPU (A10/L4/3090/4090) or larger; on 16 GB use micro 16 with accum 16 |
| 2 (full) | Laya encoder, head and projector (~423M) | ~2 GB weights + ~1.7 GB grads + ~3.4 GB AdamW ≈ 7–8 GB | 8 | 24 GB comfortably. The notebook ran the text-only model on 2×T4 16 GB with micro 8 |
| 2 (LoRA) | adapters (a few M), head and projector | ~2.5 GB | 8–16 | 16 GB |

- **Disk:**
  - Laya is ~1.7 GB and SigLIP-base ~0.8 GB.
  - Each checkpoint is ~0.9 GB, saved in fp16. A run keeps `checkpoint_latest`, `best` and `final`, and each of these holds `trainer_state.pt` with the AdamW state, which is ~3.4 GB in stage 2.
  - Images for stage 1 (COCO, 500k rows ≈ 125k images) and stage 2 need tens of GB.
- **Precision:** `amp: auto` gives bf16 on Ampere or newer (no GradScaler) and fp16 with GradScaler on older GPUs such as T4/V100. CPU and MPS run in fp32.
- **Throughput:** not measured on real hardware yet. Read the `sec` field in `metrics.jsonl` after ~50 steps and extrapolate the total time. Use `train.max_steps` for a timing run.

## 1. Environment

```bash
git clone <this repo> laya-vision && cd laya-vision        # or: rsync -a --exclude-from=.gitignore visual_laya/ host:laya-vision/
bash scripts/setup_train_env.sh                            # venv in .venv, CUDA torch, pip install -e ".[train,eval,serve,dev]"
source .venv/bin/activate
```

- The torch wheel index defaults to `cu124`. Match it to the driver's CUDA version (shown in `nvidia-smi`), for example `TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup_train_env.sh`.
- The script checks `torch.cuda.is_available()` and pre-downloads `convaiinnovations/laya` and `google/siglip-base-patch16-224`.
- Run `huggingface-cli login` (or `export HF_TOKEN=...`) for gated datasets, to avoid Hub rate limits, and to push checkpoints.

Check the install, then run the smoke test, which takes 1–2 min on CPU or GPU:

```bash
python -m pytest -q                        # unit tests (tiny models, CPU)
python -m pytest -m slow tests/test_real_weights.py   # real Laya + SigLIP: text parity with stock Laya on this machine's transformers (~2.5 GB download)
bash scripts/smoke_train.sh                # stage 1 → stage 2 → calibrate → VisionAgent.predict; ends with "SMOKE OK"
```

## 2. Data

The data is JSONL, one (state, question) row per line. The schema is in `docs/interfaces.md` and `laya_vision/data/schema.py`. Output goes to `data/{task}.{train,calib,test}.jsonl`, with images under `data/images/`. Check the licence table in `laya_vision/data/LICENSES.md` first. KonIQ has to be downloaded by hand into `data/raw/koniq10k` (see `configs/data_stage2.yaml`).

```bash
python -m laya_vision.data.prepare --config configs/data_stage1.yaml   # data/coco.*.jsonl       (stage 1, ~500k rows)
python -m laya_vision.data.prepare --config configs/data_stage2.yaml   # data/{task}.*.jsonl + data/typed_decisions.*.jsonl + data/{ag_news,boolq}.test.jsonl
```

- Pin the dataset `revision:` values in those configs before a real run.
- The trainer reads `data.train` as either a list of files or a `{name: {files, ratio}}` mixture, the same block as `mixture:` in the data configs.
- If you skip a task, remove its files from `configs/stage2_a.yaml`, `configs/calibrate.yaml` and `configs/eval.yaml`.
- The stage-2 data config also writes the text-regression sets `data/ag_news.test.jsonl` and `data/boolq.test.jsonl` (5,000 items each, `text_eval:` block) that `configs/eval.yaml` uses. To build only those, run `python -m laya_vision.data.prepare --text-eval`.

## 3. Stage 1: alignment (projector only)

> Recipe A, kept for reference. It collapses (see "Recommended recipe (B)" above); use `configs/stage1_b.yaml`.

```bash
python -m laya_vision.train.stage1 --config configs/stage1_a.yaml
```

- **Trainable:** pooler, projector and modality embedding, at LR 1e-3 with 3% warmup, cosine, weight decay 0, for 1 epoch.
- **Batch:** 256 rows per step (micro 32 × accum 8). On N GPUs keep micro × accum × N at 128–256.
- The Laya encoder, the head and the SigLIP tower are frozen. Gradients still flow through the frozen encoder to the projector (`detach_encoder: false`).
- **Outputs** in `runs/stage1_a/`:
  - `metrics.jsonl`: train loss, reward, σ, grad norm and LRs every `log_every` steps, plus eval every `eval_every` steps
  - `checkpoint_latest/` (rolling, with `trainer_state.pt`), `best/` (by `image.accuracy`) and `final/`
  - `config.yaml`: the resolved config
- The eval reports image and text rows separately:
  - accuracy, Brier, NLL, ECE@10 and ECE@15 (on max p), score MAE
  - logit mean, std and mean max-logit
- Text metrics should not move in stage 1, because Laya is frozen.
- **Exit (plan §4 Phase 4):** held-out caption-matching accuracy is clearly above chance and above SigLIP zero-shot on the same items:

  ```bash
  python -m laya_vision.eval.run_eval --checkpoint runs/stage1_a/best --data data/coco.test.jsonl \
      --image-root data/images --out eval/results/stage1_a.json
  python -m laya_vision.eval.run_eval --baseline siglip_zeroshot --data data/coco.test.jsonl \
      --image-root data/images --out eval/results/stage1_siglip_zs.json
  ```

- **Ablations** (each on a 100k subset), set with overrides:
  - `-o vision.feature_layer=-1`
  - `-o vision.pool_mode=shuffle`
  - `-o base=convaiinnovations/laya-multilingual`
  - Use `-o data.samples_per_epoch=100000 output_dir=runs/abl_fl1` for the subset size and a separate output directory.

## 4. Stage 2: decisions

```bash
python -m laya_vision.train.stage2 --config configs/stage2_a.yaml            # init_from: runs/stage1_a/final
```

- **LRs:** projector 1e-4, Laya encoder 2.5e-5, head/scorer/type embedding 1e-4. Weight decay 0.01, 3 epochs (plan says 2–4).
- **Batch:** effective 64 (micro 8 × accum 8 on 1 GPU; accum 4 on 2 GPUs).
- **Mix:** 75% image rows and 25% text-only typed-decisions rows.
- **Early stop:** training stops when held-out text accuracy (`data/typed_decisions.test.jsonl` in `data.eval`) falls more than 2 points (`early_stop_text_drop: 0.02`) below its step-0 value for `early_stop_patience` consecutive evals. The step-0 value is stock Laya, because stage 1 leaves Laya unchanged.

Variants:

- **LoRA on the encoder.** Use it if full fine-tuning regresses text by more than 2 points:

  ```bash
  python -m laya_vision.train.stage2 --config configs/stage2_a.yaml \
      -o train.lora.enabled=true output_dir=runs/stage2_a_lora
  ```

  The LoRA settings are r=16, alpha 32, on `Wqkv`, `Wo` and `Wi`. Adapters are merged before every save, so checkpoints stay plain Laya state dicts that stock `laya.Agent` can load. On resume, the adapters restart from the merged weights and the optimizer state is reset.
- **Unfreeze the last 2 tower blocks** at LR 1e-6. The tower is then saved with the checkpoint (`tower_trained: true`):

  ```bash
  -o train.unfreeze_vision_blocks=2 train.lr.vision=1.0e-6
  ```

## 5. Stage 3: calibration

```bash
python -m laya_vision.train.calibrate --config configs/calibrate.yaml
# == --checkpoint runs/stage2_a/best --out runs/stage2_a/calibrated --data data/*.calib.jsonl --image-root data/images --refit-text
```

- The script runs the model on the calib split. For each modality it fits T per question type and per `(qtype, temp_bucket)` bucket, using LBFGS on log T (the notebook's `fit_one_temp`). A bucket needs at least 100 items; smaller buckets fall back to the per-type T.
- Every T is clamped to [0.5, 5.0], the range the runtime applies. Pre-clamp values are logged and kept in `calibration_report.json`.
- Image temperatures go to `rl_agent_config.json` → `vision.temperature` / `vision.temperature_by_options`.
- Text temperatures replace the stock `temperature` / `temperature_by_options` only with `--refit-text`. `configs/calibrate.yaml` sets it because stage 2 changes the encoder. Drop it if the encoder was not trained.
- The script prints ECE@10, ECE@15 and NLL on `answer_confidence` (max p), before (current temperatures) and after, per modality.

## 6. Evaluation and acceptance

```bash
# baselines, once (pin captioner / SigLIP revisions in configs/eval.yaml first)
python -m laya_vision.eval.run_eval --config configs/eval.yaml --baseline caption_laya \
    --record-baseline eval/results/baselines.json --out eval/results/caption_laya.json
python -m laya_vision.eval.run_eval --config configs/eval.yaml --baseline stock_laya \
    --record-baseline eval/results/baselines.json --out eval/results/stock_laya.json
python -m laya_vision.eval.latency --checkpoint runs/stage2_a/calibrated --device cuda --stock \
    --out eval/results/latency.json
python -m laya_vision.eval.run_eval --config configs/eval.yaml --checkpoint runs/stage2_a/calibrated \
    --compare eval/results/baselines.json --latency eval/results/latency.json --out eval/results/stage2_a.json
```

**Exit (plan §4 Phase 5 / ARCH §6),** reported under `acceptance` in the output JSON:

- beats caption→Laya on at least 3 of 4 image tasks
- text accuracy within 2 points of stock Laya
- image ECE@10 ≤ 0.10
- p50 GPU latency for 1 question ≤ 2× text-only Laya

Use the calibrated checkpoint:

```python
import laya_vision
agent = laya_vision.load("runs/stage2_a/calibrated")
agent.predict({"image": "cat.jpg"}, {"q": {"type": "noul", "instructions": "Is there a cat?"}})
```

## 7. Multi-GPU (torchrun DDP)

```bash
torchrun --standalone --nproc_per_node=2 -m laya_vision.train.stage2 --config configs/stage2_a.yaml -o train.grad_accum=4
```

- Each rank reads a disjoint shard of the same deterministic (seed, epoch) mixture order.
- Rank 0 evaluates and saves. The other ranks wait and receive the early-stop decision.
- `find_unused_parameters=True` is always on: text-only batches leave the projector unused, and `feature_layer: -2` leaves the last tower block unused.
- Keep micro × accum × world at the effective batch.

## 8. Resume

`resume: auto` (the default) continues from `<output_dir>/checkpoint_latest` when it holds a `trainer_state.pt`. That file stores:

- model weights, optimizer, scheduler and GradScaler state
- step, epoch and position within the epoch
- best metric and the text baseline

Re-run the same command. To resume from an explicit directory use `-o resume=runs/x/checkpoint_latest`, and to start fresh use `-o resume=false`. Resumed runs start from the fp16-saved weights, which is a negligible change. Save frequency is `train.save_every` optimizer steps (default 500).

## 9. Config knobs (all overridable with `-o dotted.key=value`, values parsed as YAML)

| Key | Meaning |
|---|---|
| `base`, `base_revision` | Laya checkpoint (Hub id or dir); `tiny` for tests |
| `init_from` | laya_vision checkpoint to start from (stage 2) |
| `vision.{tower,feature_layer,pool_mode,pool_k}` | image tower and token layout (A: siglip-base, -2, avg, 2 → 49 tokens) |
| `data.train` | files, or a `{name: {files, ratio}}` mixture |
| `data.eval` | held-out files, scored per modality |
| `data.samples_per_epoch` | rows per epoch (default: one pass over the largest source relative to its ratio) |
| `data.max_eval_items` | cap on the eval set |
| `data.augment` | `true`, or `Augmenter` kwargs (option shuffle, renaming, paraphrase) |
| `data.max_len`, `head_max_len` | default: the Laya config (512 / 192 for English) |
| `train.lr.{projector,encoder,head,vision}` | 0 = frozen |
| `train.unfreeze_vision_blocks` | train only the last N tower blocks |
| `train.micro_batch`, `grad_accum`, `epochs`, `max_steps` | batch geometry and length |
| `train.warmup_ratio`, `min_lr_ratio`, `weight_decay`, `grad_clip` | optimiser |
| `train.group_size`, `sigma_start`, `sigma_end`, `w_sph`, `w_rps`, `ce_weight` | RLCD (notebook: 4, 0.4, 0.1, 0.75, 1.0, 1.0) |
| `train.amp` | auto / bf16 / fp16 / fp32 |
| `train.gradient_checkpointing` | encoder + head checkpointing |
| `train.lora.*` | LoRA variant |
| `train.eval_every`, `save_every`, `log_every`, `best_metric`, `early_stop_text_drop`, `early_stop_patience` | eval, checkpointing and early stop |
| `wandb.enabled` | log to Weights & Biases if installed |

## 10. Troubleshooting

- **CUDA OOM:**
  - Halve `train.micro_batch` and double `train.grad_accum`.
  - Keep `gradient_checkpointing: true`.
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, as the notebook does.
- **`image does not fit` / rows silently dropped:** a question with many or long options leaves no room for the 49 image tokens inside `max_len`. The dataset drops such rows: eval sets are pre-scanned and the count is logged, and training rows are dropped in the collate. Shorten the options or reduce the choice option count in the data config.
- **Non-finite loss:** the trainer raises `FloatingPointError` with the stats. With fp16 on older GPUs, try `-o train.amp=fp32` or lower LRs, then resume from `checkpoint_latest`.
- **Text accuracy drops early in stage 2:** switch to the LoRA variant, lower `lr.encoder` to 1e-5, or raise the text ratio to 0.3.
- **`wandb disabled: ...`:** wandb is optional. Run `pip install wandb && wandb login`, or leave `wandb.enabled: false`.
- **DDP hangs at start:** check that every rank sees the data files and that `NCCL_P2P_DISABLE=1` is not needed on the box (some consumer multi-GPU boards need it).
- **Hub downloads during training:** run `setup_train_env.sh` without `SKIP_DOWNLOAD`, or set `HF_HUB_OFFLINE=1` once everything is cached.
