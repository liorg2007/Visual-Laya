# Handoff: eval / baselines / serve — DONE

Status as of 2026-09-28: all owned files are done and tested.

- Test command: `.venv/bin/python -m pytest tests/test_metrics.py tests/test_eval_run.py tests/test_serve.py -q`
- Test result: **27 passed, 1 deselected** (the deselected one is `@pytest.mark.slow`, which downloads BLIP and SigLIP).
- ruff (E9,F63,F7,F82,F401,F811): clean.

| file | status |
|---|---|
| laya_vision/eval/__init__.py | done |
| laya_vision/eval/metrics.py | done |
| laya_vision/eval/baselines.py | done |
| laya_vision/eval/run_eval.py | done |
| laya_vision/eval/latency.py | done |
| laya_vision/serve.py | done |
| configs/eval.yaml | done |
| tests/test_metrics.py, test_eval_run.py, test_serve.py | done |

## CLI
```
python -m laya_vision.eval.run_eval --checkpoint DIR --data a.jsonl b.jsonl --image-root ROOT --out res.json
python -m laya_vision.eval.run_eval --config configs/eval.yaml --baseline caption_laya|siglip_zeroshot|stock_laya \
    --record-baseline eval/results/baselines.json --out eval/results/<name>.json
python -m laya_vision.eval.run_eval --config configs/eval.yaml --checkpoint DIR \
    --compare eval/results/baselines.json --latency eval/results/latency.json --out eval/results/model.json
python -m laya_vision.eval.latency --checkpoint DIR [--device cuda] [--stock] [--out eval/results/latency.json]
python -m laya_vision.serve --checkpoint DIR --port 8000 [--device cuda]
```

## Outputs of `run_eval --out X.json`
- `X.json`:
  - metrics: overall / by_task / by_modality / by_qtype / by_modality_qtype, each with acc, brier, nll, ece10, ece15, score_mae, reliability
  - `task_modality`
  - `acceptance`, when `--compare` is passed
- `X.predictions.jsonl`
- `X.md`: summary table plus the acceptance result

## Design notes and gotchas
- **Grouping**:
  - Records are grouped by (image, text) state, so there is one `predict` per state.
  - `--max-questions` chunks large groups (default 32).
  - A `ValueError` or `OSError` for a state marks its rows as skipped, with the `error` field set.
- **Metrics**:
  - Gold = argmax(target). Brier and NLL use the full (possibly soft) target distribution.
  - Score MAE = |E_p[level] − E_target[level]|.
  - ECE uses `laya.common.ece_score`, with confidence = max(p).
- **baselines.json** has the shape `{baseline_name: run_eval result}`. `--record-baseline` merges each run into it.
- **Acceptance** (plan §4 Phase 5):
  - beats caption_laya on ≥ ceil(0.75·#image tasks) tasks
  - text accuracy drop ≤ 0.02 vs stock_laya on every text task
  - image ECE@10 ≤ 0.10
  - latency: `image_q1` / `text_q1` p50 ≤ 2× from the latency JSON; the criterion is meant for GPU, and a CPU run is flagged
- **Baselines**:
  - `caption_laya` uses a stock `laya.Agent` loaded from `--checkpoint`, or from `--laya` (default convaiinnovations/laya).
  - The caption state is the caption string, or `{"image_caption", "ocr_text"?, "text"?}`.
  - The caption cache lives at `<cache>/<captioner id>/<sha256>.json`.
  - `siglip_zeroshot` skips noul and score questions, which show up as `skipped` in the metrics.
  - Captioner and SigLIP revisions are unpinned (`null`) in configs/eval.yaml. **Pin them before recording baselines.json.**
- **serve.py**:
  - Only base64 or data-URL images are accepted; path strings get 400, so the server never reads local files.
  - Images are limited to 10 MB decoded and 40 MP (413) and to the formats JPEG/PNG/WEBP/GIF (400). BMP is refused even though `images.py` accepts it.
  - The text part and the questions go through `laya.serve._check_request_limits`.
  - Text-only bodies keep Laya's 2 MiB cap. Image bodies are capped by `LAYA_VISION_MAX_BODY_BYTES` (default ≈ 16 MiB).
  - `LAYA_API_KEY`, `LAYA_MAX_CONCURRENT`, `LAYA_MAX_TOKEN_BUDGET` and `LAYA_THREADS` behave as in laya.serve.
  - Text-only responses equal `laya.serve` responses when laya.serve serves a single `laya.Agent` through an injected router (tested). The real laya `Router` adds a `routing` key, which a single-checkpoint server does not.
- The tests build a tiny checkpoint with `checkpoint.init_from_laya(tiny_laya_dir, VisionConfig(image_size=32, patch_size=8, vision_width=32, pool_k=2), tower_config=tiny_siglip_config)` and `save_checkpoint`.

## Not done / follow-ups
- There are no real numbers yet. `data/*.test.jsonl` does not exist until the data pipeline produces it; the paths in configs/eval.yaml assume that naming.
- `eval/results/latency_phase2.json` is not recorded yet. Run latency.py on the target GPU.
- The OCR hook is a plain callable (`ocr=`), and no OCR engine is wired into the CLI.
