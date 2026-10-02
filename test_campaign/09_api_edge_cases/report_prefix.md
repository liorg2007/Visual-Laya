# 09 API edge cases: Python API, /v1/systemone, web UI

**Question.** How robust are `laya_vision` (Python `load`/`predict`/`predict_batch`), the HTTP `/v1/systemone` endpoint and the bundled web UI (`python -m laya_vision.ui`) to malformed or extreme input? Checkpoint: `runs/stage2_c/wise085_calibrated`, fp16, shared GPU. A previous run was killed by a reboot; parts 3 and 4 were completed here.

**Method.** Four scripted check suites (no repo edits, no training; deterministic inputs, single image `oxford_pets/basset_hound_129.jpg`): part 1 Python API (`run_api.py`), part 2 in-process HTTP (`run_part2.py`), part 3 real server started through the UI CLI on port 8793 and shut down afterwards (`part3_http.py`), part 4 execution modes (`part4_modes.py`, `part4b_nocompile.py`). Checks are pass/fail on "clean 2xx or clean 4xx, never 5xx/crash". No accuracy metrics, so no CIs apply; counts below are check counts.

| Suite | Checks | Pass | Fail |
|---|---|---|---|
| 1 Python API | 137 | 123 | 14 |
| 2 HTTP in-process | 51 | 43 | 8 |
| 3 HTTP via UI CLI | 31 | 30 | 1 |
| 4 / 4b modes | 7 | 3 | 3 (+1 info) |

**Classification of the 8 part-2 FAILs.** 7 are a test bug (`enc()` undefined in `run_part2.py`, so NameError and no request was sent: PNG, BMP, TIFF, RGBA PNG, CMYK JPEG, 1x1, 7000x7000); re-run in part 3 they all behave correctly (200, 400, 400, 200, 200, 200, 413). 1 is a wrong expectation: `max_len=64` is exactly enough for 49 image tokens plus the head (63 tokens), so 200 is right; `max_len=40` correctly returns 422 with a readable message.
The 14 part-1 FAILs: 1 real minor bug (zero-size ndarray accepted), 2 fp16 tolerance (batch-shape dependent probabilities differ by up to 0.03; same-shape repeats are bit-identical), and the rest are wrong expectations or test bugs (noul result has no `probabilities` key; `fast=True` without tilelang falls back with a warning; truncation/clipping of very long text instead of an error; `state=None` gives a clear TypeError). Details in `results.json`.

**Findings.**
- Validation is solid: 400 for BMP/TIFF/invalid JSON/deeply nested JSON, 413 for 7000x7000 images, 65 questions, 300 options; 422 for an image that does not fit and for malformed score questions. All image modes/formats (RGBA, L, P, CMYK, GIF, WEBP, 4000x100, 1x1) run. Server stays healthy after abuse; `/` serves the UI with the file input and `/v1/systemone` reference.
- **Real bug B1 (medium):** a lone surrogate (JSON `"\ud800"`) in an option label, instruction or state text raises `TypeError` in the tokenizer, which `serve.py` does not map to 4xx, so the server returns **500 "inference failed"** (and Python raises a non-ValueError TypeError). Repro: `part3_http.py` (HTTP) and `part4b_nocompile.py` (Python).
- **Real bug B2 (low):** `images._from_array` accepts zero-size arrays (`np.zeros((0,0,3),np.uint8)`) and predicts on them like a blank image; not reachable via HTTP.
- Modes: `compile=True` loads and rejects image states with a clear ValueError; a simulated fast path does too. Text-only `predict` under `compile=True` did not finish within 600 s (inductor compile on a contended GPU), so it is unverified. `accelerate(strict=True)` raises ModuleNotFoundError (tilelang not installed), as expected.

**Caveats.** Single image and single checkpoint; GPU was shared, so cold start was 34 s and compile timing is not representative. Silent truncation of very long text and `head_max_len=8` acceptance are design behaviors, noted not scored as bugs.

**Take-aways.** (1) The API/HTTP/UI handle nearly all edge cases with clean 4xx errors. (2) Fix surrogate handling (catch TypeError or sanitize text) to avoid the only observed 500. (3) The earlier 8 part-2 failures are almost entirely test defects, not model/server defects.
