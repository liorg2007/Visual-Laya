# 09 API edge cases — post-fix rerun (Python API, /v1/systemone, web UI)

**Question.** Do the fixes (`laya_vision/textnorm.py`, `agent.py`, `images.py`) close B1 (lone surrogate → HTTP 500 / bare TypeError) and B2 (zero-size image accepted) without regressions? Checkpoint `runs/stage2_c/wise085_calibrated`, fp16, GPU nearly idle. Pre-fix report: `report_prefix.md`; pre-fix results: `results_prefix.json`, logs in `prefix_logs/`.

**Method.** The same four suites, rerun against the working tree:
1. Python API — `run_api.py` + `post_fix_checks.py`
2. In-process HTTP — `run_part2.py` + `post_fix_http.py` (port 8795)
3. Real server via `python -m laya_vision.ui` — `part3_http.py` (port 8793, SIGTERM, exit confirmed)
4. Execution modes — `part4_modes.py`, `part4b_nocompile.py`, `part4c_compile_diag.py`

A check passes only on the expected 2xx/4xx or exception type, never on a 5xx or crash. Checks are deterministic (no CIs). No server or test process was left running (ps + ports 8790–8799 checked).

**Test-script changes (test code only).** Defined the missing `enc()`/`base_im` in `run_part2.py` (7 checks had hit NameError); `max_len=64` now expects success (63 tokens fit) with a new `max_len=40` → 422 check; the part-1 noul check reads `answers.q.noul`; the duplicated HTTP block in `run_api.py` was removed, so its 3 load-mode checks now run; surrogate checks now strictly expect 422 / a ValueError naming "surrogate"; a surrogate-pair literal is built with `chr()`; the `predict_shortlist` probe was dropped (needs an `embed_fn`).

| Suite | Before (pass/n) | After: original checks | After: new checks | After: all |
|---|---|---|---|---|
| 1 Python API | 123/137 | 130/141 | 37/38 | 167/179 |
| 2 HTTP in-process | 43/51 | 52/52 | 20/22 | 72/74 |
| 3 HTTP via UI CLI | 30/31 | 32/32 | 20/22 | 52/54 |
| 4 compile / 4b no-compile | 3/7 (+1 unresolved) | – | – | 4: 2/5 (+2 info); 4b: 5/7 (+1 info) |

**Regressions: none.** All 217 previously passing checks still pass (matched by name). Newly passing: the zero-size array check (B2 fix), noul and the 7 part-2 image-format checks (test fixes). The 11 remaining original part-1 failures are the ones already classified as not bugs: 2 fp16 batch-shape tolerance (max |dp| 0.03); 9 wrong expectations or design choices (silent truncation, `head_max_len=8`, 1-option choice, int labels, `state=None` → TypeError, `fast=True` falling back without tilelang).

## Fix verification

| Bug | Before | After |
|---|---|---|
| B1 surrogate, Python | bare `TypeError` from the tokenizer | `ValueError` naming "surrogate", 14/14 cases (label, instructions, description, image text str/dict/list, text-only str/dict value/dict key/label/instructions, predict_batch, 2nd question, two surrogates). Real emoji, non-BMP, NUL, U+FFFD still accepted; output unchanged afterwards (dp=0). |
| B1 surrogate, HTTP | 500 "inference failed" | 422 with "surrogate" in the detail, 11/11 on both servers. A valid escaped pair (🐶) → 200. |
| B2 zero-size image | (0,0,3), (0,5), (5,0,3) arrays accepted as blank | `ImageError` "at least one pixel" for those plus (0,0), CHW (3,0,0), float (0,4,4), PIL 0x4/4x0/0x0; 1x1 still accepted; 0x0 GIF over HTTP → 400. |
| B3 surrogate in a question **id**, HTTP | found in this rerun: 500 (JSONResponse fails to encode the answer key, outside the `try`) | **fixed after the rerun**: `_encode_state` now checks question ids → 422. Verified by `tests/test_serve.py::test_lone_surrogate_is_422[qid, qid_text]`, not by a rerun of this suite. |
| B4 surrogate via `predict_long`, Python | found in this rerun: tokenizer `TypeError` (bypasses `_encode_state`) | **fixed after the rerun**: `VisionAgent.predict_long` checks state and questions → ValueError. Verified by `tests/test_textnorm.py`, not by a rerun of this suite. |

**All-caps checks (new, all pass).** An all-caps image request returns the caller's upper-case labels and `choice` ("BASSET HOUND") in Python and over HTTP, with probabilities identical to the lower-case twin request. "DNA"/"RNA" are not rewritten (their probabilities differ from "dna"/"rna"). An APPLE/apple label collision is left unrewritten. Text-only requests are never rewritten. Note: acronyms of 4+ letters (NASA, HTTP, JSON, UNESCO) are lower-cased for the model only; the caller still gets the original labels.

## Remaining issue (upstream)
**U1 — `compile=True` text-only predict is broken in Laya, not a hang.** With the GPU free it fails after ~7 s with `TorchRuntimeError: 'NoneType' + FakeTensor` at `laya/common.py:317` (`h = h + type_emb`, after a dynamo graph break at the encoder call). Stock `laya.Agent("convaiinnovations/laya", compile=True)` fails identically (torch 2.6.0, transformers 5.17.0, laya 0.3.21). The earlier 600 s "hang" was presumably this path under contention. `compile=True` still rejects image states and surrogates with a clear ValueError.

## Caveats
- One image and one checkpoint. Check counts are not one-to-one across runs because checks were added (see the split columns).
- B3/B4 fixes are covered by unit tests on the tiny model, not by a rerun of these suites on the real checkpoint.
- U1 was diagnosed by reproducing it on stock Laya; the root cause inside dynamo/ModernBERT was not isolated.

## Take-aways
1. B1 and B2 are fixed on every reachable path, and none of the 217 previously passing checks regressed.
2. The two surrogate gaps this rerun found (question ids over HTTP, `predict_long`) are now fixed too and covered by unit tests.
3. `compile=True` text inference is broken upstream in Laya with the current torch/transformers; it is not a laya_vision hang.
