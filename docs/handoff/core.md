# Handoff: core model — DONE

## Status
All owned files are done and tested:
- `laya_vision/images.py`, `vision.py`, `projector.py`, `model.py`, `checkpoint.py`, `agent.py` and `__init__.py`, which lazily exports `load`, `VisionAgent`, `VisionConfig` and `LayaVisionModel`.
- Tests: `tests/test_sequence.py`, `test_images.py`, `test_shapes.py`, `test_text_parity.py`, `test_cache.py`, `test_checkpoint.py` and `test_agent_api.py`, plus the helper `tests/_core_utils.py`, which has seeded weight perturbation and cached tiny vision checkpoints.

## Test status
- Core tests: `75 passed`.
- Full `pytest -q`, including the other agents' tests at that time: `108 passed, 1 deselected`.
- `ruff check --select=E9,F63,F7,F82,F401,F811`: clean. `compileall`: ok.

## Design as implemented
- **VisionAgent** subclasses `laya.Agent` and overrides only `_encode_state`, `_forward`, `_decode_answers` and `predict_long`.
  - `predict`, `system_one` and `predict_batch` (hooks, batching, usage, `min_confidence`) are Laya's own.
  - Text-only requests go through the stock methods, so their JSON is identical to `laya.Agent` (tested on about 40 requests plus batches).
  - Image items carry `image_ref` (the sha256) and `pixel_values`, which pass by reference into `b["meta"]`. `_forward` runs one vision forward per distinct image in the pass.
  - Image rows are decoded by the stock `Agent._decode_answers` bound to a namespace holding the image temperatures, which fall back to the text temperatures.
- **Checkpoint:** stock `laya.Agent` loads a saved vision checkpoint; this is tested. For the tower rule, see the "Core-model additions" section of docs/interfaces.md.
- **Parity test:** `LayaVisionModel` with no images vs `DecisionModel` gives max |Δlogit| ≤ 1e-5.

## Gotchas
- The tiny random Laya produces almost identical hidden states at every [MASK], so all option logits come out equal. The tests perturb every weight (std 0.3, seeded) so that parity and round-trip comparisons are meaningful.
- An fp16 save cannot be compared with the fp32 model at a tight tolerance. Round the reference model's `state_dict()` to fp16 instead. Do not round every buffer, or RoPE `inv_freq` changes too.
- Different vision batch sizes change logits by about 1e-4 at |z| of about 15. That is float noise, not a bug.
- `LayaVisionModel.__init__` sets `encoder.config.reference_compile=False`. VisionAgent then restores the value `laya.Agent` chose (compile=True).
- Not yet verified on transformers 5.x (only 4.53.3 is available locally). The 5.x-sensitive calls are:
  - `SiglipVisionModel._from_config`, which has a fallback
  - `ModernBertModel(inputs_embeds=...)`
  - `hidden_states[-2]`

## Possible follow-ups (not required)
- Optional cross-request LRU image cache (plan §2.6, default off): not implemented.
- `@pytest.mark.slow` test against the real `convaiinnovations/laya` and SigLIP-base: not written.
