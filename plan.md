# Laya-Vision — Model Description and Implementation Plan

**Status:** plan, nothing implemented yet. This builds on `ARCHITECTURE.md`, and that file stays the design rationale.
**What this file adds:**
1. A full description of the model down to tensor shapes and module definitions.
2. Answers to the `[VERIFY]` items in `ARCHITECTURE.md §8`, checked against the Laya source.
3. A phased implementation plan with deliverables, tests and exit criteria.

**Source checked:** `NandhaKishorM/laya` at `main` (package version **0.3.21**), the fine-tuning notebook `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, the shipped `convaiinnovations/laya` `rl_agent_config.json`, and `transformers` **5.17.0** (`modeling_modernbert.py`, `modeling_siglip.py`).
File and line references below point into those sources.

---

## 0. Summary

Laya-Vision is stock Laya (ModernBERT-large encoder + typed decision head, 421M params) with one extra input path:

```
image → SigLIP vision tower → 2D pool → MLP projector → +modality embedding → N_img × 1024 vectors
```

These vectors replace placeholder tokens at the start of Laya's `<state>` region. They enter the encoder through `inputs_embeds`. Everything after the encoder is Laya's code and weights, unchanged. The output schema is also unchanged: `choice` / `score` / `noul`, with probabilities, `confidence` and `answer_confidence`.

Four design properties make this cheap to build and safe to ship:

- **No Laya source changes.** `build_sequence` accepts pre-tokenized `state_ids`, so the image placeholders are passed in as state tokens (§2.4). The model reuses `DecisionModel`'s modules and adds only a new forward.
- **Checkpoint is a superset of a Laya checkpoint.** `model.safetensors` keeps Laya's exact keys, so stock `laya.Agent` can still load it for text-only use. Vision weights live in a separate `vision.safetensors`.
- **Vision work runs once per image**, not once per question (§2.6).
- **Text-only behaviour is bit-for-bit Laya** when no image is present. This is enforced by a parity test (§4, Phase 1).

---

## 1. Findings from the Laya source (the `[VERIFY]` list, resolved)

| # | Question (ARCHITECTURE §8) | Finding | Consequence for the plan |
|---|---|---|---|
| 1 | Where is `build_sequence`, and how are markers recorded? | `laya/common.py:135`. It returns `(ids, markers[, stats])`, where `markers` are absolute positions of each option's `[MASK]`. It accepts **`state_ids`**, a pre-tokenized state that it slices to the remaining room: `state_ids[:room]`, or the tail when `truncate_left=True`. | Pass `[PAD]*N_img + [SEP] + text_ids` as `state_ids`, so no fork of `build_sequence` is needed. Force `truncate_left=False` so text is cut and the image never is. |
| 2 | How is the backbone called? Is `inputs_embeds` supported? | `DecisionModel.forward` (`common.py:313`) calls `self.encoder(input_ids=..., attention_mask=...)` only. The encoder is built with `attn_implementation="sdpa"` (`common.py:397/402`), so there is no flash-attn unpadding path. `ModernBertModel.forward` in transformers 5.17 accepts `inputs_embeds` (it requires exactly one of the two inputs). | New forward: compute embeddings ourselves and call `encoder(inputs_embeds=...)`. The head part is copied from `DecisionModel.forward`, and a parity test catches drift. |
| 3 | Is the embedding LayerNorm applied to `inputs_embeds`? | **Yes.** `ModernBertEmbeddings.forward` does `drop(norm(inputs_embeds))`, and for text `drop(norm(tok_embeddings(ids)))` (`modeling_modernbert.py:64-70`). | Overwrite placeholders in the **pre-LN** `tok_embeddings` output. The shared LN then normalises image and text tokens the same way, so the "scale matching" worry in ARCH §3.3 mostly goes away. Still re-check on the lowest supported transformers 4.x version (Phase 0). |
| 4 | Truncation logic for the state region | `room = max(0, max_len - len(head) - 1)`. List states (conversations) use `truncate_left=True` (`agent.py:765`). The final sequence is also hard-cut to `max_len`. | Assert `room >= N_img + 1` before building. For list-type text states, left-truncate the **text** ourselves before appending it after the image block. |
| 5 | Is RLCD training importable? | **Only partly.** `build_model`, `proper_reward`, `collate_items`, `ece_score` and `temp_bucket` are importable from `laya.common`. The training loop (noise sampling, advantage, loss) and `fit_one_temp` exist only in the notebook. | Port the loop into `laya_vision/train/rlcd.py` (about 60 lines) and import everything else. |
| 6 | How are temperatures stored? | `rl_agent_config.json` has `temperature: [choice, score, noul]` and `temperature_by_options: {"choice:2", "choice:3-5", "choice:6-10", "choice:11+", "score:3-5", "noul:2", ...}` (buckets from `temp_bucket`, `common.py:499`). The runtime clamps them to **[0.5, 5.0]** (`clamp_temperature`). | Keep text temperatures in the stock fields. Add image temperatures under `cfg["vision"]["temperature"]` and `cfg["vision"]["temperature_by_options"]`, and select by modality at decode time. |
| 7 | Dataset licences | Not checkable from code. | Phase 3 task: a licence table before any download goes into training. |

### 1.1 Corrections to `ARCHITECTURE.md`

- **RLCD hyperparameters.** ARCH §1/§5.3 says G=8, σ 1.0→0.3, w_sph=0.5, pure REINFORCE. The published fine-tuning notebook, which is the only runnable training code, uses different values:
  - **G=4**, **σ 0.4→0.1** (linear per epoch)
  - **w_sph=0.75**, w_rps=1.0
  - advantage = group-mean-centred, then divided by its std
  - **plus a soft cross-entropy term with weight 1.0**
  - AdamW with weight decay 0.01, cosine schedule, grad clip 1.0, fp16 autocast
  - LR encoder 2.5e-5, head 1e-4
  - 4 epochs, effective batch 64

  **Plan: use the notebook's values as the known-working default.** Treat the ARCH values as an ablation.
- **Confidence.** Laya returns both. `confidence` is `1 − H/log K` for choice/score and `max(p)` for noul. `answer_confidence` is `max(p)` and is the calibrated one (`common.py:470-496`). Gate and compute ECE on `answer_confidence`.
- **ECE bins.** `laya.common.ece_score` defaults to 15 bins. Report ECE@15 for comparability with Laya, and ECE@10 as ARCH specifies.
- **Local attention window.** ModernBERT's `local_attention=128` means ±64 tokens (`sliding_window = local_attention // 2`). Vertical neighbours are 7 or 9 positions apart, so the conclusion in ARCH §9 still holds.
- **Temperature clamp.** Any fitted temperature outside [0.5, 5.0] is clamped at load time with a warning. Our calibration must fit inside that range, or the saved values will not be the ones applied.

---

## 2. Model description

### 2.1 End-to-end data flow with shapes

Notation: `B` = question rows in the batch, `M` = distinct images in the batch, `L` = padded sequence length, `N` = image tokens per image, `d = 1024`.

```
                    ┌──────────────────────── once per image ─────────────────────────┐
image (PIL/bytes)   │ SiglipImageProcessor  → pixel_values        [M, 3, S, S]         │
                    │ SiglipVisionModel     → hidden_states[l]    [M, G·G, C_v]        │
                    │ reshape                → grid                [M, C_v, G, G]       │
                    │ Pooler (avg k×k | pixel-shuffle k)         [M, N, C_v']         │
                    │ Projector MLP          → image_tokens        [M, N, d]            │
                    │ + modality_emb (1 × d) → image_tokens        [M, N, d]            │
                    └─────────────────────────────────────────────────────────────────┘
                                                        │ gather by row → [B, N, d]
text: question, options, optional text state            ▼
build_sequence(state_ids=[PAD]*N+[SEP]+text) → input_ids [B, L], image_start [B], image_index [B]
tok_embeddings(input_ids)                    → emb       [B, L, d]  (pre-LayerNorm)
emb[b, image_start[b] : image_start[b]+N] = image_tokens[image_index[b]]   (rows with index ≥ 0)
ModernBERT(inputs_embeds=emb, attention_mask)→ h         [B, L, d]  (LN → 28 layers → final norm)
h + type_emb(qtype)                          → h         [B, L, d]
2 × TransformerEncoderLayer (pre-LN, 16 heads, FFN 4096, key-padding mask) → h [B, L, d]
gather h at marker_pos                       → m         [B, K, d]
scorer: LN → Linear d→d → GELU → Linear d→1  → logits    [B, K]   (masked −1e4 beyond K_b)
act head on [h_CLS ; top1, top1−top2, H_norm, k/255] → act_logits [B, 2]
decode: softmax(logits[:K_b] / T(modality, qtype, bucket(K_b))) → answer JSON
```

### 2.2 Vision encoder

| | Option A (v1 default) | Option B (quality) |
|---|---|---|
| Checkpoint | `google/siglip-base-patch16-224` | `google/siglip-so400m-patch14-384` |
| Input S | 224 | 384 |
| Patch grid G×G | 14×14 = 196 | 27×27 = 729 |
| Width C_v | 768 | 1152 |
| Params | ~86M (vision tower only) | ~400M |

- Load only the vision tower (`SiglipVisionModel`), not the text tower.
- **Feature layer.** `last_hidden_state` in HF SigLIP has `post_layernorm` applied (`modeling_siglip.py:612`). The second-to-last layer (`hidden_states[-2]`, pre-post-LN) is the LLaVA convention. Make this a config value `vision_feature_layer ∈ {-1, -2}` and ablate it in Phase 4. Default: **−2**.
- **Preprocessing.** Use `SiglipImageProcessor`: resize to S×S without crop, normalise with mean = std = 0.5. Convert every input to RGB first. For RGBA, composite on white.
- **Precision.** Run the tower in the agent's autocast dtype. Frozen in stages 1 and 3. In stage 2 it is optionally unfrozen for the last 2 blocks, at LR 1e-6.

### 2.3 Pooler and projector

```python
class Pooler(nn.Module):            # config: mode ∈ {"avg", "shuffle"}, k
    # in:  [M, G*G, C_v]   out: [M, N, C_v']
    # avg:     reshape → [M, C_v, G, G] → avg_pool2d(k, stride=k, ceil_mode=True) → [M, C_v, G', G']
    #          C_v' = C_v,     N = G'^2     (A: 7×7=49, B: ceil(27/3)=9 → 81)
    # shuffle: pad G to multiple of k, space-to-depth → [M, C_v·k², G/k, G/k]
    #          C_v' = C_v·k²,  N = (G/k)^2 (A, k=2: 49 tokens × 3072)
    # flatten row-major (top-left → bottom-right)

class Projector(nn.Module):
    # Linear(C_v' → d) → GELU → Linear(d → d)
    # last Linear: weight ~ N(0, 0.02), bias 0 (embedding LN downstream normalises the scale)

modality_emb = nn.Parameter(torch.zeros(1, 1, d))   # added after the projector, before injection
```

| Config | Pooler out | Projector params | Tokens |
|---|---|---|---|
| A + avg 2 | 49 × 768 | 1.84M | 49 |
| A + shuffle 2 | 49 × 3072 | 4.20M | 49 |
| B + avg 3 | 81 × 1152 | 2.23M | 81 |

There is no LayerNorm at the projector output. The embedding LayerNorm inside ModernBERT already normalises every token, text or image (§1 #3). The LN weights stay shared, so image and text sit on the same scale.

### 2.4 Sequence layout

The layout reuses `build_sequence` unchanged:

```
pos: 0     1 … h           h+1   h+2 …                      s-1   s … s+N-1        s+N    s+N+1 …      last
     [CLS] <t> question: … [SEP] [MASK] o0 [MASK] o1 … [MASK] oK [SEP] [PAD]×N (image)  [SEP]  text state … [SEP]
                                  ▲ markers                          ▲ image_start = s
```

- `state_ids = [pad_id]*N + [sep_id] + text_ids`, or `[pad_id]*N` when there is no text.
- **`image_start`** is computed as `len(head_part)`, which is the index just after the options' `[SEP]`. It is derived exactly as `build_sequence` builds it: `ids` before the state is appended. Implement it as `image_start = len(ids) - len(st) - 1` from the returned `ids`, or recompute it. Assert that `ids[image_start : image_start+N]` is all `pad_id`.
- **Why `pad_id` as the placeholder.** The tokenizer never emits it inside text. `tok_embeddings` has `padding_idx=pad_id`, so no gradient reaches that row even before it is overwritten. The value never reaches the encoder because it is overwritten. Positions come from `image_start`, not from searching for `pad_id`, so trailing batch padding can never be mistaken for image tokens.
- **Budget.** `room = max_len − len(head) − 1`. On the English checkpoint `len(head) ≤ head_max_len + 3` (CLS, two SEPs), so `room ≥ 316`.
  - Option A uses 49 + 1 of that and leaves ≥ 266 text tokens.
  - Option B uses 81 + 1 and leaves ≥ 234.
  - If `room < N + 1`, raise `ValueError("image does not fit: reduce options or use a 1024-context checkpoint")`. Never cut the image.
- **Text-state truncation.** Right-truncation happens in `build_sequence`. For list states (conversations), left-truncate `text_ids` ourselves to `room − N − 1` before building, which keeps Laya's "keep the newest turn" semantics.
- **Multiple images (hook only in v1).** `state_ids = img_1 + [SEP] + img_2 + [SEP] + … + text`, with `image_start` becoming a list. v1 rejects more than one image with a clear error.

### 2.5 Injection and forward

```python
class LayaVisionModel(nn.Module):
    def __init__(self, laya: DecisionModel, vision: VisionEncoder, pooler, projector):
        self.laya = laya                 # keeps Laya's state-dict keys under "laya."
        self.vision, self.pooler, self.projector = vision, pooler, projector
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, laya.encoder.config.hidden_size))

    def encode_images(self, pixel_values):           # [M,3,S,S] → [M,N,d]
        return self.projector(self.pooler(self.vision(pixel_values))) + self.modality_emb

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                image_tokens=None, image_index=None, image_start=None, detach_encoder=False):
        emb = self.laya.encoder.embeddings.tok_embeddings(input_ids)          # [B,L,d], pre-LN
        if image_tokens is not None:
            rows = (image_index >= 0).nonzero(as_tuple=True)[0]
            pos = image_start[rows, None] + torch.arange(N, device=emb.device)  # [R,N]
            emb = emb.index_put((rows[:, None], pos), image_tokens[image_index[rows]])  # out-of-place
        h = self.laya.encoder(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        return decision_head(self.laya, h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)
```

- `decision_head(...)` is `DecisionModel.forward` from the line after the encoder call onward (`common.py:315-350`), copied verbatim into `laya_vision/model.py` with a comment naming the Laya version it was copied from. `tests/test_text_parity.py` fails if Laya changes it.
- Use `index_put` out-of-place, not in-place `emb[...] = ...`. This keeps autograd correct when the embedding matrix is frozen but the projector trains.
- `ModernBertModel` is called with `inputs_embeds`, so it runs its own embedding LN and dropout on our tensor. This is identical to the text path for text positions, which is why the no-image output equals stock Laya exactly.

### 2.6 Per-request caching

Laya builds one row per question, so a request with 10 questions has 10 rows sharing one state.

- `encode_images` runs on the **M distinct images** (`M ≤ number of states`), giving `[M, N, d]`.
- Each row carries `image_index` into that tensor, and `-1` for text-only rows.
- Cost per request is 1 SigLIP forward per image + B ModernBERT rows. For 10 questions, the vision cost is amortised 10×.
- Optional cross-request LRU cache keyed by `sha256(image bytes)` → `[N, d]` tensor (`cache_size` config, default 0 = off). It is keyed on bytes, never on a path.

### 2.7 What attends to what

- **Global layers (0, 3, …, 27; 10 of 28) and both head layers.** Every token sees every token, so `[MASK]` markers read the image directly.
- **Local layers.** Image tokens exchange information within ±64 positions. The whole image block (≤ 82 tokens) plus its closest options fits within one or two windows, so image tokens mix with each other in every layer.
- **RoPE is 1D over the flattened grid.** 2D layout survives because SigLIP's own position embeddings are baked into the patch features (ARCH §3.1).

### 2.8 Parameters and compute

| Block | Option A | Option B | Trainable in stage 1 / 2 |
|---|---|---|---|
| SigLIP vision tower | 86M | 400M | no / optional last 2 blocks |
| Pooler + projector + modality emb | 1.8M | 2.2M | yes / yes |
| ModernBERT-large encoder | ~395M | ~395M | no / yes (full or LoRA) |
| Type emb + 2 head layers + scorer + act head | ~26M | ~26M | no / yes |
| **Total** | **~509M** | **~823M** | |

Expected latency overhead is one SigLIP-base forward: 224 px, 196 patches, 12 layers. That should be small next to a 28-layer encoder over ~150–512 tokens. **Measure it in Phase 2; don't assume it.**

### 2.9 Checkpoint format

```
laya-vision-<name>/
  rl_agent_config.json      # stock Laya fields (encoder, head_layers, max_len, head_max_len,
                            #   temperature, temperature_by_options, amp_dtype, act_costs, …)
                            # + "vision": {
                            #     "tower": "google/siglip-base-patch16-224",
                            #     "tower_revision": "<sha>",
                            #     "feature_layer": -2, "pool_mode": "avg", "pool_k": 2, "n_tokens": 49,
                            #     "tower_trained": false,
                            #     "temperature": [..3..], "temperature_by_options": {...},
                            #     "laya_base": "convaiinnovations/laya@<sha>", "laya_version": "0.3.21" }
  model.safetensors         # DecisionModel keys ONLY → loadable by stock laya.Agent (text-only)
  vision.safetensors        # pooler/projector/modality_emb (+ vision tower if tower_trained)
  encoder/  tokenizer/      # as in Laya
```

- Stock `laya.Agent(path)` loads this directory and answers text questions with the text temperatures. That is a free regression check and a deployment escape hatch.
- The vision tower is fetched from the Hub by id and revision unless `tower_trained` is true.
- Save fp16, as the notebook does.

### 2.10 Inference API

```python
import laya_vision
agent = laya_vision.load("path/or/hub-id", device=None)
agent.predict({"image": img, "text": "optional"}, questions)   # img: path | bytes | PIL.Image | np.ndarray
agent.predict("plain text state", questions)                   # identical to laya.Agent
agent.predict_batch([state1, state2, ...], questions)
```

- **State forms.**
  - a dict with key `"image"`, plus optional `"text"`: str/dict/list
  - any other str/dict/list, which is handled exactly as Laya handles it
- A dict without `"image"` is **not** reinterpreted. It is serialised as Laya would. Only the reserved `"image"` key triggers the vision path, and it is documented as reserved.
- **Implementation.** `VisionAgent(laya.Agent)`:
  - `__init__` calls `super().__init__` (tokenizer, config, DecisionModel, temperatures, device/AMP policy), then builds the vision side, loads `vision.safetensors` and wraps the model.
  - It overrides `_encode_state` (image-aware `state_ids`, adds `image_start`/`image_index` to the item dict, which `collate_items` carries into `meta`), `_infer` (encodes images once, passes the extra tensors) and `_decode_answers` (image rows use the vision temperatures).
  - Verify during Phase 2 that `predict_batch`'s internal flow passes these through. If it does not, implement a lean `predict`/`predict_batch` that reuses `_check_question`, `_to_internal`, `collate_items` and `_decode_answers` directly.
- **Unsupported in v1**, with a clear error rather than silent wrong answers when an image is present: `fast=True` (TileLang forward takes `input_ids` only), `compile=True`, ONNX, `predict_long`, `predict_shortlist`. Text-only requests keep all of them.

### 2.11 HTTP

`laya_vision.serve` wraps Laya's `/v1/systemone` contract:
- `state` may be `{"image": "<base64 | data:image/...;base64,...>", "text": ...}`.
- Laya's `MAX_STATE_CHARS = 50000` and `MAX_BODY_BYTES = 2 MiB` (`serve.py:68-69`) would reject most base64 images. Validate the image separately, with its own limits: ≤ 10 MB decoded, ≤ 40 MP, allowed formats JPEG/PNG/WebP/GIF (first frame). Apply Laya's text limits to the `text` part only.
- Decode with Pillow under `Image.MAX_IMAGE_PIXELS` to guard against decompression bombs.

---

## 3. Repository layout

The repo currently holds only `ARCHITECTURE.md` and an empty `package.json`/`package-lock.json`. The implementation is Python. The npm files are unused and can be deleted.

```
pyproject.toml                # deps: laya==0.3.21, torch>=2.1, transformers (pinned, see Phase 0),
                              #       safetensors, pillow, numpy; extras: train (datasets, accelerate),
                              #       serve (fastapi, uvicorn), eval (scikit-learn, pandas)
laya_vision/
  __init__.py                 # load(), VisionAgent
  config.py                   # VisionConfig dataclass ↔ cfg["vision"]
  vision.py                   # VisionEncoder: SigLIP tower, feature layer, grid reshape, preprocessing
  projector.py                # Pooler, Projector
  sequence.py                 # image_state_ids(), build_item_with_image() → ids, markers, image_start
  model.py                    # LayaVisionModel, decision_head() (copied from laya, version-tagged)
  agent.py                    # VisionAgent(laya.Agent)
  checkpoint.py               # save()/load() in the §2.9 format
  serve.py                    # HTTP wrapper
  images.py                   # safe decode: path/bytes/base64/PIL/ndarray → RGB PIL
  data/
    schema.py                 # TrainItem: state(image ref, text), question, target distribution
    caption_align.py          # stage-1 generator
    vision_tasks.py           # stage-2 converters (one function per dataset)
    augment.py                # option shuffle, label renaming, instruction paraphrase, distractor questions
    text_mix.py               # LocalLLaMA/typed-decisions loader (as in the notebook)
  train/
    rlcd.py                   # noise sampling + proper_reward + advantage + CE (ported from notebook)
    loop.py                   # shared train loop: param groups, freezing, AMP, checkpointing, DDP
    stage1.py  stage2.py  calibrate.py
  eval/
    metrics.py                # acc, Brier, ECE@10/15 (laya.common.ece_score), score MAE, latency
    baselines.py              # caption→Laya, SigLIP zero-shot
    run_eval.py
tests/
  test_sequence.py  test_shapes.py  test_text_parity.py  test_cache.py
  test_checkpoint.py  test_agent_api.py  test_images.py  test_serve.py
configs/
  stage1_a.yaml  stage2_a.yaml  calibrate.yaml  eval.yaml
```

---

## 4. Implementation plan

Each phase ends with a check that can be run. Don't start a phase until the previous exit criteria pass.

### Phase 0: Environment and pins (½ day)

1. Create `pyproject.toml` and a venv with Python 3.11. Pin `laya==0.3.21`, the version inspected here. Several reused helpers are underscore-private (`_fix_tokenizer_config`, `_load_tokenizer`), and `decision_head` is copied from this version.
2. Pick and pin one `transformers` version. Re-confirm on that exact version that:
   - `ModernBertModel.forward` accepts `inputs_embeds`
   - `ModernBertEmbeddings` applies `norm` to it
   - the sdpa path doesn't unpad

   All three hold on 5.17. On 4.x, read `modeling_modernbert.py` before choosing it.
3. Download `convaiinnovations/laya` (English) and `google/siglip-base-patch16-224`. Record their revisions.
4. **Exit:** `python -c "import laya, laya_vision"` works, and stock `laya.load()` answers the README example.

### Phase 1: Plumbing (2–3 days) → ARCH milestone 1

1. `vision.py`, `projector.py`: modules as in §2.2–2.3, with random-init projector.
2. `sequence.py`: `image_state_ids(tok, n_img, text, is_list)` and `build_item_with_image(...)` wrapping `laya.common.build_sequence`, returning `image_start`. Budget assertion as in §2.4.
3. `model.py`: `LayaVisionModel` and `decision_head`.
4. Tests:
   - `test_sequence.py`:
     - placeholders sit at `image_start`
     - long text never displaces them
     - list-state text is left-truncated
     - overflow raises
     - markers are unchanged versus text-only `build_sequence` on the same question
   - `test_shapes.py`: for a batch with mixed image and text rows and K from 2 to 20, logits are `[B, Kmax]` and probabilities sum to 1 over valid options.
   - **`test_text_parity.py`**: for 50 varied text requests, `LayaVisionModel` with no images vs. stock `DecisionModel` gives max |Δlogit| ≤ 1e-5 in fp32 on CPU. Also `laya.Agent.predict` vs. `VisionAgent.predict` match on the full answer JSON.
5. **Exit:** all tests pass. A random-projector image request returns valid, if meaningless, probabilities.

### Phase 2: Agent, caching, checkpoint (2–3 days) → ARCH milestones 2 and 6a

1. `images.py`: safe decoding (§2.11), plus `test_images.py` (RGBA, grayscale, CMYK, animated GIF, oversized, corrupt).
2. `agent.py`: `VisionAgent` as in §2.10. Check whether `predict_batch` passes the item-dict extras through, and choose between subclassing and the lean path.
3. Per-request caching (§2.6). `test_cache.py`:
   - outputs are identical with caching on vs. images re-encoded per row
   - a mock counts SigLIP calls, and there is exactly 1 per distinct image
4. `checkpoint.py`, in the §2.9 format. `test_checkpoint.py`:
   - save → load round-trip gives identical outputs
   - **stock `laya.Agent` loads the saved directory** and matches the Phase 1 parity outputs
5. Latency harness: p50/p95 for 1 and 10 questions, with and without image, on CPU and on GPU/MPS if available. Record the baseline numbers in `eval/results/latency_phase2.json`.
6. **Exit:** tests pass, and the latency overhead is recorded.

### Phase 3: Data and baselines (4–6 days) → ARCH milestone 3

1. **Licence table** (`data/LICENSES.md`): dataset, licence, allowed use, and whether it is used in training or eval only. Block any dataset without a clear licence. Check in particular moderation/NSFW data, AVA and CC3M/CC12M (URL-based, so link rot matters).
2. `data/schema.py`: `TrainItem = {image_ref, text|None, question (Laya question dict), target: List[float], split}`. Store as JSONL + image shards (WebDataset tar or HF `datasets`). Resolve images lazily.
3. Converters (`vision_tasks.py`, `caption_align.py`) as in ARCH §5.2/5.3:
   - `choice` with ≤ 20 options; for large label sets, sample 20 including the gold label
   - `noul` balanced 50/50
   - `score` bucketed to 3–5 levels, with a soft target when the source is a rating distribution (AVA)
   - hard negatives for caption matching: nearest-neighbour captions by SigLIP text embedding
4. `augment.py`, mirroring Laya:
   - option order shuffle (target permuted too)
   - label renaming (never yes/no or true/false as `choice` keys)
   - 5–10 instruction templates per task
   - 0–2 random distractor questions per state, which also exercises the multi-row cache
5. Splits: train / calib (held out, ~2k items per question-type × modality) / test. Split **by image**, never by question, so no image leaks across splits.
6. `eval/baselines.py`:
   - **Caption → Laya:** caption with a small open captioner (e.g. Florence-2-base or BLIP-base; pin it), optionally add OCR, then run stock Laya on the text.
   - **SigLIP zero-shot:** cosine similarity between the image and `"a photo of {option description}"`, softmax with SigLIP's logit scale. `choice` only.
7. Eval task set (held out from training). Four image tasks, one per decision family:
   - Oxford Pets (choice)
   - VQAv2 yes/no (noul)
   - KonIQ-10k bucketed (score)
   - a safety `noul` set chosen once licensing is clear, or EuroSAT (choice) as a fallback

   Text regression set: AG News, BoolQ, and the typed-decisions test split.
8. **Exit:** baseline numbers for all 4 tasks + text regression in `eval/results/baselines.json`. These are the numbers to beat.

### Phase 4: Stage 1, alignment (3–5 days incl. runs) → ARCH milestone 4

1. `train/rlcd.py`: port the notebook loss exactly (§1.1 values), with a unit test that reproduces the notebook's loss on a fixed seed and batch.
2. `train/loop.py`:
   - param groups by name prefix
   - freezing via `requires_grad=False`. The encoder is frozen but **not detached**: gradients must flow through it to the projector, so `detach_encoder=False`.
   - `model.laya.encoder.gradient_checkpointing_enable(use_reentrant=False)` and `head_checkpointing=True`
   - AMP: bf16 where supported, else fp16 + GradScaler
   - DDP via `torchrun`
   - rolling `checkpoint_latest` every N steps
3. Stage 1 config (`configs/stage1_a.yaml`):
   - trainable: pooler, projector, modality_emb only
   - data: caption-matching `choice` (4–8 options, ≥ 1 hard negative) + `noul` caption-match, 500k–1M items
   - optimiser: AdamW, LR 1e-3, warmup 3%, cosine, weight decay 0, batch 128–256 rows, 1 epoch
   - loss: RLCD + CE (notebook form)
4. Ablations, each on a 100k subset: `feature_layer` −1 vs. −2, pool `avg` vs. `shuffle`, English vs. `multilingual` base (answers ARCH §9's open question).
5. **Exit:** held-out caption-matching accuracy clearly above chance and above SigLIP zero-shot on the same items. Record the chosen ablation settings in `configs/`.

### Phase 5: Stage 2 and Stage 3, decisions and calibration (5–8 days incl. runs) → ARCH milestone 5

1. Stage 2 config (`configs/stage2_a.yaml`):
   - Initialise from the stage-1 projector.
   - Trainable: projector (LR 1e-4), encoder (LR 2.5e-5), head, scorer and type embedding (LR 1e-4). Tower frozen by default.
   - Data mix: 70–80% image tasks (§ARCH 5.3 table) + 20–30% text-only rows from `LocalLLaMA/typed-decisions`, **built exactly as the notebook builds them**, so text rows are the original training distribution.
   - Schedule: G=4, σ 0.4→0.1, 2–4 epochs, effective batch 64, grad clip 1.0.
   - Variant: LoRA (r=16) on encoder attention and MLP, if full fine-tuning shows text regression > 2 points.
2. Track every eval interval: image accuracy/Brier per task, and text accuracy on the regression set. Early-stop on text regression > 2 points.
3. Stage 3, `train/calibrate.py`:
   - Run the model on the held-out calib split, separately per modality.
   - Fit one temperature per `(qtype, bucket)` by minimising NLL with LBFGS on log T (the notebook's `fit_one_temp`), for buckets with ≥ 100 items. Otherwise fall back to the per-type value.
   - **Clamp to [0.5, 5.0]** before saving.
   - Text temperatures go to the stock fields. Refit them only if the encoder changed; otherwise keep the originals. Image temperatures go to `cfg["vision"]`.
4. **Exit:** the ARCH §6 acceptance criteria.
   - beats caption→Laya on ≥ 3 of 4 image tasks
   - text accuracy within 1–2 points of stock Laya
   - image ECE ≤ 0.10 (on `answer_confidence`)
   - p50 GPU latency for 1 question ≤ 2× text-only Laya

### Phase 6: Serving and packaging (2–3 days) → ARCH milestone 6

1. `serve.py` with the §2.11 limits, plus `test_serve.py`: base64, data URLs, oversize, bad format, text-only parity with `laya.serve`.
2. `laya_vision.load(hub_id)` fetches the checkpoint, the pinned vision tower revision and `vision.safetensors`. Verify digests with the same approach as `laya.revisions.verify_digests`.
3. Model card: intended use, the §9 limitations from ARCH, eval tables vs. baselines, calibration per modality, licences.
4. Optional, v1.1: ONNX export of vision tower + projector as a separate graph producing `[N, d]`. This needs a Laya-side ONNX graph that takes `inputs_embeds`, which is a separate piece of work.

### Rough total

About 4–6 weeks for one engineer, dominated by data preparation and training runs.

---

## 5. Test matrix (always green on `main`)

| Test | Guards |
|---|---|
| `test_text_parity` | No image gives exactly stock Laya logits and answers. Catches drift in the copied `decision_head` after a Laya upgrade. |
| `test_sequence` | Image never truncated; markers unchanged; budget errors are explicit |
| `test_cache` | One vision forward per distinct image; cached = uncached outputs |
| `test_checkpoint` | Round-trip; stock `laya.Agent` still loads the checkpoint |
| `test_shapes` | Mixed image/text batches, K = 2…20, all 3 question types |
| `test_images` | Safe decode across formats and malicious inputs |
| `test_serve` | Wire format, limits, text-only parity with `laya.serve` |
| `test_rlcd` | Ported loss equals notebook loss on a fixed batch |

Mirror Laya's CI gates: `ruff check --select=E9,F63,F7,F82,F401,F811` and `python -m compileall`.

---

## 6. Risks specific to implementation

| Risk | Mitigation |
|---|---|
| Laya upgrade changes `DecisionModel.forward` or `build_sequence` | Pin `laya==0.3.21`; the parity test fails loudly; upgrade deliberately |
| Frozen encoder plus projector-only training is too weak to ground images | Stage-1 exit criterion vs. SigLIP zero-shot; if it fails, allow LoRA on the encoder in stage 1 |
| Text regression in stage 2 | Text mix at 20–30%, early stop, LoRA variant, separate text temperatures |
| Temperatures fitted outside [0.5, 5] get clamped silently at load | Clamp at fit time and log the pre-clamp value |
| Base64 images exceed Laya's HTTP limits, or decompression bombs | Separate image limits and safe decoding (§2.11) |
| URL-based datasets (CC3M) rot | Snapshot to local shards once; record the hit rate |
| `predict_batch` internals don't pass the extra item fields through | Lean `predict` path (§2.10), decided in Phase 2 |

---

## 7. Decisions needed from the owner

1. **Base checkpoint:** English `laya` (512 ctx, 49-token images comfortable) vs. `laya-multilingual` (1024 ctx, room for Option B / less pooling). Default: English, with the Phase 4 ablation deciding.
2. **Compute budget:** stage 2 as full fine-tune vs. LoRA-only. This decides GPU needs (full fine-tune of ~509M with checkpointing: a single 24–48 GB GPU or 2×T4 as in the notebook).
3. **Safety task:** which moderation dataset (licence-dependent), or drop it from v1.
4. **Captioner for the baseline:** which model counts as the fair "caption → Laya" comparison.

---

## 8. Current status and next step (updated 2026-09-28)

### Where we are: code complete, ready to move to the training machine

| Area | State |
|---|---|
| Core model: images, vision, projector, model, checkpoint, VisionAgent | done |
| Data pipeline, including the AG News / BoolQ text-regression sets | done. Nothing downloaded at scale yet. |
| Training: stage 1, stage 2 (full or LoRA), calibrate, configs, `TRAINING.md` | done |
| Eval, baselines, serving | done |

**Integration checks on this machine (Intel Mac, CPU, transformers 4.53):**
- `pytest -q`: 144 passed.
- `pytest -m slow tests/test_real_weights.py`: 3 passed. With the real 421M Laya, text-only answers match stock Laya, stock `laya.Agent` loads our checkpoint, and an image request runs end to end.
- `scripts/smoke_train.sh` (stage 1 → stage 2 → calibrate → predict) ends with SMOKE OK.
- ruff and compileall are clean.

**Settled during integration:** `images.py` accepts BMP for local training data, while the HTTP server accepts only JPEG, PNG, WebP and GIF (plan §2.11).

**Housekeeping:** the `docs/handoff/*.md` notes are all DONE and kept for their gotchas. The repo isn't under git yet, but `.gitignore` is in place. The empty `package.json` and `package-lock.json` are unused.

### Then: on the training machine (Linux + NVIDIA)

Follow `TRAINING.md`. In order:

1. **Setup.** Clone or rsync the repo without `.venv/`, then run `scripts/setup_train_env.sh`: modern torch, transformers 5.x, and a pre-download of Laya and SigLIP.
2. **Verify on real weights.** Run `pytest -q` and `pytest -m slow tests/test_real_weights.py`. These have passed on transformers 4.53; this run checks 5.x, where the `inputs_embeds` path could differ.
3. **Data (Phase 3).**
   - Resolve every `VERIFY` item in `laya_vision/data/LICENSES.md`: dataset ids, column names, licences.
   - Run `python -m laya_vision.data.prepare` for stage 1 (captions) and stage 2 (tasks + text mix).
   - Spot-check about 50 records per task by hand.
4. **Baselines.** Pin the captioner and SigLIP revisions in `configs/eval.yaml`. Record `eval/results/baselines.json` (caption→Laya, SigLIP zero-shot, stock Laya on text) and GPU latency. These are the numbers to beat.
5. **Stage 1 (Phase 4).**
   - Train the projector only.
   - Ablations on a 100k subset: feature layer −1 vs. −2, avg vs. shuffle pooling, English vs. multilingual base.
   - Exit when held-out caption matching beats SigLIP zero-shot.
6. **Stage 2 + calibration (Phase 5).**
   - Full fine-tune, or LoRA if text regresses by more than 2 points.
   - Then `train.calibrate` for the image temperatures.
   - Evaluate with `run_eval --compare` against the §4 Phase 5 acceptance criteria.
7. **Serving (Phase 6).** Model card, `laya_vision.serve` smoke test on GPU, and optionally the cross-request image cache (§2.6, not implemented yet).

### Open decisions (unchanged from §7)

- base checkpoint (English vs. multilingual)
- full fine-tune vs. LoRA budget
- safety dataset choice
- baseline captioner
