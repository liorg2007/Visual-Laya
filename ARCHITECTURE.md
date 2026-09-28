# Laya-Vision — Architecture Spec (Proposal)

**Status:** design proposal, not implemented. Working name "Laya-Vision".
**Goal:** a small, fast, non-autoregressive decision model that accepts an **image** (optionally with text) as the state and answers Jev/Laya-style typed questions (`choice`, `score`, `noul`) with calibrated probabilities in one forward pass.
**Audience:** an engineer or coding agent implementing it on top of the open-source Laya codebase.

Conventions in this doc:
- **[FACT]** = published by Laya / ModernBERT / SigLIP authors.
- **[PROPOSAL]** = a design decision made in this spec; change it if you have a reason.
- **[VERIFY]** = must be checked in the actual source code before implementing.

---

## 1. Background: what Laya is (the part we reuse)

[FACT] Laya (Convai Innovations, Apache 2.0, https://github.com/NandhaKishorM/laya, `pip install laya`) is a 421M-parameter decision model:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
        │
        ▼  token embeddings (50,368 × 1024) + LayerNorm
        ▼  ModernBERT-large encoder: 28 layers, d=1024, 16 heads, GeGLU 2624,
        │  global attention every 3rd layer (0,3,…,27), others local window 128, RoPE
        ▼  + question-type embedding (choice=0, score=1, noul=2)
        ▼  decision head: 2 × TransformerEncoderLayer (d=1024, 16 heads, FFN 4096, pre-LN, dropout 0.1)
        ├─► gather hidden states at [MASK] positions → option scorer MLP
        │     (LN → Linear 1024→1024 → GELU → Linear 1024→1) → 1 logit per option
        │     → softmax(z / T), T fitted per (question type, option count)
        └─► [CLS] + 4 distribution stats → act/escalate head (1028→256→2)
```

Key Laya facts that constrain this design:
- [FACT] English checkpoint `max_len` = 512; `head_max_len` = 192 tokens for question + options, remainder (~320) for the state. `laya-multilingual` and `laya-typed-decisions` use 1024 / 256.
- [FACT] Each question is its own sequence row; all rows of a request are batched into one forward pass. The state is therefore re-encoded once per question.
- [FACT] Training is RLCD: Gaussian noise on logits (G = 8 samples, σ decays 1.0 → 0.3, zero-mean across options), reward = log score + 0.5 × spherical score − 1.0 × RPS (RPS for ordinal `score` questions), GRPO-style group-mean advantage, REINFORCE.
- [FACT] Confidence = 1 − H(p) / log K.
- [FACT] The act/escalate head currently carries no usable signal (issue #185). Keep it, but don't depend on it.
- [FACT] Quality degrades past ~20 options per `choice` question (shared option token budget).

---

## 2. Core idea

Vision-language models feed images to a language model by encoding the image into patch vectors with a vision encoder, projecting them into the LM's embedding space, and inserting them into the token sequence.

Laya's encoder only consumes 1024-dim vectors after its embedding layer, so the same trick applies. **Image patch vectors go into the `<state>` slot of the sequence.** Everything downstream (type embedding, decision head, marker gather, scorer, temperature/softmax, RLCD) is unchanged. The `[MASK]` option markers reach the image tokens through the encoder's global attention layers and the full-attention head layers.

---

## 3. Architecture

```
image ──► [Vision encoder: SigLIP, frozen at first] ──► patch grid [H', W', C_v]
                                                            │
                                                  [Pooler: 2D avg-pool / pixel-shuffle]
                                                            │  N_img tokens × C_v'
                                                  [Projector: MLP C_v' → 1024]
                                                            │  N_img × 1024
                                                  + image-modality embedding (1 × 1024)
                                                            │
question + options (+ optional text) ──► Laya token embeddings ──┐
                                                                  ▼
             [CLS] q [SEP] [MASK] o0 [MASK] o1 … [SEP] [IMG]×N_img [SEP] text… [SEP]
                                                                  ▼
                                  ModernBERT-large encoder (unchanged architecture)
                                                                  ▼
                           type embedding → 2-layer decision head → scorer → softmax(z/T)
                                                                  ▼
                                    same output schema as Laya / Jev
```

### 3.1 Vision encoder — [PROPOSAL]

| Option | Model | Input | Patch grid | Width C_v | Params |
| --- | --- | --- | --- | --- | --- |
| A (start here) | `google/siglip-base-patch16-224` vision tower | 224 px | 14 × 14 = 196 | 768 | ~86M |
| B (higher quality) | `google/siglip-so400m-patch14-384` vision tower | 384 px | 27 × 27 = 729 | 1152 | ~400M |

- Use the **last hidden state** (pre-pooling patch tokens), not the pooled embedding. [VERIFY] which layer (last vs. second-to-last) works best; LLaVA-style models often use the second-to-last.
- SigLIP encodes 2D position internally, so 2D spatial layout survives even though ModernBERT's RoPE is 1D. No extra 2D position scheme is needed for v1.
- Why SigLIP: small, strong, image-text aligned, widely used in open VLMs. Swapping in another ViT should only require changing C_v and the grid size.

### 3.2 Pooler — [PROPOSAL]

This is required by Laya's token budget (~320 state tokens on the English checkpoint).

| Encoder | Pooling | Result |
| --- | --- | --- |
| Option A (14×14) | 2×2 average pool | 7×7 = **49 tokens** |
| Option B (27×27) | 3×3 average pool | 9×9 = **81 tokens** |

Alternative: pixel-shuffle, i.e. concatenate each 2×2 neighbourhood to 4·C_v channels before the projector. This loses less information but gives a larger projector input. Make the pooling factor a config value.

Flatten in row-major order (top-left → bottom-right).

### 3.3 Projector — [PROPOSAL]

```
Linear(C_v' → 1024) → GELU → Linear(1024 → 1024)
```

- Params: ~1.8M (option A, C_v' = 768), ~2.2M (option B, C_v' = 1152), more with pixel-shuffle.
- Follow it with a learned **image-modality embedding** (a single 1024 vector) added to every image token, so the encoder can tell image tokens from text tokens. ~1K params.
- Initialise the last linear layer with small weights so image tokens start near the scale of text embeddings. [VERIFY] Match against the post-LayerNorm scale of Laya's token embeddings; consider a LayerNorm at the projector output.

### 3.4 Sequence builder — [PROPOSAL]

Extend Laya's `build_sequence` (see [VERIFY] list) to produce:

```
[CLS] <type> question: <instructions> [SEP] [MASK] o0 [MASK] o1 … [SEP] <IMG_0> … <IMG_{N-1}> [SEP] <optional text state> [SEP]
```

- Image tokens are **placeholders** in `input_ids`: use the pad or an unused token id, and keep a boolean `image_mask`. At embedding time, overwrite those positions with projector outputs.
- Put the image first in the state region and any text after it, so text truncation never cuts the image.
- Budget, English checkpoint: 512 total − 192 head = 320 state → 49 or 81 image tokens leaves 271 or 239 for text.
- [PROPOSAL] Start from **`laya-typed-decisions` or `laya-multilingual`** (1024 context) if you want more room, or raise `max_len` on the English checkpoint. ModernBERT supports up to 8,192.
- Multiple images: v1 supports **one image per state**. Leave hooks for N images (concatenate their token blocks separated by `[SEP]`).

### 3.5 Injection into the encoder — [VERIFY]

Preferred path: call the ModernBERT backbone with `inputs_embeds` instead of `input_ids`.

```python
emb = encoder.embeddings.tok_embeddings(input_ids)       # [B, L, 1024]
emb[image_mask] = image_tokens.reshape(-1, 1024)         # overwrite placeholders
out = encoder(inputs_embeds=emb, attention_mask=attn)    # [VERIFY] support + whether the
                                                         # embedding LayerNorm is applied
```

[VERIFY] Whether HF `ModernBertModel` accepts `inputs_embeds` in the installed transformers version, and whether it applies the embedding LayerNorm to `inputs_embeds` (it normally does inside `ModernBertEmbeddings`). If it doesn't, apply the norm yourself so image tokens and text tokens get the same treatment. Also check the unpadding / flash-attention path, which can bypass `inputs_embeds`.

### 3.6 Caching across question rows — [PROPOSAL]

Laya builds one row per question. **Run the vision tower, pooler and projector once per image**, then broadcast the resulting `[N_img, 1024]` block into every row for that state. Only the text encoder repeats per question. This keeps multi-question latency close to text-only Laya.

### 3.7 Unchanged components

The following are copied from Laya with no architectural change: the ModernBERT encoder, the type embedding, the 2 decision-head layers, the marker gather, the option scorer, the act head, temperature scaling, and the output schema.

### 3.8 Parameter budget (option A)

| Block | Params |
| --- | --- |
| SigLIP-base vision tower | ~86M |
| Projector + modality embedding | ~1.8M |
| Laya (unchanged) | ~421M |
| **Total** | **~509M** |

Option B totals ~823M.

---

## 4. API — [PROPOSAL]

Keep Laya's request and response shapes; only the state changes.

```python
agent = laya_vision.load("path/or/hub-id")

result = agent.predict(
    {"image": "path/or/bytes/or/PIL.Image", "text": "optional caption, OCR, metadata"},
    {
      "content": {"type": "choice", "instructions": "What is the main subject?",
                  "criteria": {"cat": "a cat", "dog": "a dog", "other": "anything else"}},
      "unsafe":  {"type": "noul",   "instructions": "Does the image contain violence?"},
      "quality": {"type": "score",  "instructions": "How sharp is the photo?",
                  "criteria": ["blurry", "acceptable", "sharp"]},
    },
)
# result["answers"][...] identical in shape to Laya's output
```

HTTP (Jev-compatible `POST /v1/systemone`): accept `state.image` as a base64 string or data URL. Text-only requests must behave exactly as in Laya.

---

## 5. Training plan

### 5.1 Freezing schedule — [PROPOSAL]

| Stage | Vision tower | Projector | Laya encoder | Decision head + scorer | Goal |
| --- | --- | --- | --- | --- | --- |
| 1. Align | frozen | **train** | frozen | frozen | map image features into Laya's embedding space |
| 2. Decide | frozen (optionally last few blocks) | **train** | **train** (full or LoRA) | **train** | learn visual decisions |
| 3. Calibrate | frozen | frozen | frozen | frozen | fit T per (type, option count) on held-out image data |

### 5.2 Stage 1 — alignment through Laya's own interface

No generative captioning objective is needed; use RLCD with auto-generated typed questions from image-caption data (e.g. COCO Captions, CC3M/CC12M subsets):
- `choice`: pick the true caption among 4–8 candidates (true + distractors from other images; include hard negatives with similar captions).
- `noul`: "Does this caption describe the image?" with a balanced 50/50 true/false split.

Suggested starting hyperparameters: projector LR 1e-3, batch 64–256 rows, 1 epoch over ~500k–1M pairs. These are starting guesses to tune, not known-good values.

### 5.3 Stage 2 — decision fine-tuning

Convert labelled vision datasets into typed questions:

| Source (examples) | Question type | Notes |
| --- | --- | --- |
| CIFAR-10, Oxford Pets, Food-101, EuroSAT | `choice` | ≤ 20 options per question; for larger label sets sample 20 including the gold label |
| A-OKVQA, ScienceQA (image subset) | `choice` | multiple-choice VQA |
| VQAv2 yes/no subset | `noul` | |
| Moderation / NSFW datasets | `noul`, `score` | check licences |
| Aesthetic / quality ratings (e.g. AVA, KonIQ) | `score` | bucket to 3–5 ordinal levels |
| Laya's original text data | all | **mix in 20–30% text-only rows** to prevent forgetting |

Augmentations, mirroring Laya's text pipeline: shuffle option order, paraphrase instructions, vary label names (avoid yes/no and true/false as `choice` keys), add random distractor questions.

Objective: unchanged RLCD (G = 8, σ 1.0 → 0.3, reward = log + 0.5·spherical − 1.0·RPS, group-mean advantage). Suggested LR: 1e-5 for the encoder, 1e-4 for the projector and head.

### 5.4 Stage 3 — calibration

Fit one temperature per (question type, option count) on a held-out image split. Report ECE separately for image and text inputs, because calibration does not transfer across modalities automatically.

---

## 6. Evaluation

**Baselines. Implement these first:**
1. **Caption → Laya:** caption (and/or OCR) the image with an off-the-shelf model, pass the text to stock Laya. This is the number to beat.
2. **SigLIP zero-shot:** image–text similarity against option descriptions, softmaxed. It only works for `choice`.

**Metrics:** accuracy, Brier, ECE (10 bins), score MAE for `score`, latency p50/p95 for 1 and 10 questions per image (GPU and CPU), and text-only regression against stock Laya on AG News, BoolQ and the typed-decisions set.

**Acceptance criteria for v1 [PROPOSAL]:**
- Beats baseline 1 on ≥ 3 of 4 held-out image tasks.
- Text-only accuracy within 1–2 points of stock Laya.
- Image ECE ≤ 0.10 after Stage 3.
- p50 latency, 1 question, GPU ≤ ~2× text-only Laya.

---

## 7. Implementation plan

Suggested package layout:

```
laya_vision/
  config.py          # vision model id, pooling factor, projector dims, max_len, head_max_len
  vision.py          # VisionEncoder wrapper (SigLIP), returns patch grid
  projector.py       # Pooler + MLP projector + modality embedding
  sequence.py        # build_sequence_with_image(): placeholders + image_mask
  model.py           # LayaVision(nn.Module): vision → projector → inject → Laya
  agent.py           # predict()/predict_batch() with the same output schema as laya.Agent
  data/
    caption_align.py # Stage-1 question generation from caption datasets
    vision_tasks.py  # Stage-2 dataset → typed-question converters
  train/
    rlcd.py          # reuse/import Laya's RLCD loss; do not reimplement if importable
    stage1.py, stage2.py, calibrate.py
  eval/
    baselines.py, run_eval.py
tests/
  test_shapes.py, test_text_parity.py, test_cache.py
```

Milestones:
1. **Plumbing.** Load Laya + SigLIP; build a sequence with image placeholders; forward pass returns valid probabilities. *Test:* shapes, and with no image the output matches stock Laya to 1e-4.
2. **Cache.** Vision features are computed once per image and shared across question rows. *Test:* identical outputs with and without the cache.
3. **Baselines.** Caption → Laya and SigLIP zero-shot numbers on the eval tasks.
4. **Stage 1.** Projector-only training; report caption-matching accuracy.
5. **Stage 2 + 3.** Full fine-tune, calibrate, evaluate against the acceptance criteria.
6. **Serving.** `predict` API, HTTP `image` field, ONNX export (optional).

---

## 8. [VERIFY] before writing code

1. Where Laya's `build_sequence` lives and how it records `[MASK]` marker positions. Reuse it rather than rewriting it.
2. How Laya's model class calls the ModernBERT backbone, and whether `inputs_embeds` is supported on that path (including the unpadded / flash-attention path).
3. Whether the embedding LayerNorm is applied to `inputs_embeds`.
4. The exact truncation logic for the state region, so image placeholders are never truncated.
5. Whether Laya's RLCD training code is importable from the package or only in the notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`).
6. How `rl_agent_config.json` stores temperatures, so the calibrated checkpoint loads with stock Laya tooling.
7. Licences of every dataset used in Stages 1 and 2.

---

## 9. Risks and open questions

- **Weak vision-language grounding.** ModernBERT was never pretrained on images; Stage 1 does a job VLMs spend billions of tokens on. Expect good perception-style decisions (category, presence, safety) and weaker compositional reasoning (counting, spatial relations, multi-step).
- **Text in images.** SigLIP at 224–384 px is weak at OCR. For screenshots and documents, an OCR → text-state pipeline may beat the native path. Consider a hybrid: image tokens plus OCR text in the same state.
- **Token budget.** 49–81 image tokens is coarse. If fine detail matters, move to a 1024-context checkpoint and use less pooling.
- **1D RoPE over a 2D grid.** Relying on SigLIP's internal positions should be enough for v1; a 2D-aware scheme is a v2 experiment.
- **Local attention.** In local layers, image tokens only see nearby patches in flattened order, so vertically adjacent patches are 7–9 positions apart. This is well inside the 128 window at these sizes, so it's fine.
- **Option markers far from the image.** As in text Laya, markers reach the state via the global layers and the head; this is expected, not a bug.
- **Open question:** Is it better to start from the English checkpoint (stronger English, 512 context) or `laya-multilingual` (1024 context, 2× faster, weaker English)? Try both in milestone 4.

---

## 10. Prior art

- **Jev (TypeSafe AI):** text-only; multimodal requires a separate front-end.
- **Jev-Omni** (https://huggingface.co/akhilaaa3/Jev-Omni): typed-decision interface on Gemma 4 12B IT; text, image, audio and video; ~26 ms per image on H200. It shows the interface works for images, at ~24× the proposed size.
- **PixelJev** (arXiv 2609.29283): image + instruction + runtime candidate set → choice with probabilities, using small open multimodal models; few-shot adaptation raised Pets accuracy from 60% to 92%.
- **Guillaume Laforge's blog** (Sept 2026): re-enabled DiffusionGemma's SigLIP vision tower in a Jev-style pipeline for zero-shot visual classification.
- **LLaVA-style VLMs:** the vision encoder → MLP projector → token injection pattern this spec borrows.

The niche for Laya-Vision is a ~0.5B, encoder-only, CPU- and browser-deployable visual decision model, which none of the above is.

## References

- Laya: https://github.com/NandhaKishorM/laya, https://huggingface.co/convaiinnovations/laya
- Laya architecture write-up: https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me
- ModernBERT-large: https://huggingface.co/answerdotai/ModernBERT-large
- SigLIP: https://huggingface.co/google/siglip-base-patch16-224, https://huggingface.co/google/siglip-so400m-patch14-384
- PixelJev: https://arxiv.org/abs/2609.29283
- Jev-Omni: https://huggingface.co/akhilaaa3/Jev-Omni
