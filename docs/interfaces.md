# laya_vision — module interfaces (contract between packages)

The design is in `plan.md` §2. This file pins the Python interfaces that `model`, `data`, `train`,
`eval` and `serve` code rely on. Change an interface only together with every caller.

## Already implemented

- `laya_vision/config.py`: `VisionConfig` (dataclass; `.n_tokens`, `.grid`, `.pooled_width`, `.to_dict()`, `.from_dict()`)
- `laya_vision/sequence.py`
  - `is_image_state(state)`, `split_state(state) -> (image, text)`; an image state is `{"image": ..., "text": optional}`
  - `to_internal(public_qdef, qid) -> {"t","ins","crit"[,"labels"]}` (validates with Laya's own checks)
  - `build_item(tok, state, q_internal, n_img, max_len, head_max_len, option_order=None, state_ids=None) -> {"ids","markers","qtype","options","image_start"}` (`image_start=-1` for text rows; text rows are byte-identical to Laya's)
  - `collate_vision(items, pad_id) -> batch`: `laya.common.collate_items` output (`input_ids, attention_mask, marker_pos, marker_mask, qtype, label, meta[, target]`) **plus** `image_refs` (unique refs, first-seen order), `image_index [B]`, `image_start [B]` (-1 = no image). Image rows must carry `item["image_ref"]` (hashable).
- `laya_vision/testing.py`: `make_tiny_laya_dir(dir)`, `tiny_siglip_config()`, `tiny_encoder_config()`, `laya_tokenizer_dir()`
- `tests/conftest.py`: fixtures `tiny_laya_dir`, `laya_cfg`, `tok`, `tiny_siglip_config`

## Core model (`laya_vision/{images,vision,projector,model,checkpoint,agent}.py`)

```python
# images.py
load_image(x) -> PIL.Image (RGB)       # x: path | bytes | base64 str | data URL | PIL.Image | np.ndarray
                                        # safe: size/pixel limits, RGBA composited on white, first GIF frame
image_sha256(x) -> str                  # stable hash of the decoded source bytes (or of PIL pixels)
preprocess(images: list[PIL.Image], image_size: int) -> FloatTensor [M,3,S,S]
                                        # == SiglipImageProcessor: bicubic resize to SxS, /255, (x-0.5)/0.5

# vision.py
class VisionEncoder(nn.Module):
    from_pretrained(name, revision=None, feature_layer=-2) / from_config(siglip_vision_config, feature_layer=-2)
    forward(pixel_values) -> [M, G*G, C_v]
    image_size, patch_size, width, grid   # attributes

# projector.py
class Pooler(nn.Module):   Pooler(mode, k, grid, width); forward [M,G*G,C] -> [M,N,C']; .n_tokens, .out_width
class Projector(nn.Module): Projector(in_width, d); Linear -> GELU -> Linear

# model.py
class LayaVisionModel(nn.Module):
    # submodules / state-dict prefixes: laya.* (a laya.common.DecisionModel, keys unchanged),
    #   vision.*, pooler.*, projector.*, modality_emb
    n_image_tokens: int
    encode_images(pixel_values) -> [M, N, d]
    forward(input_ids, attention_mask, marker_pos, marker_mask, qtype,
            image_tokens=None, image_index=None, image_start=None, pixel_values=None,
            detach_encoder=False) -> (logits [B,K] float32, act_logits [B,2])
    # pixel_values given and image_tokens None -> encodes them. No images -> identical to DecisionModel.
    trainable_groups() -> dict[str, list[Parameter]]   # keys: "vision", "projector" (pooler+projector+modality_emb),
                                                       #       "encoder" (laya.encoder), "head" (rest of laya)
decision_head(laya_model, h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False)

# checkpoint.py
init_from_laya(laya_id_or_dir, vcfg: VisionConfig, tower_config=None, revision=None)
    -> (model: LayaVisionModel, tok, cfg: dict)       # tower_config: SiglipVisionConfig for tiny/random tower
save_checkpoint(model, tok, cfg, vcfg, out_dir, dtype=torch.float16)   # plan.md §2.9 layout; model.safetensors = laya.*
                                                      # keys without prefix; dtype=torch.float32 for full precision
load_checkpoint(path_or_hub_id, device="cpu", revision=None) -> (model, tok, cfg, vcfg)

# agent.py
class VisionAgent(laya.Agent): predict(state, questions, ...), predict_batch(states, questions, ...)
load(path_or_hub_id, device=None, **kw) -> VisionAgent          # re-exported as laya_vision.load
```

Core-model additions (implemented; additive, no signature above changed):

- `images.py`: `load_image_and_hash(x) -> (PIL, sha256)` decodes once; `ImageError(ValueError)` for every
  rejected input; limits `MAX_IMAGE_BYTES` (10 MB encoded), `MAX_PIXELS` (40 MP), `ALLOWED_FORMATS`
  (JPEG/PNG/WEBP/GIF/BMP). sha256 is of the encoded bytes for path/bytes/base64/data URL (same image, same
  hash whichever way it arrives) and of the RGB pixels for PIL/ndarray inputs.
- `model.py`: `build_vision_side(vision, vcfg, d) -> (pooler, projector)` also fills vcfg.image_size/patch_size/vision_width.
- `checkpoint.py`: `resolve_dir(path_or_hub_id, revision=None)`, `build_vision_from_dir(dir, vcfg, decision_model, device)`.
  **Tower rule:** the tower is stored (`vision.*` in vision.safetensors + `vision_tower/config.json`) when
  `vcfg.tower_trained` is true or it was built from a config (`init_from_laya(..., tower_config=...)`); otherwise
  it is re-fetched from `vcfg.tower@vcfg.tower_revision`. On load, `vision_tower/config.json` decides.
  `load_checkpoint` returns fp32 weights in eval mode; `init_from_laya` sets `vcfg.laya_base/laya_version`.
- `VisionAgent`: `self.vlm` (LayaVisionModel sharing `self.model`'s weights), `self.vcfg`, `self.n_image_tokens`,
  `self.model_dir`. `self.model` stays the plain DecisionModel, so every stock text path is untouched.
  Image rows are rejected (ValueError) under fast=True / compile=True / predict_long.
- `textnorm.py`: `check_text` rejects lone UTF-16 surrogates in states and questions (ValueError, so the server
  answers 422) on both paths. On image rows only, `unshout_question` / `unshout_state` lower-case all-caps
  strings (no lower-case letters, a word of >= 4 letters) before tokenizing; answers keep the caller's labels.
  Text-only rows are not rewritten, so they stay identical to `laya.Agent`.
- `images.py` also rejects zero-size images (0xN / Nx0 arrays or PIL images).


## Data (`laya_vision/data/`)

Training records are JSONL, one row = one (state, question) pair (`data/schema.py`, `validate_record`):

```json
{"id": "oxford_pets/Abyssinian_1/q0", "task": "oxford_pets", "split": "train|calib|test",
 "image": "oxford_pets/Abyssinian_1.jpg",   // relative to image_root, or null for text-only rows
 "text": null,                              // optional text state (str | dict | list)
 "question": {"type": "choice", "instructions": "...", "criteria": {...}},   // public Laya question dict
 "target": [0.0, 1.0, 0.0],                 // distribution in option order (noul: [false, true])
 "template": "oxford_pets",                 // optional: augment.TEMPLATES key for paraphrases
 "fields": {"q": "..."}}                    // optional: format fields for those templates
```

Target length == `len(render_options(to_internal(question)))`, entries >= 0, sum 1 (1e-4). Splits are
hashed from the image id (`schema.split_for`), never from the question. Files: `data/{task}.{split}.jsonl`.

```python
# data/dataset.py
class DecisionDataset(torch.utils.data.Dataset):
    DecisionDataset(jsonl_paths, image_root, tok, max_len, head_max_len, n_img, augment=None, seed=0,
                    prescan=False, records=None)
    __getitem__(i) -> item dict from sequence.build_item plus "target", "label", "image_ref"
                     (abs image path or None), "task", "id"  -- or None when the record does not fit
    set_epoch(e)       # augmentation rng = Random(f"{seed}:{epoch}:{i}")
    # Skip policy: build_item ValueError -> __getitem__ returns None, collate drops it (deterministic).
    # prescan=True removes non-fitting records at construction (exact len(); use for calib/eval).
make_collate(pad_id, image_size, image_cache=None, loader=None) -> VisionCollate (picklable) fn(items) -> batch | None
    # drops None items; returns None if nothing is left
    # collate_vision(...) incl. batch["target"] [B,Kmax], batch["label"] [B]
    # + batch["pixel_values"] [M,3,S,S] aligned with batch["image_refs"] (None if no images);
    #   each unique image loaded/preprocessed once per batch; image_cache e.g. LRUCache(512)
MixtureSampler(sizes: {name: n}, ratios: {name: r}, num_samples=None, seed=0, rank=0, world_size=1,
               drop_last=False)   # indices into ConcatDataset in `sizes` order; set_epoch(e); DDP-sharded
build_mixture({name: {"files": [...], "ratio": r}}, image_root, tok, max_len, head_max_len, n_img,
              augment=None, num_samples=None, seed=0, rank=0, world_size=1) -> (MixtureDataset, MixtureSampler)
# data/augment.py
Augmenter(p_shuffle=1.0, p_rename=0.3, p_paraphrase=0.5, templates=None, augment_text=False)(record, rng) -> record
# data/text_mix.py: convert_row(row), load_text_mix(hf_split="train", ...)   (notebook targets)
# data/prepare.py:  python -m laya_vision.data.prepare --config configs/data_stage{1,2}.yaml
```

## Train (`laya_vision/train/`)

```python
rlcd_loss(logits, target, marker_mask, qtype, sigma, group_size=4, w_sph=0.75, w_rps=1.0, ce_weight=1.0)
    -> (loss, stats: dict)          # exact port of the notebook loss (plan.md §1.1)
python -m laya_vision.train.stage1 --config configs/stage1_a.yaml
python -m laya_vision.train.stage2 --config configs/stage2_a.yaml
python -m laya_vision.train.calibrate --checkpoint DIR --data calib.jsonl ...
```
