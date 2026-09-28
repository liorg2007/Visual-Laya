"""Save/load round trip; stock laya.Agent still loads the checkpoint (plan.md §2.9)."""
import json
import os

import pytest
import torch
from safetensors.torch import load_file

from _core_utils import QUESTIONS, noise_image, perturbed_laya, vision_ckpt
from laya_vision.checkpoint import load_checkpoint, save_checkpoint
from laya_vision.sequence import build_item, collate_vision, to_internal


def _batch(tok, n):
    items = []
    for s, ref in [({"image": 0, "text": "caption"}, "a"), ("text", None), ({"image": 1}, "b")]:
        for qid, qd in QUESTIONS.items():
            it = build_item(tok, s, to_internal(qd, qid), n)
            if ref:
                it["image_ref"] = ref
            items.append(it)
    return collate_vision(items, tok.pad_token_id)


def _run(model, b, pv):
    with torch.no_grad():
        return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                     image_index=b["image_index"], image_start=b["image_start"], pixel_values=pv)


def test_layout(tiny_laya_dir):
    d = vision_ckpt(tiny_laya_dir)
    assert {"rl_agent_config.json", "model.safetensors", "vision.safetensors", "encoder", "tokenizer",
            "vision_tower"} <= set(os.listdir(d))
    keys = set(load_file(os.path.join(d, "model.safetensors")))
    assert not any(k.startswith(("laya.", "vision.", "projector.")) for k in keys)
    assert any(k.startswith("encoder.") for k in keys) and "scorer.1.weight" in keys
    vkeys = set(load_file(os.path.join(d, "vision.safetensors")))
    assert "modality_emb" in vkeys and any(k.startswith("projector.") for k in vkeys)
    assert any(k.startswith("vision.") for k in vkeys)  # random tower -> saved
    with open(os.path.join(d, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    v = cfg["vision"]
    assert v["n_tokens"] == 4 and v["image_size"] == 32 and v["patch_size"] == 8 and v["vision_width"] == 32
    assert v["laya_version"] == "0.3.21"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_round_trip(tiny_laya_dir, tmp_path, dtype):
    model, tok, cfg, vcfg = load_checkpoint(vision_ckpt(tiny_laya_dir))
    out = save_checkpoint(model, tok, cfg, vcfg, str(tmp_path / "rt"), dtype=dtype)
    if dtype == torch.float16:
        assert load_file(os.path.join(out, "model.safetensors"))["scorer.1.weight"].dtype == torch.float16
    model2, _, cfg2, vcfg2 = load_checkpoint(out)
    assert vcfg2 == vcfg and cfg2["max_len"] == cfg["max_len"]
    from laya_vision.images import preprocess

    b = _batch(tok, vcfg.n_tokens)
    pv = preprocess([noise_image(0), noise_image(1)], vcfg.image_size)
    if dtype == torch.float16:
        # the reference is the in-memory model with its weights rounded to fp16
        with torch.no_grad():
            for t in model.state_dict().values():  # saved tensors only (not RoPE inv_freq)
                if t.is_floating_point():
                    t.copy_(t.half().float())
    l1, a1 = _run(model, b, pv)
    l2, a2 = _run(model2, b, pv)
    assert (l1 - l2)[b["marker_mask"]].abs().max().item() <= 1e-5
    assert (a1 - a2).abs().max().item() <= 1e-5
    if dtype == torch.float32:
        # the images really matter to the outputs (so this comparison is meaningful)
        l3, _ = _run(model, b, pv.flip(0))
        assert (l1 - l3)[b["marker_mask"]].abs().max().item() > 1e-3


def test_hub_tower_not_saved(tiny_laya_dir, tmp_path, monkeypatch):
    """A tower loaded from the Hub (not trained) is referenced by id, not stored."""
    model, tok, cfg, vcfg = load_checkpoint(vision_ckpt(tiny_laya_dir))
    model.vision._laya_vision_saved_tower = False
    out = save_checkpoint(model, tok, cfg, vcfg, str(tmp_path / "hub"))
    assert not os.path.exists(os.path.join(out, "vision_tower"))
    assert not any(k.startswith("vision.") for k in load_file(os.path.join(out, "vision.safetensors")))
    # loading it asks for the pinned tower from the Hub
    import laya_vision.checkpoint as ck

    seen = {}

    def fake(name, revision=None, feature_layer=-2, **kw):
        seen.update(name=name, revision=revision)
        return model.vision

    monkeypatch.setattr(ck.VisionEncoder, "from_pretrained", staticmethod(fake))
    load_checkpoint(out)
    assert seen == {"name": vcfg.tower, "revision": vcfg.tower_revision}
    # tower_trained forces saving it
    vcfg.tower_trained = True
    out2 = save_checkpoint(model, tok, cfg, vcfg, str(tmp_path / "trained"))
    assert os.path.exists(os.path.join(out2, "vision_tower", "config.json"))


def test_stock_laya_agent_loads_checkpoint(tiny_laya_dir):
    import laya

    ref = laya.Agent(perturbed_laya(tiny_laya_dir), device="cpu")
    stock = laya.Agent(vision_ckpt(tiny_laya_dir), device="cpu")
    for s in ["refund now", {"a": 1}, ["x", "y"]]:
        assert stock.predict(s, QUESTIONS) == ref.predict(s, QUESTIONS)


def test_text_only_checkpoint_rejected(tiny_laya_dir):
    with pytest.raises(ValueError, match="text-only"):
        load_checkpoint(tiny_laya_dir)
