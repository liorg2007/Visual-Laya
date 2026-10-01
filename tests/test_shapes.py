"""Mixed image/text batches, K = 2..20, all question types, both pool modes (plan.md §5)."""
import pytest
import torch

from _core_utils import perturbed_laya
from laya_vision.config import VisionConfig
from laya_vision.projector import Pooler, Projector
from laya_vision.sequence import build_item, collate_vision, to_internal


@pytest.mark.parametrize("mode,k,grid", [("avg", 2, 14), ("avg", 3, 27), ("avg", 2, 5), ("shuffle", 2, 14),
                                         ("shuffle", 2, 5), ("shuffle", 3, 27), ("avg", 1, 4)])
def test_pooler_matches_config(mode, k, grid):
    vcfg = VisionConfig(pool_mode=mode, pool_k=k, image_size=grid * 16, patch_size=16, vision_width=8)
    p = Pooler(mode, k, grid, 8)
    out = p(torch.randn(3, grid * grid, 8))
    assert p.n_tokens == vcfg.n_tokens and p.out_width == vcfg.pooled_width
    assert out.shape == (3, vcfg.n_tokens, vcfg.pooled_width)


def test_pooler_row_major():
    g, c = 4, 1
    x = torch.arange(g * g, dtype=torch.float32).view(1, g * g, c)  # value = row * 4 + col
    avg = Pooler("avg", 2, g, c)(x)[0, :, 0]
    assert avg.tolist() == [2.5, 4.5, 10.5, 12.5]
    sh = Pooler("shuffle", 2, g, c)(x)[0]
    assert sh[0].tolist() == [0, 1, 4, 5] and sh[1].tolist() == [2, 3, 6, 7] and sh[3].tolist() == [10, 11, 14, 15]
    # ceil_mode on a 5x5 grid: the last window averages only the real cells
    x5 = torch.ones(1, 25, 1)
    assert torch.allclose(Pooler("avg", 2, 5, 1)(x5), torch.ones(1, 9, 1))


def test_projector_init():
    p = Projector(12, 64)
    assert p.fc2.bias.abs().sum() == 0
    assert 0.01 < p.fc2.weight.std().item() < 0.03


def _questions(K):
    return {
        "c": to_internal({"type": "choice", "instructions": "pick", "criteria": ["opt%d" % i for i in range(K)]}),
        "s": to_internal({"type": "score", "instructions": "rate", "criteria": ["lvl%d" % i for i in range(K)]}),
        "n": to_internal({"type": "noul", "instructions": "true?"}),
    }


@pytest.fixture(scope="module", params=[("avg", 2), ("shuffle", 2)])
def tiny_model(request, laya_cfg, tiny_laya_dir, tiny_siglip_config):
    from laya_vision.checkpoint import init_from_laya

    mode, k = request.param
    vcfg = VisionConfig(tower="tiny", pool_mode=mode, pool_k=k)
    model, tok, _ = init_from_laya(perturbed_laya(tiny_laya_dir), vcfg, tower_config=tiny_siglip_config)
    return model.eval(), tok, vcfg


@pytest.mark.parametrize("K", [2, 3, 5, 11, 20])
def test_mixed_batch_shapes(tiny_model, K):
    model, tok, vcfg = tiny_model
    states = [{"image": "a", "text": "caption"}, "text only", {"image": "b"}, ["t1", "t2"], {"image": "a"}]
    refs = ["A", None, "B", None, "A"]
    items = []
    for s, r in zip(states, refs):
        for q in _questions(K).values():
            it = build_item(tok, s, q, vcfg.n_tokens)
            if r:
                it["image_ref"] = r
            items.append(it)
    b = collate_vision(items, tok.pad_token_id)
    assert b["image_refs"] == ["A", "B"]
    pv = torch.randn(len(b["image_refs"]), 3, vcfg.image_size, vcfg.image_size)
    with torch.no_grad():
        logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                            image_index=b["image_index"], image_start=b["image_start"], pixel_values=pv)
    B = len(items)
    assert logits.shape == (B, K) and logits.dtype == torch.float32 and act.shape == (B, 2)
    p = torch.softmax(logits, -1)
    assert torch.allclose(p.sum(-1), torch.ones(B))
    assert torch.allclose((p * ~b["marker_mask"]).sum(-1), torch.zeros(B), atol=1e-6)
    assert b["marker_mask"].sum(-1).tolist() == [K, K, 2] * len(states)
    assert torch.isfinite(logits[b["marker_mask"]]).all()


def test_encode_images_shape_and_grad(tiny_model):
    model, tok, vcfg = tiny_model
    pv = torch.randn(2, 3, vcfg.image_size, vcfg.image_size)
    t = model.encode_images(pv)
    assert t.shape == (2, vcfg.n_tokens, model.laya.encoder.config.hidden_size) and model.n_image_tokens == vcfg.n_tokens
    # frozen Laya + trainable projector: the gradient reaches the projector through the encoder
    groups = model.trainable_groups()
    assert set(groups) == {"vision", "projector", "encoder", "head"}
    n_all = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for g in groups.values() for p in g) == n_all
    for g, ps in groups.items():
        for p in ps:
            p.requires_grad_(g == "projector")
    q = to_internal({"type": "noul", "instructions": "x?"})
    it = dict(build_item(tok, {"image": 1}, q, vcfg.n_tokens), image_ref=1)
    b = collate_vision([it], tok.pad_token_id)
    model.train()
    logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                      image_index=b["image_index"], image_start=b["image_start"], pixel_values=pv[:1])
    logits[0, 1].backward()
    model.eval()
    assert model.projector.fc1.weight.grad is not None and model.projector.fc1.weight.grad.abs().sum() > 0
    assert model.modality_emb.grad is not None
    assert model.laya.encoder.embeddings.tok_embeddings.weight.grad is None
    for ps in groups.values():
        for p in ps:
            p.requires_grad_(True)
            p.grad = None


def test_image_changes_output_text_rows_unaffected(tiny_model):
    model, tok, vcfg = tiny_model
    q = to_internal({"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c"]})
    img = dict(build_item(tok, {"image": 1}, q, vcfg.n_tokens), image_ref=1)
    txt = build_item(tok, "hello", q, vcfg.n_tokens)
    b = collate_vision([img, txt], tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    kw = dict(image_index=b["image_index"], image_start=b["image_start"])
    with torch.no_grad():
        tok_a = torch.randn(1, vcfg.n_tokens, model.laya.encoder.config.hidden_size)
        la, _ = model(*args, image_tokens=tok_a, **kw)
        lb, _ = model(*args, image_tokens=tok_a * -3 + 1, **kw)
        ref, _ = model.laya(*args)
    assert not torch.allclose(la[0], lb[0])
    assert torch.allclose(la[1], lb[1], atol=1e-5) and torch.allclose(la[1], ref[1], atol=1e-5)


def test_feature_layer(tiny_siglip_config):
    from laya_vision.vision import VisionEncoder

    torch.manual_seed(0)
    v = VisionEncoder.from_config(tiny_siglip_config, feature_layer=-1).eval()
    pv = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = v.tower(pixel_values=pv, output_hidden_states=True)
        assert torch.equal(v(pv), out.last_hidden_state)
        v.feature_layer = -2
        assert torch.equal(v(pv), out.hidden_states[-2])
    assert (v.grid, v.width, v.image_size, v.patch_size) == (4, 32, 32, 8)


def test_projector_standardize_keeps_images_apart():
    import torch
    from laya_vision.projector import Projector

    torch.manual_seed(0)
    p = Projector(12, 64, in_norm=True, standardize=True)
    with torch.no_grad():  # collapse: a huge shared output direction, as stage-1 training produced
        p.fc2.bias.fill_(50.0)
    x = torch.randn(8, 5, 12)
    p.train()
    for _ in range(300):
        p(x)
    p.eval()
    y = p(x).mean(1)
    cos = torch.nn.functional.cosine_similarity(y[:, None], y[None], dim=-1)
    assert cos[~torch.eye(8, dtype=torch.bool)].mean() < 0.5
    assert abs(p(x).norm(dim=-1).mean().item() - 2.0) < 1.0
    assert {"std.running_mean", "std.running_var", "in_norm.weight"} <= set(p.state_dict())
    assert set(Projector(12, 64).state_dict()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
