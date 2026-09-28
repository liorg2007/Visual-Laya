"""No image -> exactly stock Laya (plan.md §5). Catches drift in the copied decision_head."""
import copy

import pytest
import torch

from _core_utils import QUESTIONS, perturbed_laya, vision_ckpt

STATES = [
    "I want my money back for order 1234, this is ridiculous!",
    "Please cancel my subscription.",
    "hello",
    "",
    "The weather is nice today and I have no complaints at all. " * 40,  # truncated to max_len
    {"ticket": 42, "body": "refund please", "tags": ["billing", "urgent"]},
    {"image_url": "not an image state, only the reserved 'image' key is"},
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "how can I help?"},
     {"role": "user", "content": "cancel it now " * 80}],
    ["turn one", "turn two", "turn three"],
    "Ünïcödé text — ñ, 中文, emoji 🙂 [MASK] inside",
]

QSETS = [
    QUESTIONS,
    {"yn": {"type": "noul", "instructions": "Does the text mention money?",
            "criteria": {"false": "no money", "true": "money mentioned"}, "labels": {"false": "B", "true": "A"}}},
    {"many": {"type": "choice", "instructions": "Pick a topic",
              "criteria": ["t%d" % i for i in range(12)]}},
    {"s5": {"type": "score", "instructions": {"rate": "sentiment"}, "criteria": ["a", "b", "c", "d", "e"]}},
]


def _batch(tok, states, q):
    from laya_vision.sequence import build_item, collate_vision, to_internal

    items = []
    for s in states:
        for qid, qd in q.items():
            items.append(build_item(tok, s, to_internal(qd, qid), n_img=4))
    return collate_vision(items, tok.pad_token_id)


def test_model_logits_match_decision_model(tiny_laya_dir, tok):
    from laya_vision.checkpoint import load_checkpoint

    model, tok2, _, _ = load_checkpoint(vision_ckpt(tiny_laya_dir))
    ref = copy.deepcopy(model.laya).eval()
    total_rows = 0
    for q in QSETS:
        b = _batch(tok2, STATES, q)
        args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        with torch.no_grad():
            lo, act = model(*args)
            lo_ref, act_ref = ref(*args)
        valid = b["marker_mask"]
        assert (lo - lo_ref)[valid].abs().max().item() <= 1e-5
        assert (act - act_ref).abs().max().item() <= 1e-5
        total_rows += lo.shape[0]
        # the perturbed weights make options distinguishable, so this comparison can fail
        assert lo[valid].std() > 1e-3
    assert total_rows >= 50


def test_decision_head_detects_drift(tiny_laya_dir, tok):
    """Sanity: the comparison above is sensitive (a tiny head change is caught)."""
    from laya_vision.checkpoint import load_checkpoint

    model, _, _, _ = load_checkpoint(vision_ckpt(tiny_laya_dir))
    ref = copy.deepcopy(model.laya).eval()
    with torch.no_grad():
        ref.type_emb.weight.mul_(1.01)
    b = _batch(tok, STATES[:3], QUESTIONS)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    with torch.no_grad():
        d = (model(*args)[0] - ref(*args)[0])[b["marker_mask"]].abs().max().item()
    assert d > 1e-5


@pytest.fixture(scope="module")
def agents(tiny_laya_dir):
    import laya
    from laya_vision import load

    return laya.Agent(perturbed_laya(tiny_laya_dir), device="cpu"), load(vision_ckpt(tiny_laya_dir), device="cpu")


def test_agent_answers_identical(agents):
    stock, vis = agents
    n = 0
    for q in QSETS:
        for s in STATES:
            assert vis.predict(s, q) == stock.predict(s, q)
            n += 1
        assert vis.predict_batch(STATES, q, batch_size=3) == stock.predict_batch(STATES, q, batch_size=3)
    assert n >= 20
    # the answers actually vary, so equality is meaningful
    probs = {str(stock.predict(s, QUESTIONS)["answers"]["intent"]["probabilities"]) for s in STATES}
    assert len(probs) > 3


def test_agent_text_kwargs_identical(agents):
    stock, vis = agents
    kw = dict(max_len=64, head_max_len=48, min_confidence=0.9)
    assert vis.predict(STATES[0], QUESTIONS, **kw) == stock.predict(STATES[0], QUESTIONS, **kw)
    assert vis.predict(STATES[0], {}) == stock.predict(STATES[0], {})
