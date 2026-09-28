"""Checks against the real Laya (421M) and SigLIP-base weights. Run on the training machine:

    python -m pytest -m slow tests/test_real_weights.py

The fast suite proves text-only parity on tiny models under the local transformers version; this
proves it on the real checkpoint under whatever transformers the training box has (5.x), which is
where the ``inputs_embeds`` path could differ. Downloads ~2.5 GB on first run.
"""
import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.slow

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is this about?",
              "criteria": {"sports": "sports", "tech": "technology", "food": "food and cooking"}},
    "urgent": {"type": "noul", "instructions": "Does this need an urgent reply?"},
    "tone": {"type": "score", "instructions": "How positive is the tone?",
             "criteria": ["negative", "neutral", "positive"]},
}
STATES = [
    "The new GPU doubles training throughput compared to last year's model.",
    {"from": "customer", "body": "My order never arrived and I need it by tomorrow!"},
    [{"role": "user", "content": "any good pasta recipes?"}, {"role": "assistant", "content": "sure"}],
]


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    from laya_vision.checkpoint import init_from_laya, save_checkpoint
    from laya_vision.config import VisionConfig

    vcfg = VisionConfig()
    model, tok, cfg = init_from_laya("convaiinnovations/laya", vcfg)
    out = str(tmp_path_factory.mktemp("real_ckpt"))
    save_checkpoint(model, tok, cfg, vcfg, out)
    return out


def _probs(ans):
    return np.array([v for _, v in sorted(ans.get("probabilities", {"p": ans.get("noul")}).items())])


def test_text_parity_with_stock_laya(ckpt_dir):
    import laya
    from laya_vision import load

    stock = laya.Agent("convaiinnovations/laya", device="cpu")
    ours = load(ckpt_dir, device="cpu")
    for state in STATES:
        a = stock.predict(state, QUESTIONS)["answers"]
        b = ours.predict(state, QUESTIONS)["answers"]
        for qid in QUESTIONS:
            # save_checkpoint stores fp16 weights, so allow fp16 rounding against the Hub weights.
            np.testing.assert_allclose(_probs(a[qid]), _probs(b[qid]), atol=5e-3, err_msg=qid)


def test_stock_laya_loads_our_checkpoint(ckpt_dir):
    import laya

    out = laya.Agent(ckpt_dir, device="cpu").predict(STATES[0], QUESTIONS)
    assert set(out["answers"]) == set(QUESTIONS)


def test_image_request_runs(ckpt_dir):
    from laya_vision import load

    agent = load(ckpt_dir, device="cpu")
    img = Image.fromarray((np.random.RandomState(0).rand(300, 400, 3) * 255).astype("uint8"))
    out = agent.predict({"image": img, "text": "photo from the kitchen"}, QUESTIONS)["answers"]
    for qid, ans in out.items():
        if "probabilities" in ans:
            assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-3, qid
