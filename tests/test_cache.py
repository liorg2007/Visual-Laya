"""One vision forward per distinct image per request; cached == per-row (plan.md §2.6)."""
import pytest
import torch

from _core_utils import QUESTIONS, noise_image, png_bytes, vision_ckpt


@pytest.fixture(scope="module")
def agent(tiny_laya_dir):
    from laya_vision import load

    return load(vision_ckpt(tiny_laya_dir), device="cpu")


class _Counter:
    def __init__(self, module):
        self.module, self.calls, self.images = module, 0, 0
        self.orig = module.forward

    def __enter__(self):
        def fwd(pixel_values):
            self.calls += 1
            self.images += pixel_values.shape[0]
            return self.orig(pixel_values)
        self.module.forward = fwd
        return self

    def __exit__(self, *a):
        del self.module.forward


def test_one_vision_forward_per_distinct_image(agent):
    a, b = png_bytes((250, 0, 0)), noise_image(3)
    states = [{"image": a, "text": "x"}, "text only", {"image": b}, {"image": a}, {"image": bytes(a)}]
    with _Counter(agent.vlm.vision) as c:
        agent.predict_batch(states, QUESTIONS)
    assert c.calls == 1 and c.images == 2  # 2 distinct images, 4 image states, 12 image rows
    with _Counter(agent.vlm.vision) as c:
        agent.predict({"image": a}, QUESTIONS)
    assert c.calls == 1 and c.images == 1
    with _Counter(agent.vlm.vision) as c:
        agent.predict_batch(["t1", {"k": "v"}], QUESTIONS)
    assert c.calls == 0


def test_batched_equals_per_row(agent):
    """Encoding each distinct image once and gathering == encoding it separately for every row."""
    from laya_vision.images import preprocess
    from laya_vision.sequence import build_item, collate_vision, to_internal

    imgs = [noise_image(5), noise_image(6)]
    tok, n = agent.tok, agent.n_image_tokens
    items = []
    for i, img in enumerate([imgs[0], imgs[1], imgs[0]]):
        for qid, qd in QUESTIONS.items():
            items.append(dict(build_item(tok, {"image": img, "text": "t%d" % i}, to_internal(qd, qid), n),
                              image_ref=id(img)))
    b = collate_vision(items, tok.pad_token_id)
    pv = preprocess(imgs, agent.vcfg.image_size)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    with torch.no_grad():
        cached, act_c = agent.vlm(*args, image_index=b["image_index"], image_start=b["image_start"], pixel_values=pv)
        per_row_pv = pv[b["image_index"]]
        per_row, act_r = agent.vlm(*args, image_index=torch.arange(len(items)), image_start=b["image_start"],
                                   pixel_values=per_row_pv)
    # only float noise from different vision batch sizes (logits here reach |z| ~ 15)
    m = b["marker_mask"]
    assert torch.allclose(cached[m], per_row[m], rtol=1e-4, atol=1e-4)
    assert torch.allclose(act_c, act_r, rtol=1e-4, atol=1e-4)


def test_agent_batch_equals_single(agent):
    states = [{"image": noise_image(1), "text": "a"}, "plain text", {"image": noise_image(2)},
              {"image": noise_image(1), "text": "b"}]
    batch = agent.predict_batch(states, QUESTIONS)
    single = [agent.predict(s, QUESTIONS) for s in states]
    for x, y in zip(batch, single):
        for qid in QUESTIONS:
            for key in ("probabilities", "noul"):
                if key in x["answers"][qid]:
                    xv, yv = x["answers"][qid][key], y["answers"][qid][key]
                    if isinstance(xv, dict):
                        assert all(abs(xv[k] - yv[k]) <= 2e-4 for k in xv)
                    else:
                        assert abs(xv - yv) <= 2e-4
