"""VisionAgent public API: image input forms, image+text, temperatures, errors (plan.md §2.10)."""
import base64
import json
import os
import shutil

import numpy as np
import pytest

from _core_utils import QUESTIONS, noise_image, png_bytes, vision_ckpt


@pytest.fixture(scope="module")
def agent(tiny_laya_dir):
    import laya_vision

    return laya_vision.load(vision_ckpt(tiny_laya_dir), device="cpu")


def _probs(res):
    return {qid: a.get("probabilities", a.get("noul")) for qid, a in res["answers"].items()}


def test_exports():
    import laya_vision

    assert {"load", "VisionAgent", "VisionConfig", "LayaVisionModel"} <= set(dir(laya_vision)) | set(laya_vision.__all__)
    assert laya_vision.VisionConfig().n_tokens == 49


def test_image_input_forms_agree(agent, tmp_path):
    data = png_bytes((30, 140, 220), size=(40, 30))
    p = tmp_path / "img.png"
    p.write_bytes(data)
    b64 = base64.b64encode(data).decode()
    from PIL import Image
    import io

    forms = [data, str(p), p, b64, "data:image/png;base64," + b64, Image.open(io.BytesIO(data)),
             np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))]
    results = [agent.predict({"image": f}, QUESTIONS) for f in forms]
    for r in results:
        assert r == results[0]
    r = results[0]
    assert r["model"] == "laya-rl-agent"
    assert set(r["answers"]) == set(QUESTIONS)
    a = r["answers"]
    assert a["intent"]["choice"] in ("refund", "cancel", "other")
    assert abs(sum(a["intent"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["urgency"]["score"] <= 2 and 0 <= a["angry"]["noul"] <= 1
    for ans in a.values():
        assert 0 <= ans["answer_confidence"] <= 1 and "act_probability" in ans["action"]
    assert r["usage"]["input_tokens"] > agent.n_image_tokens


def test_image_matters_and_text_matters(agent):
    img_a, img_b = noise_image(10), png_bytes((0, 0, 0))
    ra = _probs(agent.predict({"image": img_a}, QUESTIONS))
    rb = _probs(agent.predict({"image": img_b}, QUESTIONS))
    rt = _probs(agent.predict({"image": img_a, "text": "I demand a refund immediately"}, QUESTIONS))
    assert ra != rb and ra != rt
    # an image state is not the same as its text alone
    assert _probs(agent.predict("I demand a refund immediately", QUESTIONS)) != rt


def test_text_state_types_with_image(agent):
    for text in ["str", {"k": [1, 2]}, ["turn 1", "turn 2"], "", None]:
        r = agent.predict({"image": noise_image(0), "text": text}, QUESTIONS)
        assert set(r["answers"]) == set(QUESTIONS)
    assert (agent.predict({"image": noise_image(0), "text": None}, QUESTIONS)
            == agent.predict({"image": noise_image(0)}, QUESTIONS))


def test_predict_batch_mixed_order(agent):
    states = ["text a", {"image": noise_image(1)}, {"image": noise_image(2), "text": "b"}, ["c"]]
    batch = agent.predict_batch(states, QUESTIONS, batch_size=2)
    assert len(batch) == 4
    assert batch[0] == agent.predict("text a", QUESTIONS)


def test_hooks_see_image_states(agent):
    seen = []
    agent.predict({"image": noise_image(0)}, QUESTIONS, on_predict_start=lambda ctx: seen.append(len(ctx.states)))
    assert seen == [1]


def test_image_temperatures(tiny_laya_dir):
    """Image rows use cfg["vision"] temperatures; text rows keep the stock ones."""
    import laya_vision

    base = laya_vision.load(vision_ckpt(tiny_laya_dir), device="cpu")
    hot = laya_vision.load(vision_ckpt(tiny_laya_dir, temps=([5.0, 5.0, 5.0], {})), device="cpu")
    img = {"image": noise_image(4)}
    pb, ph = base.predict(img, QUESTIONS)["answers"], hot.predict(img, QUESTIONS)["answers"]
    assert ph["intent"]["answer_confidence"] < pb["intent"]["answer_confidence"]
    assert abs(ph["angry"]["noul"] - 0.5) < abs(pb["angry"]["noul"] - 0.5)
    assert base.predict("some text", QUESTIONS) == hot.predict("some text", QUESTIONS)
    # a bucket entry wins over the per-type value
    b = laya_vision.load(vision_ckpt(tiny_laya_dir, temps=([1.0, 1.0, 1.0], {"choice:3-5": 5.0})), device="cpu")
    assert b.predict(img, QUESTIONS)["answers"]["intent"] == ph["intent"]
    assert b.predict(img, QUESTIONS)["answers"]["angry"] == pb["angry"]


def test_errors(agent):
    with pytest.raises(ValueError, match="only the keys"):
        agent.predict({"image": noise_image(0), "caption": "x"}, QUESTIONS)
    with pytest.raises(ValueError, match="one image"):
        agent.predict({"image": [noise_image(0), noise_image(1)]}, QUESTIONS)
    with pytest.raises(ValueError):
        agent.predict({"image": b"garbage"}, QUESTIONS)
    with pytest.raises(ValueError, match="does not fit"):
        agent.predict({"image": noise_image(0)}, QUESTIONS, max_len=24, head_max_len=20)
    with pytest.raises(ValueError, match="predict_long"):
        agent.predict_long({"image": noise_image(0)}, QUESTIONS)
    with pytest.raises(TypeError):
        agent.predict_batch({"image": noise_image(0)}, QUESTIONS)


def test_fast_and_compile_rejected_with_image(agent, monkeypatch):
    monkeypatch.setattr(agent, "_compiled", True)
    with pytest.raises(ValueError, match="compile"):
        agent.predict({"image": noise_image(0)}, QUESTIONS)
    monkeypatch.setattr(agent, "_compiled", False)
    monkeypatch.setattr(agent, "_fast", object())
    with pytest.raises(ValueError, match="fast"):
        agent.predict({"image": noise_image(0)}, QUESTIONS)


def test_text_only_checkpoint_rejected(tiny_laya_dir):
    import laya_vision

    with pytest.raises(ValueError, match="laya.Agent"):
        laya_vision.load(tiny_laya_dir, device="cpu")


def test_subfolder(tiny_laya_dir, tmp_path):
    import laya_vision

    root = tmp_path / "repo"
    shutil.copytree(vision_ckpt(tiny_laya_dir), root / "sub")
    a = laya_vision.load(str(root), subfolder="sub", device="cpu")
    with open(os.path.join(a.model_dir, "rl_agent_config.json")) as f:
        assert "vision" in json.load(f)
    assert "VisionAgent" in repr(a)
