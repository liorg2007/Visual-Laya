import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from laya_vision.eval import baselines as B
from laya_vision.eval import run_eval as R

PETS = {"type": "choice", "instructions": "Which animal is shown?",
        "criteria": {"cat": "a cat", "dog": "a dog", "bird": "a bird"}}
YESNO = {"type": "noul", "instructions": "Is it outdoors?"}
QUALITY = {"type": "score", "instructions": "How sharp is the photo?", "criteria": ["blurry", "ok", "sharp"]}


def _png(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture(scope="module")
def eval_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("evaldata")
    os.makedirs(d / "img")
    recs = []
    for i in range(3):
        _png(str(d / "img" / ("%d.png" % i)), i)
        img = "img/%d.png" % i
        recs += [
            {"id": "pets/%d" % i, "task": "pets", "split": "test", "image": img, "text": None,
             "question": PETS, "target": [1.0 if j == i else 0.0 for j in range(3)]},
            {"id": "yn/%d" % i, "task": "yesno", "split": "test", "image": img, "text": None,
             "question": YESNO, "target": [0.0, 1.0] if i % 2 else [1.0, 0.0]},
            {"id": "q/%d" % i, "task": "quality", "split": "test", "image": img, "text": None,
             "question": QUALITY, "target": [0.0, 0.5, 0.5]},
        ]
    for i in range(2):
        recs.append({"id": "news/%d" % i, "task": "news", "split": "test", "image": None,
                     "text": "Stocks rallied today on strong earnings %d." % i,
                     "question": {"type": "choice", "instructions": "Topic?",
                                  "criteria": {"business": "business news", "sports": "sports news"}},
                     "target": [1.0, 0.0]})
    img_path, txt_path = str(d / "image.jsonl"), str(d / "text.jsonl")
    with open(img_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs if r["image"])
    with open(txt_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs if not r["image"])
    return {"dir": str(d), "records": recs, "paths": [img_path, txt_path]}


class CountingRunner:
    """Uniform answers; records every predict call."""

    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, sorted(questions)))
        ans = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                ans[qid] = {"type": "noul", "noul": 0.5, "answer_confidence": 0.5}
            elif q["type"] == "score":
                k = len(q["criteria"])
                ans[qid] = {"type": "score", "probabilities": {str(i): 1 / k for i in range(k)}}
            else:
                keys = list(q["criteria"])
                ans[qid] = B.choice_answer(keys, np.full(len(keys), 1 / len(keys)))
        return {"model": "fake", "answers": ans, "usage": {}}


def test_one_predict_call_per_state(eval_data):
    runner = CountingRunner()
    rows = R.run_records(runner, eval_data["records"], image_root=eval_data["dir"])
    assert len(rows) == len(eval_data["records"])
    assert len(runner.calls) == 3 + 2  # 3 images (3 questions each) + 2 text states
    assert all(len(qs) == 3 for s, qs in runner.calls if isinstance(s, dict))
    assert os.path.isabs(runner.calls[0][0]["image"])
    assert {r["modality"] for r in rows} == {"image", "text"}
    # max_questions splits a state's questions over several calls
    runner2 = CountingRunner()
    R.run_records(runner2, eval_data["records"], image_root=eval_data["dir"], max_questions=2)
    assert len(runner2.calls) == 3 * 2 + 2


def test_acceptance_logic():
    def res(accs, ece=0.05):
        return {"metrics": {"by_task": {t: {"accuracy": a} for t, a in accs.items()},
                            "by_modality": {"image": {"ece10": ece}}},
                "task_modality": {t: ("text" if t.startswith("t") else "image") for t in accs}}

    model = res({"i1": 0.9, "i2": 0.8, "i3": 0.7, "i4": 0.1, "t1": 0.80})
    base = {"caption_laya": res({"i1": 0.5, "i2": 0.5, "i3": 0.5, "i4": 0.5}),
            "stock_laya": res({"t1": 0.81})}
    lat = {"device": "cuda", "results": {"image_q1": {"p50_ms": 30}, "text_q1": {"p50_ms": 20}}}
    a = R.acceptance(model, base, lat)
    assert a["pass"], a
    assert a["checks"]["beats_caption_laya"]["wins"] == 3
    assert not R.acceptance(res({"i1": 0.9, "i2": 0.1, "i3": 0.1, "i4": 0.9, "t1": 0.8}), base)["pass"]
    assert not R.acceptance(res({"i1": 0.9, "i2": 0.8, "i3": 0.7, "t1": 0.70}), base)["pass"]  # text drop
    assert not R.acceptance(model, base, dict(lat, results={"image_q1": {"p50_ms": 50},
                                                            "text_q1": {"p50_ms": 20}}))["pass"]
    assert not R.acceptance(res({"i1": 0.9, "i2": 0.8, "i3": 0.7, "t1": 0.8}, ece=0.2), base)["pass"]


class FakeLaya:
    def __init__(self):
        self.states = []

    def predict(self, state, questions, **kw):
        self.states.append(state)
        return CountingRunner().predict(state, questions)


def test_caption_laya_state_and_cache(tmp_path, eval_data):
    calls = []

    def captioner(img):
        calls.append(img.size)
        return "a small dog on grass"

    img = os.path.join(eval_data["dir"], "img", "0.png")
    laya = FakeLaya()
    cl = B.CaptionLaya(laya, captioner=captioner, cache_dir=str(tmp_path))
    cl.predict({"image": img}, {"q": PETS})
    cl.predict({"image": img, "text": "owner says: it barks"}, {"q": PETS})
    cl.predict("plain text", {"q": PETS})
    assert laya.states == ["a small dog on grass",
                           {"image_caption": "a small dog on grass", "text": "owner says: it barks"},
                           "plain text"]
    assert len(calls) == 1  # second call hit the disk cache
    # a fresh instance reads the same cache
    B.CaptionLaya(FakeLaya(), captioner=captioner, cache_dir=str(tmp_path)).predict({"image": img}, {"q": PETS})
    assert len(calls) == 1
    # OCR hook
    s = B.CaptionLaya(FakeLaya(), captioner=captioner, ocr=lambda im: "STOP").text_state({"image": img})
    assert s == {"image_caption": "a small dog on grass", "ocr_text": "STOP"}


def test_stock_laya_skips_images(eval_data):
    sl = B.StockLaya(FakeLaya())
    assert sl.predict({"image": "x.png"}, {"q": PETS})["answers"] == {}
    assert "q" in sl.predict("text", {"q": PETS})["answers"]


class FakeSiglipProcessor:
    def __call__(self, text=None, images=None, return_tensors="pt", **kw):
        if images is not None:
            from laya_vision.images import preprocess
            return {"pixel_values": preprocess(images, 32)}
        ids = [[(ord(c) % 97) + 1 for c in t[:16]] + [0] * (16 - len(t[:16])) for t in text]
        return {"input_ids": torch.tensor(ids)}


def _tiny_siglip():
    from transformers import SiglipConfig, SiglipModel

    torch.manual_seed(0)
    c = SiglipConfig(text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                      num_attention_heads=2, max_position_embeddings=16),
                     vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                        num_attention_heads=2, image_size=32, patch_size=8))
    m = SiglipModel(c)
    with torch.no_grad():
        m.logit_scale.fill_(np.log(10.0))
        m.logit_bias.fill_(-1.0)
    return m


def test_siglip_zeroshot_matches_manual(eval_data):
    zs = B.SiglipZeroShot(_tiny_siglip(), FakeSiglipProcessor())
    img = os.path.join(eval_data["dir"], "img", "1.png")
    out = zs.predict({"image": img}, {"a": PETS, "b": YESNO, "c": QUALITY})["answers"]
    assert set(out) == {"a"}  # noul / score skipped
    assert zs.option_prompts(PETS) == ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    from laya_vision.images import load_image

    iv = zs.encode_image(load_image(img))
    tv = zs.encode_texts(zs.option_prompts(PETS))
    z = 10.0 * tv @ iv - 1.0
    p = np.exp(z - z.max())
    p /= p.sum()
    got = [out["a"]["probabilities"][k] for k in ("cat", "dog", "bird")]
    assert got == pytest.approx(p, abs=1e-4)
    assert out["a"]["answer_confidence"] == pytest.approx(p.max(), abs=1e-4)
    assert zs.predict("text only", {"a": PETS})["answers"] == {}


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory, tiny_laya_dir, tiny_siglip_config):
    from laya_vision.checkpoint import init_from_laya, save_checkpoint
    from laya_vision.config import VisionConfig

    torch.manual_seed(0)
    vcfg = VisionConfig(image_size=32, patch_size=8, vision_width=32, pool_k=2)
    model, tok, cfg = init_from_laya(tiny_laya_dir, vcfg, tower_config=tiny_siglip_config)
    return save_checkpoint(model, tok, cfg, vcfg, str(tmp_path_factory.mktemp("tiny_vision")))


def test_run_eval_end_to_end(tiny_ckpt, eval_data, tmp_path):
    out = str(tmp_path / "res" / "model.json")
    base_file = str(tmp_path / "baselines.json")
    R.main(["--checkpoint", tiny_ckpt, "--baseline", "stock_laya", "--data", *eval_data["paths"],
            "--image-root", eval_data["dir"], "--out", str(tmp_path / "stock.json"), "--record-baseline", base_file])
    res = R.main(["--checkpoint", tiny_ckpt, "--data", *eval_data["paths"], "--image-root", eval_data["dir"],
                  "--out", out, "--compare", base_file])
    m = res["metrics"]
    assert m["overall"]["n"] == len(eval_data["records"]) and m["overall"]["skipped"] == 0
    assert set(m["by_task"]) == {"pets", "yesno", "quality", "news"}
    assert "score_mae" in m["by_task"]["quality"] and 0 <= m["overall"]["ece10"] <= 1
    assert "text_regression" in res["acceptance"]["checks"]
    # the text-only path is stock Laya: identical text-task numbers
    with open(base_file) as f:
        stock = json.load(f)["stock_laya"]
    assert stock["metrics"]["by_task"]["news"]["nll"] == pytest.approx(m["by_task"]["news"]["nll"], abs=1e-3)
    assert stock["metrics"]["overall"]["skipped"] == 9  # image rows are skipped by stock_laya
    preds = [json.loads(line) for line in open(str(tmp_path / "res" / "model.predictions.jsonl"))]
    assert len(preds) == len(eval_data["records"]) and all(p["probs"] for p in preds)
    assert "| pets |" in open(str(tmp_path / "res" / "model.md")).read()


def test_latency_measure(tiny_ckpt):
    import laya_vision
    from laya_vision.eval.latency import measure, random_image

    agent = laya_vision.load(tiny_ckpt, device="cpu")
    res = measure(agent, iters=2, warmup=1, image=random_image(64))
    assert set(res) == {"text_q1", "image_q1", "image_only_q1", "text_q10", "image_q10", "image_only_q10"}
    assert all(v["p95_ms"] >= v["p50_ms"] > 0 for v in res.values())


@pytest.mark.slow
def test_real_baselines_smoke(eval_data):  # downloads BLIP-base and SigLIP-base
    img = os.path.join(eval_data["dir"], "img", "0.png")
    assert isinstance(B.BlipCaptioner()(Image.open(img).convert("RGB")), str)
    out = B.SiglipZeroShot.from_pretrained().predict({"image": img}, {"a": PETS})["answers"]["a"]
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
