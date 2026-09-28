import base64
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from laya_vision import serve

QS = {"animal": {"type": "choice", "instructions": "Which animal is shown?",
                 "criteria": {"cat": "a cat", "dog": "a dog"}},
      "outdoor": {"type": "noul", "instructions": "Is it outdoors?"},
      "sharp": {"type": "score", "instructions": "How sharp?", "criteria": ["blurry", "ok", "sharp"]}}


def _img_bytes(fmt="PNG", size=(40, 30), seed=0):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


def _b64(data):
    return base64.b64encode(data).decode()


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, tiny_laya_dir, tiny_siglip_config):
    from laya_vision.checkpoint import init_from_laya, save_checkpoint
    from laya_vision.config import VisionConfig

    torch.manual_seed(0)
    vcfg = VisionConfig(image_size=32, patch_size=8, vision_width=32, pool_k=2)
    model, tok, cfg = init_from_laya(tiny_laya_dir, vcfg, tower_config=tiny_siglip_config)
    return save_checkpoint(model, tok, cfg, vcfg, str(tmp_path_factory.mktemp("tiny_vision_serve")))


@pytest.fixture(scope="module")
def agent(tiny_ckpt):
    import laya_vision

    return laya_vision.load(tiny_ckpt, device="cpu")


@pytest.fixture
def client(agent, monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(serve.create_app(agent=agent, checkpoint="tiny")) as c:
        yield c


def test_image_base64_and_data_url(client, agent):
    data = _img_bytes()
    r1 = client.post("/v1/systemone", json={"state": {"image": _b64(data)}, "questions": QS})
    assert r1.status_code == 200, r1.text
    ans = r1.json()["answers"]
    assert set(ans) == set(QS) and ans["animal"]["choice"] in ("cat", "dog")
    r2 = client.post("/v1/systemone", json={"state": {"image": "data:image/png;base64," + _b64(data)},
                                            "questions": QS})
    assert r2.status_code == 200 and r2.json()["answers"] == ans
    # same answers as the library on the same image
    assert agent.predict({"image": data}, QS)["answers"] == ans
    r3 = client.post("/v1/systemone", json={"state": {"image": _b64(_img_bytes("JPEG")), "text": "a note"},
                                            "questions": QS})
    assert r3.status_code == 200 and "X-Inference-Time-Ms" in r3.headers


def test_image_limits(client, monkeypatch):
    big = _b64(_img_bytes(size=(64, 64)))
    monkeypatch.setattr(serve, "MAX_IMAGE_BYTES", 100)
    r = client.post("/v1/systemone", json={"state": {"image": big}, "questions": QS})
    assert r.status_code == 413
    monkeypatch.setattr(serve, "MAX_IMAGE_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(serve, "MAX_IMAGE_PIXELS", 64 * 64 - 1)
    r = client.post("/v1/systemone", json={"state": {"image": big}, "questions": QS})
    assert r.status_code == 413 and "pixels" in r.json()["detail"]


def test_body_cap(agent, monkeypatch):
    monkeypatch.setenv("LAYA_VISION_MAX_BODY_BYTES", "2000")
    with TestClient(serve.create_app(agent=agent)) as c:
        r = c.post("/v1/systemone", json={"state": {"image": _b64(_img_bytes(size=(64, 64)))}, "questions": QS})
    assert r.status_code == 413


def test_text_body_keeps_laya_cap(client):
    from laya import serve as lserve

    r = client.post("/v1/systemone", json={"state": "x" * (lserve.MAX_BODY_BYTES + 10), "questions": QS})
    assert r.status_code == 413


@pytest.mark.parametrize("image", [
    _b64(_img_bytes("BMP")),                       # decodable but not an allowed serve format
    "not base64 at all!!",
    _b64(b"GIF89a but not really an image"),
    "data:text/plain;base64,aGVsbG8=",
    12345,
])
def test_bad_image(client, image):
    r = client.post("/v1/systemone", json={"state": {"image": image}, "questions": QS})
    assert r.status_code in (400, 422), r.text


def test_paths_are_never_read(client, tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(_img_bytes())
    r = client.post("/v1/systemone", json={"state": {"image": str(p)}, "questions": QS})
    assert r.status_code == 400


def test_request_validation(client):
    img = _b64(_img_bytes())
    assert client.post("/v1/systemone", json={"state": {"image": img, "extra": 1}, "questions": QS}).status_code == 400
    many = {"q%d" % i: QS["outdoor"] for i in range(100)}
    assert client.post("/v1/systemone", json={"state": {"image": img}, "questions": many}).status_code == 413
    assert client.post("/v1/systemone", json={"state": {"image": img, "text": "y" * 60000},
                                              "questions": QS}).status_code == 413
    assert client.post("/v1/systemone", json={"questions": QS}).status_code == 400
    assert client.post("/v1/systemone", content=b"{not json").status_code == 400
    bad_q = {"q": {"type": "choice", "instructions": "x"}}
    assert client.post("/v1/systemone", json={"state": {"image": img}, "questions": bad_q}).status_code == 422
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["vision"] and h["limits"]["max_image_bytes"] == 10 * 1024 * 1024


def test_bearer_token(agent, monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    with TestClient(serve.create_app(agent=agent)) as c:
        body = {"state": "hello", "questions": {"q": QS["outdoor"]}}
        assert c.post("/v1/systemone", json=body).status_code == 401
        assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


class _AgentRouter:
    """Minimal Router stand-in so laya.serve serves one stock laya.Agent."""

    loaded = []

    def __init__(self, agent):
        self.agent = agent

    def predict(self, state, questions, model=None, **kw):
        return self.agent.predict(state, questions, **kw)


@pytest.mark.parametrize("state", [
    "The package arrived broken and I want a refund.",
    {"ticket": 42, "body": "cannot log in", "tags": ["auth"]},
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello, how can I help?"}],
])
def test_text_only_parity_with_laya_serve(client, tiny_ckpt, state, monkeypatch):
    import laya
    from laya import serve as lserve

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    body = {"state": state, "questions": QS}
    with TestClient(lserve.create_app(router=_AgentRouter(laya.Agent(tiny_ckpt, device="cpu")))) as ref:
        want = ref.post("/v1/systemone", json=body)
    got = client.post("/v1/systemone", json=body)
    assert got.status_code == want.status_code == 200
    assert got.json() == want.json()
    # and the same errors for invalid text requests
    for bad in ({"state": None, "questions": QS}, {"state": "x" * 60000, "questions": QS}):
        with TestClient(lserve.create_app(router=_AgentRouter(None))) as ref:
            assert client.post("/v1/systemone", json=bad).status_code == ref.post("/v1/systemone", json=bad).status_code
