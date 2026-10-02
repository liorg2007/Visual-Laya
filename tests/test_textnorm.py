"""Request text checks: lone surrogates rejected, all-caps text lower-cased on the image path."""
import pytest

from _core_utils import QUESTIONS, noise_image
from laya_vision.textnorm import check_text, is_shouted, unshout_question, unshout_state


@pytest.mark.parametrize("s,want", [
    ("APPLE PIE", True), ("WHAT BREED IS THIS PET?", True), ("ÉCLAIR GLACÉ", True),
    ("apple pie", False), ("Apple Pie", False), ("DNA", False), ("CO2", False), ("A", False),
    ("USA OR UK", False), ("", False), ("1234", False), ("NaCl", False),
])
def test_is_shouted(s, want):
    assert is_shouted(s) is want


def test_unshout_question_keeps_order_and_falls_back_on_collisions():
    q = {"t": "choice", "ins": "WHICH DISH?", "crit": {"APPLE PIE": "A SWEET DESSERT", "DNA": None, 3: None}}
    out = unshout_question(q)
    assert out["ins"] == "which dish?"
    assert list(out["crit"].items()) == [("apple pie", "a sweet dessert"), ("DNA", None), (3, None)]
    assert q["crit"] == {"APPLE PIE": "A SWEET DESSERT", "DNA": None, 3: None}  # input untouched
    clash = {"t": "choice", "ins": "x", "crit": {"APPLE PIE": None, "apple pie": None}}
    assert list(unshout_question(clash)["crit"]) == ["APPLE PIE", "apple pie"]
    score = {"t": "score", "ins": "HOW BLURRY?", "crit": ["NOT BLURRY", "VERY BLURRY"]}
    assert unshout_question(score)["crit"] == ["not blurry", "very blurry"]


def test_unshout_state():
    assert unshout_state({"note": "LOUD NOISE", "n": 3, "turns": ["HELLO THERE", "ok"]}) == \
        {"note": "loud noise", "n": 3, "turns": ["hello there", "ok"]}


def test_check_text():
    check_text({"a": ["fine", 1, None, "ünïcode 😀"]}, "state")
    for bad in ("x\ud800", {"x\udfff": 1}, ["ok", {"k": "\ud83d"}]):
        with pytest.raises(ValueError, match="surrogate"):
            check_text(bad, "state")


def _q_upper(qs):
    out = {}
    for qid, q in qs.items():
        q = dict(q, instructions=q["instructions"].upper())
        c = q.get("criteria")
        if isinstance(c, dict):
            q["criteria"] = {k.upper(): (v.upper() if isinstance(v, str) else v) for k, v in c.items()}
        elif isinstance(c, list):
            q["criteria"] = [v.upper() for v in c]
        out[qid] = q
    return out


def _lower_all(qs):
    out = {}
    for qid, q in _q_upper(qs).items():
        q = dict(q, instructions=q["instructions"].lower())
        c = q.get("criteria")
        if isinstance(c, dict):
            q["criteria"] = {k.lower(): (v.lower() if isinstance(v, str) else v) for k, v in c.items()}
        elif isinstance(c, list):
            q["criteria"] = [v.lower() for v in c]
        out[qid] = q
    return out


def test_agent_unshouts_image_requests_only(tiny_laya_dir):
    import laya
    import laya_vision
    from _core_utils import vision_ckpt

    agent = laya_vision.load(vision_ckpt(tiny_laya_dir), device="cpu")
    img = noise_image(0)
    loud_s, loud_q = {"image": img, "text": "THE CUSTOMER IS FURIOUS"}, _q_upper(QUESTIONS)
    quiet_s, quiet_q = {"image": img, "text": "the customer is furious"}, _lower_all(QUESTIONS)

    def ids(state, qs):
        internal = {k: agent._to_internal(q) for k, q in qs.items()}
        return [it["ids"] for it in agent._encode_state(state, list(qs), internal)]

    # The tiny model's logits barely depend on the input, so compare the token sequences directly.
    assert ids(loud_s, loud_q) == ids(quiet_s, quiet_q)
    assert ids(loud_s, loud_q) != ids({"image": img, "text": "The Customer Is Furious"}, QUESTIONS)
    loud = agent.predict(loud_s, loud_q)
    assert list(loud["answers"]["intent"]["probabilities"]) == ["REFUND", "CANCEL", "OTHER"]
    assert loud["answers"]["intent"]["choice"] in ("REFUND", "CANCEL", "OTHER")
    # Text-only requests are untouched: still identical to stock laya.Agent on the same input.
    st = "THE CUSTOMER IS FURIOUS"
    assert ids(st, loud_q) != ids(st.lower(), quiet_q)
    stock = laya.Agent(vision_ckpt(tiny_laya_dir), device="cpu")
    assert agent.predict(st, loud_q)["answers"] == stock.predict(st, loud_q)["answers"]
    with pytest.raises(ValueError, match="surrogate"):
        agent.predict({"image": img}, {"q": dict(QUESTIONS["angry"], instructions="Angry\ud800?")})
    with pytest.raises(ValueError, match="surrogate"):
        agent.predict("hi\ud800", QUESTIONS)
    with pytest.raises(ValueError, match="surrogate"):
        agent.predict("hi", {"q\ud800": QUESTIONS["angry"]})
    with pytest.raises(ValueError, match="surrogate"):
        agent.predict_long("hi\ud800", QUESTIONS)
