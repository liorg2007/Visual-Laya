import random

from laya_vision.data.augment import BANNED_CHOICE_KEYS, Augmenter, paraphrase, rename_labels, shuffle_options
from laya_vision.data.schema import validate_record


def choice_rec():
    crit = {"cat": "a cat", "dog": None, "bird": "a bird", "fish": None, "horse": "a horse"}
    return {"id": "x/0/q0", "task": "cifar10", "split": "train", "image": "x/0.png", "text": None,
            "question": {"type": "choice", "instructions": "What?", "criteria": crit},
            "target": [0.1, 0.6, 0.0, 0.3, 0.0]}


def test_shuffle_permutes_target_with_options():
    r0 = choice_rec()
    orig = dict(zip(r0["question"]["criteria"], r0["target"]))
    seen = set()
    for s in range(20):
        r = shuffle_options(r0, random.Random(s))
        validate_record(r)
        assert dict(zip(r["question"]["criteria"], r["target"])) == orig
        assert r["question"]["criteria"]["cat"] == "a cat"
        seen.add(tuple(r["question"]["criteria"]))
    assert len(seen) > 5
    assert r0 == choice_rec()  # input not mutated


def test_score_and_noul_never_shuffled():
    score = dict(choice_rec(), question={"type": "score", "instructions": "How?", "criteria": ["lo", "mid", "hi"]},
                 target=[0.7, 0.2, 0.1])
    noul = dict(choice_rec(), question={"type": "noul", "instructions": "It is."}, target=[0.2, 0.8])
    aug = Augmenter(p_shuffle=1.0, p_rename=1.0, p_paraphrase=0.0)
    for s in range(20):
        for r0 in (score, noul):
            r = aug(r0, random.Random(s))
            assert r["question"] == r0["question"] and r["target"] == r0["target"]


def test_rename_keeps_order_and_meaning():
    r0 = choice_rec()
    for s in range(30):
        r = rename_labels(r0, random.Random(s))
        validate_record(r)
        keys = list(r["question"]["criteria"])
        assert not BANNED_CHOICE_KEYS & {k.lower() for k in keys}
        assert r["target"] == r0["target"]
        assert list(r["question"]["criteria"].values()) == ["a cat", "dog", "a bird", "fish", "a horse"]


def test_paraphrase_and_determinism():
    r0 = dict(choice_rec(), template="vqa_choice", fields={"q": "Is it red?"})
    outs = {paraphrase(r0, random.Random(s))["question"]["instructions"] for s in range(30)}
    assert len(outs) > 1 and all("Is it red?" in o for o in outs)
    aug = Augmenter(p_rename=0.5, p_paraphrase=0.5)
    assert aug(r0, random.Random(3)) == aug(r0, random.Random(3))


def test_text_rows_untouched_by_default():
    r0 = dict(choice_rec(), image=None, text={"a": 1})
    assert Augmenter(p_rename=1.0)(r0, random.Random(0)) is r0
