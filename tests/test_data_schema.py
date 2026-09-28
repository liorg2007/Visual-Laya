import pytest

from laya_vision.data.schema import TrainRecord, n_options, read_jsonl, split_for, validate_record, write_jsonl


def rec(**kw):
    r = {"id": "t/0/q0", "task": "t", "split": "train", "image": "t/0.png", "text": None,
         "question": {"type": "choice", "instructions": "Which?", "criteria": {"cat": None, "dog": "a dog"}},
         "target": [0.0, 1.0]}
    r.update(kw)
    return r


def test_valid_records():
    validate_record(rec())
    validate_record(rec(question={"type": "noul", "instructions": "It is a cat."}, target=[0.3, 0.7]))
    validate_record(rec(question={"type": "score", "instructions": "How good?", "criteria": ["a", "b", "c"]},
                        target=[0.2, 0.3, 0.5], image=None, text={"x": 1}))
    validate_record(rec(question={"type": "choice", "instructions": "Which?", "criteria": ["a", "b", "c"]},
                        target=[0.0, 0.0, 1.0]))
    assert n_options({"type": "noul", "instructions": "x"}) == 2


@pytest.mark.parametrize("bad", [
    {"target": [1.0]},                                   # wrong length
    {"target": [0.5, 0.6]},                              # does not sum to 1
    {"target": [-0.5, 1.5]},                             # negative
    {"split": "val"},
    {"image": "/abs/path.png"},
    {"question": {"type": "choice", "instructions": "x", "criteria": {}}},
    {"question": {"type": "score", "instructions": "x", "criteria": {"a": 1}}},
    {"question": {"type": "noul", "instructions": "x"}, "target": [1.0, 0.0, 0.0]},
])
def test_invalid_records(bad):
    with pytest.raises(ValueError):
        validate_record(rec(**bad))


def test_jsonl_roundtrip(tmp_path):
    p = str(tmp_path / "a" / "x.jsonl")
    tr = TrainRecord.from_dict(rec(fields={"q": "hi"}))
    assert write_jsonl(p, [rec(), tr]) == 2
    rows = list(read_jsonl(p))
    assert rows[0] == rec() and rows[1]["fields"] == {"q": "hi"} and "template" not in rows[1]
    with pytest.raises(ValueError):
        write_jsonl(p, [rec(target=[1.0])])


def test_split_for_deterministic_and_proportional():
    assert split_for("img1") == split_for("img1")
    s = [split_for("img%d" % i, calib=0.1, test=0.2) for i in range(5000)]
    assert abs(s.count("test") / 5000 - 0.2) < 0.03 and abs(s.count("calib") / 5000 - 0.1) < 0.03
