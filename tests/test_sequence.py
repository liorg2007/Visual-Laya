"""Image block placement and truncation (plan.md §2.4)."""
import pytest

from laya.common import build_sequence

from laya_vision.sequence import build_item, collate_vision, is_image_state, split_state, to_internal

N = 49
Q = to_internal({"type": "choice", "instructions": "What is shown?",
                 "criteria": {"cat": "a cat", "dog": "a dog", "bird": None}})


def test_placeholders_at_image_start(tok):
    it = build_item(tok, {"image": "x", "text": "a caption"}, Q, N)
    s = it["image_start"]
    assert s > 0
    assert it["ids"][s:s + N] == [tok.pad_token_id] * N
    assert it["ids"][s - 1] == tok.sep_token_id and it["ids"][s + N] == tok.sep_token_id
    assert all(m < s for m in it["markers"])
    assert len(it["ids"]) <= 512


def test_image_only_state(tok):
    it = build_item(tok, {"image": "x"}, Q, N)
    s = it["image_start"]
    assert it["ids"][s:] == [tok.pad_token_id] * N + [tok.sep_token_id]


def test_long_text_never_displaces_image(tok):
    long = "word " * 5000
    it = build_item(tok, {"image": "x", "text": long}, Q, N, max_len=512)
    s = it["image_start"]
    assert len(it["ids"]) == 512
    assert it["ids"][s:s + N] == [tok.pad_token_id] * N
    assert it["ids"][-1] == tok.sep_token_id
    # text is right-truncated: its start survives
    text_ids = tok(long, add_special_tokens=False)["input_ids"]
    assert it["ids"][s + N + 1:s + N + 6] == text_ids[:5]


def test_list_text_left_truncated(tok):
    turns = ["old turn %d" % i for i in range(400)] + ["NEWEST TURN MARKER"]
    it = build_item(tok, {"image": "x", "text": turns}, Q, N, max_len=256)
    s = it["image_start"]
    assert it["ids"][s:s + N] == [tok.pad_token_id] * N
    decoded = tok.decode(it["ids"][s + N + 1:-1])
    assert "NEWEST TURN MARKER" in decoded
    assert "old turn 0\"" not in decoded and "old turn 1\"" not in decoded
    assert len(it["ids"]) == 256


def test_overflow_raises(tok):
    with pytest.raises(ValueError, match="image does not fit"):
        build_item(tok, {"image": "x"}, Q, n_img=200, max_len=128)
    # exactly fitting is fine
    head, _ = build_sequence(tok, None, Q, 128, 192, state_ids=[])
    room = 128 - (len(head) - 1) - 1
    it = build_item(tok, {"image": "x"}, Q, n_img=room, max_len=128)
    assert len(it["ids"]) <= 128
    with pytest.raises(ValueError):
        build_item(tok, {"image": "x"}, Q, n_img=room + 1, max_len=128)


@pytest.mark.parametrize("text", [None, "some text", {"k": "v"}, ["a", "b"]])
def test_markers_unchanged_vs_text_only(tok, text):
    ids_t, markers_t = build_sequence(tok, "any state", Q, 512, 192)
    it = build_item(tok, {"image": "x", "text": text}, Q, N)
    assert it["markers"] == markers_t
    assert it["ids"][:it["image_start"]] == ids_t[:it["image_start"]]


def test_text_rows_identical_to_laya(tok):
    for state in ["plain", {"a": 1}, ["t1", "t2"]]:
        ids, markers = build_sequence(tok, state, Q, 512, 192, truncate_left=isinstance(state, list))
        it = build_item(tok, state, Q, N)
        assert it["ids"] == ids and it["markers"] == markers and it["image_start"] == -1


def test_state_forms():
    assert is_image_state({"image": 1}) and not is_image_state({"img": 1}) and not is_image_state("image")
    assert split_state({"image": 1, "text": "t"}) == (1, "t")
    assert split_state("s") == (None, "s")
    with pytest.raises(ValueError):
        split_state({"image": 1, "caption": "x"})
    with pytest.raises(ValueError):
        split_state({"image": [1, 2]})


def test_collate_routing(tok):
    a = dict(build_item(tok, {"image": "x"}, Q, N), image_ref="A")
    t = build_item(tok, "text", Q, N)
    b2 = dict(build_item(tok, {"image": "y", "text": "hi"}, Q, N), image_ref="B")
    batch = collate_vision([a, t, b2, a], tok.pad_token_id)
    assert batch["image_refs"] == ["A", "B"]
    assert batch["image_index"].tolist() == [0, -1, 1, 0]
    assert batch["image_start"].tolist() == [a["image_start"], -1, b2["image_start"], a["image_start"]]
