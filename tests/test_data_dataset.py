import os

import pytest
import torch

from laya_vision.data.dataset import DecisionDataset, LRUCache, MixtureSampler, build_mixture, make_collate
from laya_vision.data.schema import write_jsonl

N_IMG = 4
S = 32


def _png(path, color):
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)


def _records():
    choice = {"type": "choice", "instructions": "What is shown?", "criteria": {"cat": None, "dog": None, "bird": None}}
    noul = {"type": "noul", "instructions": "The image shows a cat."}
    score = {"type": "score", "instructions": "How good?", "criteria": ["bad", "ok", "good", "great"]}
    return [
        {"id": "a/q0", "task": "t", "split": "train", "image": "t/a.png", "text": None, "question": choice,
         "target": [0.0, 1.0, 0.0]},
        {"id": "a/q1", "task": "t", "split": "train", "image": "t/a.png", "text": "caption: a dog",
         "question": noul, "target": [0.9, 0.1]},
        {"id": "b/q0", "task": "t", "split": "train", "image": "t/b.png", "text": None, "question": score,
         "target": [0.1, 0.2, 0.3, 0.4]},
        {"id": "txt/q0", "task": "typed_decisions", "split": "train", "image": None, "text": {"ticket": "refund"},
         "question": choice, "target": [0.2, 0.3, 0.5]},
    ]


@pytest.fixture
def data(tmp_path):
    root = str(tmp_path / "images")
    _png(os.path.join(root, "t/a.png"), (255, 0, 0))
    _png(os.path.join(root, "t/b.png"), (0, 0, 255))
    p = str(tmp_path / "t.train.jsonl")
    write_jsonl(p, _records())
    return p, root


def test_items(data, tok):
    p, root = data
    ds = DecisionDataset(p, root, tok, 512, 192, N_IMG)
    assert len(ds) == 4
    it = ds[0]
    assert it["image_ref"] == os.path.abspath(os.path.join(root, "t/a.png"))
    assert it["label"] == 1 and it["target"] == [0.0, 1.0, 0.0] and it["task"] == "t"
    s = it["image_start"]
    assert s > 0 and it["ids"][s:s + N_IMG] == [tok.pad_token_id] * N_IMG
    assert ds[1]["ids"][ds[1]["image_start"] + N_IMG] == tok.sep_token_id  # text follows the image block
    assert ds[3]["image_start"] == -1 and ds[3]["image_ref"] is None


def test_collate(data, tok):
    pytest.importorskip("laya_vision.images")
    p, root = data
    ds = DecisionDataset(p, root, tok, 512, 192, N_IMG)
    cache = LRUCache(8)
    collate = make_collate(tok.pad_token_id, S, image_cache=cache)
    b = collate([ds[i] for i in range(4)] + [None])
    B = 4
    assert b["input_ids"].shape[0] == B and b["target"].shape == (B, 4)
    assert b["image_refs"] == [ds[0]["image_ref"], ds[2]["image_ref"]]
    assert b["pixel_values"].shape == (2, 3, S, S)
    assert b["image_index"].tolist() == [0, 0, 1, -1]
    assert b["image_start"].tolist() == [ds[i]["image_start"] for i in range(4)]
    for i in range(3):
        s = b["image_start"][i].item()
        assert (b["input_ids"][i, s:s + N_IMG] == tok.pad_token_id).all()
        assert b["attention_mask"][i, s:s + N_IMG].all()
    assert torch.allclose(b["target"][2], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert b["target"][0, 3] == 0 and b["label"].tolist() == [1, 0, 3, 2]
    assert len(cache) == 2
    b2 = collate([ds[2], ds[3]])  # cached tensor reused
    assert torch.equal(b2["pixel_values"][0], b["pixel_values"][1])
    tb = collate([ds[3]])
    assert tb["pixel_values"] is None and tb["image_index"].tolist() == [-1]
    assert collate([None]) is None


def test_skip_records_that_do_not_fit(data, tok, tmp_path):
    p, root = data
    many = {"type": "choice", "instructions": "Pick.",
            "criteria": {"option %d" % i: "a long description of option number %d " % i * 5 for i in range(20)}}
    bad = {"id": "big", "task": "t", "split": "train", "image": "t/a.png", "text": None, "question": many,
           "target": [1.0] + [0.0] * 19}
    p2 = str(tmp_path / "big.jsonl")
    write_jsonl(p2, [bad])
    # Laya squeezes options into head_max_len, so the big question leaves no room for 16 image tokens
    ds = DecisionDataset([p, p2], root, tok, 64, 48, 16)
    assert len(ds) == 5 and ds[4] is None and all(ds[i] is not None for i in range(4))
    ds = DecisionDataset([p, p2], root, tok, 64, 48, 16, prescan=True)
    assert len(ds) == 4 and ds.n_skipped == 1


def test_augment_deterministic_per_epoch(data, tok):
    from laya_vision.data.augment import Augmenter

    p, root = data
    ds = DecisionDataset(p, root, tok, 512, 192, N_IMG, augment=Augmenter(p_rename=0.5), seed=1)
    a = [ds[0]["ids"] for _ in range(2)]
    assert a[0] == a[1]
    orders = set()
    for e in range(10):
        ds.set_epoch(e)
        it = ds[0]
        orders.add(tuple(it["target"]))
        assert sum(it["target"]) == 1.0 and it["target"][it["label"]] == 1.0
    assert len(orders) > 1


def test_mixture_sampler_ratios_determinism_sharding():
    sizes = {"image": 1000, "text": 30}
    s = MixtureSampler(sizes, {"image": 0.75, "text": 0.25}, num_samples=400, seed=3)
    order = list(s)
    assert len(order) == 400 == len(s)
    n_text = sum(i >= 1000 for i in order)
    assert n_text == 100
    assert list(MixtureSampler(sizes, {"image": 0.75, "text": 0.25}, num_samples=400, seed=3)) == order
    s.set_epoch(1)
    assert list(s) != order
    # small source cycled without repeats inside one permutation
    s0 = MixtureSampler(sizes, {"image": 0.75, "text": 0.25}, num_samples=400, seed=3)
    txt = [i for i in s0 if i >= 1000]
    assert len(set(txt[:30])) <= 30 and sorted(set(txt)) == list(range(1000, 1030))
    shards = [list(MixtureSampler(sizes, {"image": 0.75, "text": 0.25}, num_samples=401, seed=3, rank=r,
                                  world_size=4)) for r in range(4)]
    assert all(len(x) == 101 for x in shards)
    full = MixtureSampler(sizes, {"image": 0.75, "text": 0.25}, num_samples=401, seed=3).global_order()
    assert sorted(sum(shards, [])) == sorted(full + full[:3])
    dl = MixtureSampler(sizes, {"image": 1, "text": 1}, num_samples=401, seed=0, world_size=4, drop_last=True)
    assert len(list(dl)) == 100
    with pytest.raises(ValueError):
        MixtureSampler({"a": 0}, {"a": 1.0}, 10)


def test_build_mixture_with_dataloader(data, tok):
    pytest.importorskip("laya_vision.images")
    p, root = data
    ds, sampler = build_mixture({"image": {"files": [p], "ratio": 0.5}, "text": {"files": [p], "ratio": 0.5}},
                                root, tok, 512, 192, N_IMG, num_samples=8, seed=0)
    ds.set_epoch(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, sampler=sampler,
                                         collate_fn=make_collate(tok.pad_token_id, S))
    batches = [b for b in loader if b is not None]
    assert sum(b["input_ids"].shape[0] for b in batches) == 8
    for b in batches:
        n_imgs = len(b["image_refs"])
        assert (b["pixel_values"] is None) == (n_imgs == 0)
        if n_imgs:
            assert b["pixel_values"].shape[0] == n_imgs == int(b["image_index"].max()) + 1
