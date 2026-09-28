import json
import os
from collections import Counter, defaultdict

import pytest
from PIL import Image

from laya_vision.data import caption_align, text_mix
from laya_vision.data.schema import read_jsonl, validate_record
from laya_vision.data.vision_tasks import convert_task, score_target


def _img(color=(10, 20, 30), size=(24, 24)):
    return Image.new("RGB", size, color)


# ------------------------------------------------------------------ text mix

FAKE_ROW = {
    "id": "tr_x_000001",
    "state": json.dumps({"task": "Delete data", "trace": {"steps": 7}}),
    "questions": json.dumps({
        "action": {"type": "choice", "instructions": "What next?",
                   "criteria": {"continue": "go", "stop": "halt", "review": "ask a human"}},
        "needs_review": {"type": "noul", "instructions": "Needs review.",
                         "criteria": {"false": "no", "true": "yes"}},
        "risk": {"type": "score", "instructions": "How risky?", "criteria": ["low", "mid", "high", "extreme"]},
        "no_gold": {"type": "noul", "instructions": "Unlabelled."},
    }),
    "gold": json.dumps({
        "action": {"label": "stop", "probabilities": {"continue": 0.2, "stop": 0.6}},   # review missing -> 0
        "needs_review": {"label": "true", "probabilities": {"true": 0.9}},               # false missing -> 0.5
        "risk": {"label": "1", "probabilities": {"0": 1.0, "1": 2.0, "3": 1.0}},         # unnormalised
    }),
}


def test_text_mix_matches_notebook_targets():
    recs = {r["id"].split("/")[-1]: r for r in text_mix.convert_row(FAKE_ROW)}
    assert set(recs) == {"action", "needs_review", "risk"}
    assert recs["action"]["target"] == pytest.approx([0.25, 0.75, 0.0])
    assert recs["needs_review"]["target"] == pytest.approx([0.5 / 1.4, 0.9 / 1.4])
    assert recs["risk"]["target"] == pytest.approx([0.25, 0.5, 0.0, 0.25])
    for r in recs.values():
        validate_record(r)
        assert r["image"] is None and r["text"] == json.loads(FAKE_ROW["state"])
        assert r["task"] == "typed_decisions"
    assert len({r["split"] for r in recs.values()}) == 1  # all questions of a case share the split
    assert all(r["split"] == "test" for r in text_mix.convert_row(FAKE_ROW, split="test"))


# ------------------------------------------------------------------ stage 2

def _classify_rows(n=40, n_cls=25):
    from datasets import ClassLabel, Features, Image as HFImage

    feats = Features({"img": HFImage(), "label": ClassLabel(names=["class_%d" % i for i in range(n_cls)])})
    return [("train", i, {"img": _img((i, 0, 0)), "label": i % n_cls}, feats) for i in range(n)]


def test_classify_converter(tmp_path):
    root = str(tmp_path / "img")
    recs = list(convert_task("food101", root, {"image_col": "img", "calib": 0.2, "test": 0.3}, seed=0,
                             n_distractors=2, rows=_classify_rows()))
    assert len(recs) == 40 * 3
    by_img = defaultdict(set)
    types = Counter()
    for r in recs:
        validate_record(r)
        assert os.path.exists(os.path.join(root, r["image"]))
        by_img[r["image"]].add(r["split"])
        types[r["question"]["type"]] += 1
        if r["question"]["type"] == "choice":
            k = len(r["question"]["criteria"])
            assert 5 <= k <= 20 and sum(r["target"]) == 1.0
            gold = list(r["question"]["criteria"])[r["target"].index(1.0)]
            if r["id"].endswith("q0") and r["split"] == "test":
                assert k == 20
            assert gold == "class %d" % (int(r["image"].split("-")[-1].split(".")[0]) % 25)
    assert all(len(s) == 1 for s in by_img.values())  # split by image
    assert types == {"choice": 80, "noul": 40}
    again = list(convert_task("food101", root, {"image_col": "img", "calib": 0.2, "test": 0.3}, seed=0,
                              n_distractors=2, rows=_classify_rows()))
    assert again == recs
    assert len(list(convert_task("food101", root, {"image_col": "img"}, max_n=7, rows=_classify_rows()))) == 7


def test_mc_and_vqa_converters(tmp_path):
    root = str(tmp_path / "img")
    mc_rows = [("train", i, {"image": _img((0, i, 0)), "question": "What color?", "choices": ["red", "green", "blue"],
                             "correct_choice_idx": i % 3, "question_id": "q%d" % i}, None) for i in range(6)]
    mc_rows.append(("test", 0, {"image": _img(), "question": "?", "choices": ["a", "b"], "correct_choice_idx": None,
                                "question_id": "unlabelled"}, None))
    recs = list(convert_task("aokvqa", root, rows=mc_rows))
    assert len(recs) == 6
    for i, r in enumerate(recs):
        validate_record(r)
        assert list(r["question"]["criteria"]) == ["A", "B", "C"] and r["target"][i % 3] == 1.0
        assert r["question"]["instructions"] == "What color?"

    sq = [("train", 0, {"image": _img(), "question": "Which?", "choices": ["x", "y"], "answer": 1,
                        "hint": "Context here."}, None),
          ("train", 1, {"image": None, "question": "No image", "choices": ["x", "y"], "answer": 0, "hint": ""}, None)]
    recs = list(convert_task("scienceqa", root, rows=sq))
    assert len(recs) == 1 and recs[0]["text"] == "Context here." and recs[0]["target"] == [0.0, 1.0]

    vqa = []
    for i in range(30):
        ans = ["yes"] * (8 if i % 3 else 3) + ["no"] * (2 if i % 3 else 7)
        vqa.append(("validation", i, {"image": _img((0, 0, i)), "question": "Is it blue?", "answer_type": "yes/no",
                                      "answers": [{"answer": a} for a in ans], "image_id": i // 2,
                                      "question_id": i}, None))
    vqa.append(("validation", 99, {"image": _img(), "question": "How many?", "answer_type": "number",
                                   "answers": [{"answer": "2"}], "image_id": 99, "question_id": 99}, None))
    recs = list(convert_task("vqav2_yesno", root, {"balance_slack": 1}, rows=vqa))
    for r in recs:
        validate_record(r)
        assert r["question"]["type"] == "noul" and r["target"] in ([0.2, 0.8], [0.7, 0.3])
    per_split = defaultdict(Counter)
    for r in recs:
        per_split[r["split"]][r["target"][1] > 0.5] += 1
    assert all(abs(c[True] - c[False]) <= 1 for c in per_split.values())
    assert len({r["image"] for r in recs}) <= 15  # two questions per image share the saved file


def test_score_target():
    cfg = {"mos_range": [1, 5], "n_levels": 5, "dist_cols": ["c1", "c2", "c3", "c4", "c5"], "mos_col": "MOS"}
    assert score_target({"c1": 0, "c2": 1, "c3": 3, "c4": 0, "c5": 0, "MOS": 2.75}, cfg) == [0, 0.25, 0.75, 0, 0]
    ava = {"mos_range": [1, 10], "n_levels": 5, "dist_cols": ["v%d" % i for i in range(1, 11)]}
    t = score_target({"v%d" % i: 1 for i in range(1, 11)}, ava)
    assert t == pytest.approx([0.2] * 5)
    g = score_target({"MOS": 3.0, "SD": 0.5}, {"mos_range": [1, 5], "n_levels": 5, "mos_col": "MOS", "std_col": "SD"})
    assert sum(g) == pytest.approx(1) and g.index(max(g)) == 2 and g[1] == pytest.approx(g[3])
    h = score_target({"MOS": 4.99}, {"mos_range": [1, 5], "n_levels": 3, "mos_col": "MOS"})
    assert h == [0, 0, 1]


def test_prepare_cli_koniq_csv(tmp_path):
    from laya_vision.data.prepare import main

    raw = tmp_path / "raw" / "koniq10k"
    (raw / "1024x768").mkdir(parents=True)
    lines = ["image_name,c1,c2,c3,c4,c5,c_total,MOS,SD,MOS_zscore"]
    for i in range(20):
        _img((i * 10, 0, 0)).save(raw / "1024x768" / ("%d.jpg" % i))
        lines.append("%d.jpg,%d,1,2,1,%d,5,3.0,0.8,50" % (i, i % 3, 2 - i % 3))
    lines.append("missing.jpg,1,1,1,1,1,5,3,1,50")
    (raw / "koniq10k_scores_and_distributions.csv").write_text("\n".join(lines) + "\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("out: %s\nimage_root: %s\ndata_root: %s\ntasks:\n  koniq: {test: 0.3, calib: 0.2}\n"
                   "  ava: {enabled: false}\n" % (tmp_path / "out", tmp_path / "images", tmp_path / "raw"))
    main(["--config", str(cfg)])
    stats = json.loads((tmp_path / "out" / "koniq.stats.json").read_text())
    assert stats["total"] == 20
    rows = [r for sp in ("train", "calib", "test") if (tmp_path / "out" / ("koniq.%s.jsonl" % sp)).exists()
            for r in read_jsonl(str(tmp_path / "out" / ("koniq.%s.jsonl" % sp)))]
    assert len(rows) == 20 and all(r["question"]["type"] == "score" and len(r["target"]) == 5 for r in rows)
    assert not (tmp_path / "out" / "ava.stats.json").exists()
    with pytest.raises(SystemExit):
        main([])


# ------------------------------------------------------------------ stage 1

CAPS = [
    ["a dog runs on the beach", "a brown dog running near the sea"],
    ["a dog sleeps on the beach towel", "a puppy asleep on sand"],
    ["a red bus parked on a city street", "a double decker bus in town"],
    ["a green bus driving down a city street"],
    ["two people eating pizza at a table", "friends share a pizza"],
    ["a cat sitting on a laptop keyboard"],
    ["a plate of pasta with tomato sauce"],
    ["a skier going down a snowy slope"],
    ["a surfer riding a large wave"],
    ["a plane on the runway at the airport"],
]


def _caption_rows():
    for g, caps in enumerate(CAPS):
        for c in caps:
            yield {"cocoid": g, "caption": c, "image": _img((g * 20, 0, 0))}


def test_tfidf_neighbors():
    pool = [c for caps in CAPS for c in caps]
    nn = caption_align.TfidfNeighbors(pool)
    near = [pool[j] for j in nn.nearest(pool.index("a red bus parked on a city street"), 3)]
    assert near[0] in ("a green bus driving down a city street", "a double decker bus in town")


def test_stage1_generation(tmp_path):
    root = str(tmp_path / "img")
    recs = list(caption_align.build_stage1(root, n=36, rows=_caption_rows(), questions_per_image=4, seed=0,
                                           calib=0.2, test=0.2))
    assert len(recs) == 36  # 9 images * 4
    types = Counter(r["question"]["type"] for r in recs)
    assert types == {"choice": 18, "noul": 18}
    noul = [r["target"][1] for r in recs if r["question"]["type"] == "noul"]
    assert sum(noul) == 9  # balanced 50/50
    splits = defaultdict(set)
    for r in recs:
        validate_record(r)
        splits[r["image"]].add(r["split"])
        assert os.path.exists(os.path.join(root, r["image"]))
        if r["question"]["type"] == "choice":
            crit = r["question"]["criteria"]
            assert 4 <= len(crit) <= 8
            gold = list(crit.values())[r["target"].index(1.0)]
            g = int(r["image"].split("/")[1].split(".")[0])
            assert gold in CAPS[g]
            assert not (set(crit.values()) - {gold}) & set(CAPS[g])
    assert all(len(s) == 1 for s in splits.values())
    again = list(caption_align.build_stage1(root, n=36, rows=_caption_rows(), questions_per_image=4, seed=0,
                                            calib=0.2, test=0.2))
    assert again == recs


@pytest.mark.slow
def test_text_mix_real_dataset():
    recs = text_mix.load_text_mix("test", split="test")
    assert len(recs) > 1000
    for r in recs[:200]:
        validate_record(r)
