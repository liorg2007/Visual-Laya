"""Stage-2 converters: labelled vision datasets -> typed decision records (ARCHITECTURE §5.3).

Every source is described by a config dict in ``SOURCES`` (HF dataset id, splits, column names,
options); ``configs/data_stage2.yaml`` can override any field. Entries marked ``VERIFY`` are best
guesses that were not checked against the live dataset; see ``LICENSES.md``.

Splits are assigned by hashing the *image* id (``schema.split_for``), never the question, so no
image appears in two splits; the source's own train/test split is ignored (all ``hf_splits`` are
pooled). Images are saved once to ``image_root/{task}/{image_id}{ext}``.

Distractor questions (``n_distractors``, 0-2) add extra records for the same image: for
classification a ``noul`` "the image shows {label}" (gold or a random other class, 50/50) and a
second ``choice`` over a different option subset. VQA-style sources already have several
questions per image and ignore it.
"""
import hashlib
import io
import logging
import math
import os
import random
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .augment import instruction
from .schema import normalize, split_for

log = logging.getLogger(__name__)

EUROSAT_DESCRIPTIONS = {
    "Annual Crop": "fields of annual crops",
    "Forest": "forest",
    "Herbaceous Vegetation": "grassland or other herbaceous vegetation",
    "Highway": "a highway or major road",
    "Industrial Buildings": "industrial buildings",
    "Pasture": "pasture land",
    "Permanent Crop": "permanent crops such as orchards or vineyards",
    "Residential Buildings": "residential buildings",
    "River": "a river",
    "SeaLake": "a sea or lake",
}

QUALITY_LEVELS = {
    3: ["low quality: clearly degraded, blurry, noisy or badly exposed",
        "medium quality: acceptable with some visible flaws",
        "high quality: sharp, clean and well exposed"],
    4: ["bad: severely degraded", "poor: clearly visible flaws", "good: minor flaws only",
        "excellent: no visible flaws"],
    5: ["bad: severely degraded, hard to recognise the content",
        "poor: clearly visible blur, noise, compression or exposure problems",
        "fair: acceptable, with noticeable flaws",
        "good: minor flaws only",
        "excellent: sharp, clean and well exposed"],
}
AESTHETIC_LEVELS = {
    3: ["unappealing", "average", "beautiful"],
    5: ["very unappealing", "unappealing", "average", "appealing", "very beautiful"],
}

# kind: classify | mc | vqa_yesno | score. Defaults below; "VERIFY" = not checked against the source.
SOURCES: Dict[str, Dict[str, Any]] = {
    "cifar10": dict(kind="classify", dataset_id="uoft-cs/cifar10", config="plain_text",
                    hf_splits=["train", "test"], image_col="img", label_col="label",
                    ext=".png", options=[4, 10], template="cifar10"),
    "oxford_pets": dict(kind="classify", dataset_id="timm/oxford-iiit-pet", config=None,
                        hf_splits=["train", "test"], image_col="image", label_col="label",
                        id_col="image_id", options=[5, 20], template="oxford_pets"),
    "food101": dict(kind="classify", dataset_id="ethz/food101", config=None,
                    hf_splits=["train", "validation"], image_col="image", label_col="label",
                    options=[5, 20], template="food101"),
    "eurosat": dict(kind="classify", dataset_id="blanchon/EuroSAT_RGB", config=None,
                    hf_splits=["train", "validation", "test"], image_col="image", label_col="label",
                    id_col="filename", ext=".png", options=[4, 10], template="eurosat",
                    descriptions=EUROSAT_DESCRIPTIONS),
    "aokvqa": dict(kind="mc", dataset_id="HuggingFaceM4/A-OKVQA", config=None,
                   hf_splits=["train", "validation"], image_col="image", question_col="question",
                   choices_col="choices", answer_col="correct_choice_idx", id_col="question_id",
                   image_id_fn="sha1"),  # no image id column: images are hashed (shared COCO images)
    "scienceqa": dict(kind="mc", dataset_id="derek-thomas/ScienceQA", config=None,
                      hf_splits=["train", "validation", "test"], image_col="image", question_col="question",
                      choices_col="choices", answer_col="answer", hint_col="hint", image_id_fn="sha1"),
    # VERIFY: mirror of the VQAv2 val split with 10 answers per question (lmms-lab-encoder/VQAv2);
    # the official HuggingFaceM4/VQAv2 is a loading script (not usable with datasets>=4).
    "vqav2_yesno": dict(kind="vqa_yesno", dataset_id="lmms-lab-encoder/VQAv2", config=None,
                        hf_splits=["validation"], image_col="image", question_col="question",
                        answers_col="answers", answer_type_col="answer_type", image_id_col="image_id",
                        id_col="question_id", balance=True),
    # VERIFY: KonIQ-10k is not on the HF hub as a table; point csv_path/image_dir at the official
    # release (koniq10k_scores_and_distributions.csv + 1024x768/). c1..c5 are ACR vote counts.
    "koniq": dict(kind="score", csv_path="koniq10k/koniq10k_scores_and_distributions.csv",
                  image_dir="koniq10k/1024x768", image_col="image_name", mos_col="MOS", mos_range=[1, 5],
                  dist_cols=["c1", "c2", "c3", "c4", "c5"], std_col="SD", n_levels=5,
                  template="quality_score", levels="quality"),
    # VERIFY + licence: AVA is research-only; AVA.txt is whitespace separated without header:
    # index, image id, 10 vote counts (score 1..10), 2 tag ids, challenge id.
    "ava": dict(kind="score", csv_path="AVA/AVA.txt", image_dir="AVA/images", sep=r"\s+", header=False,
                names=["idx", "image_id"] + ["v%d" % i for i in range(1, 11)] + ["tag1", "tag2", "challenge"],
                image_col="image_id", image_template="{}.jpg", mos_range=[1, 10],
                dist_cols=["v%d" % i for i in range(1, 11)], n_levels=5, template="aesthetic_score",
                levels="aesthetic", enabled=False),
}


# ---------------------------------------------------------------------------- image IO

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("._") or "img"


def image_digest(img) -> str:
    """Short stable id of an image source (bytes if available, else pixels)."""
    if isinstance(img, dict) and img.get("bytes"):
        data = img["bytes"]
    elif isinstance(img, (bytes, bytearray)):
        data = bytes(img)
    else:
        pil = to_pil(img)
        data = pil.tobytes() + repr(pil.size).encode()
    return hashlib.sha1(data).hexdigest()[:20]


def to_pil(img):
    from PIL import Image

    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(io.BytesIO(img["bytes"]))
        img = img.get("path")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    if isinstance(img, str):
        if img.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(img, timeout=30) as r:
                return Image.open(io.BytesIO(r.read()))
        return Image.open(img)
    raise TypeError("unsupported image type %s" % type(img).__name__)


def save_image(img, image_root: str, rel: str, jpeg_quality: int = 95) -> str:
    """Save once to ``image_root/rel`` as RGB (JPEG or PNG by extension); returns ``rel``."""
    path = os.path.join(image_root, rel)
    if os.path.exists(path):
        return rel
    from PIL import Image

    pil = to_pil(img)
    if pil.mode in ("RGBA", "LA", "P"):
        pil = pil.convert("RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        pil = bg
    else:
        pil = pil.convert("RGB")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    if rel.lower().endswith(".png"):
        pil.save(tmp, format="PNG")
    else:
        pil.save(tmp, format="JPEG", quality=jpeg_quality)
    os.replace(tmp, path)  # atomic: parallel writers never see a half-written file
    return rel


def hf_rows(cfg: Dict[str, Any], seed: int = 0, streaming: bool = False, shuffle: bool = True
            ) -> Iterator[Tuple[str, int, Dict[str, Any], Any]]:
    """``(hf_split, row_index, row, features)`` over all ``hf_splits`` of an HF source."""
    from datasets import load_dataset

    for sp in cfg["hf_splits"]:
        ds = load_dataset(cfg["dataset_id"], cfg.get("config"), split=sp, streaming=streaming,
                          revision=cfg.get("revision"))
        feats = ds.features
        if shuffle:  # many sources are sorted by class; max_per_task must not see only a few classes
            ds = ds.shuffle(seed=seed, buffer_size=10_000) if streaming else ds.shuffle(seed=seed)
        for i, row in enumerate(ds):
            yield sp, i, row, feats


def csv_rows(cfg: Dict[str, Any], data_root: str = "") -> Iterator[Tuple[str, int, Dict[str, Any], Any]]:
    import csv

    path = os.path.join(data_root, cfg["csv_path"])
    with open(path, newline="", encoding="utf-8") as f:
        if cfg.get("sep", ",") == r"\s+":
            lines = (ln.split() for ln in f if ln.strip())
        else:
            lines = csv.reader(f, delimiter=cfg.get("sep", ","))
        names = cfg.get("names")
        if cfg.get("header", True):
            header = next(lines)
            names = names or header
        for i, vals in enumerate(lines):
            yield "csv", i, dict(zip(names, vals)), None


# ---------------------------------------------------------------------------- helpers

def pretty_label(name: str) -> str:
    return str(name).replace("_", " ").strip()


def label_names(cfg: Dict[str, Any], feats) -> List[str]:
    if cfg.get("label_names"):
        return list(cfg["label_names"])
    f = feats[cfg["label_col"]] if feats is not None else None
    if f is None or not hasattr(f, "names"):
        raise ValueError("source needs 'label_names' (label column is not a ClassLabel)")
    return list(f.names)


def _image_id(cfg, sp, i, row) -> str:
    if cfg.get("image_id_col"):
        return str(row[cfg["image_id_col"]])
    if cfg.get("image_id_fn") == "sha1":
        return image_digest(row[cfg["image_col"]])
    if cfg.get("id_col") and cfg["kind"] == "classify":
        v = str(row[cfg["id_col"]])
        base, ext = os.path.splitext(v)
        return base if ext.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp") else v
    return "%s-%d" % (sp, i)


def _ext(cfg) -> str:
    return cfg.get("ext", ".jpg")


def _split(task, image_id, cfg) -> str:
    return split_for("%s:%s" % (task, image_id), calib=cfg.get("calib", 0.02), test=cfg.get("test", 0.05))


def _choice_question(template: str, keys: List[str], descs: Dict[str, Optional[str]], fields=None) -> Dict:
    return {"type": "choice", "instructions": instruction(template, fields),
            "criteria": {k: descs.get(k) for k in keys}}


def _sample_options(rng: random.Random, names: List[str], gold: int, lo: int, hi: int, fixed: bool) -> List[int]:
    hi = min(hi, len(names))
    k = hi if fixed else rng.randint(min(lo, hi), hi)
    others = [j for j in range(len(names)) if j != gold]
    opts = rng.sample(others, k - 1) + [gold]
    rng.shuffle(opts)
    return opts


# ---------------------------------------------------------------------------- converters

def convert_classify(task: str, cfg: Dict[str, Any], rows, image_root: str, seed: int = 0,
                     n_distractors: int = 0) -> Iterator[Dict[str, Any]]:
    descs_src = cfg.get("descriptions") or {}
    lo, hi = cfg.get("options", [2, 20])
    names = None
    for sp, i, row, feats in rows:
        if names is None:
            raw = label_names(cfg, feats)
            names = [pretty_label(n) for n in raw]
            descs = {pretty_label(k): v for k, v in descs_src.items()}
            if len(set(names)) != len(names):
                raise ValueError("%s: duplicate label names after prettifying" % task)
        gold = int(row[cfg["label_col"]])
        img_id = _safe(_image_id(cfg, sp, i, row))
        split = _split(task, img_id, cfg)
        rng = random.Random("%d:%s:%s" % (seed, task, img_id))
        rel = save_image(row[cfg["image_col"]], image_root, "%s/%s%s" % (task, img_id, _ext(cfg)))
        base = {"task": task, "split": split, "image": rel, "text": None}
        n_q = 1 + max(0, min(2, n_distractors))
        for qi in range(n_q):
            rid = "%s/%s/q%d" % (task, img_id, qi)
            if qi == 1:  # distractor: noul "the image shows {label}", balanced gold/other
                pos = rng.random() < 0.5
                lab = gold if pos else rng.choice([j for j in range(len(names)) if j != gold])
                fields = {"label": names[lab]}
                q = {"type": "noul", "instructions": instruction("class_noul", fields),
                     "criteria": {"false": "the image does not show %s" % names[lab],
                                  "true": "the image shows %s" % names[lab]}}
                yield dict(base, id=rid, question=q, target=[0.0, 1.0] if pos else [1.0, 0.0],
                           template="class_noul", fields=fields)
                continue
            opts = _sample_options(rng, names, gold, lo, hi, fixed=(split == "test" and qi == 0))
            keys = [names[j] for j in opts]
            q = _choice_question(cfg.get("template", "classify"), keys, descs)
            yield dict(base, id=rid, question=q, target=[1.0 if j == gold else 0.0 for j in opts],
                       template=cfg.get("template", "classify"))


LETTERS = "ABCDEFGHIJKLMNOPQRST"


def convert_mc(task: str, cfg: Dict[str, Any], rows, image_root: str, seed: int = 0,
               n_distractors: int = 0) -> Iterator[Dict[str, Any]]:
    """Multiple-choice VQA (A-OKVQA, ScienceQA): keys A/B/C/... with the choice texts as descriptions."""
    for sp, i, row, _ in rows:
        img = row.get(cfg["image_col"])
        ans = row.get(cfg["answer_col"])
        choices = row.get(cfg["choices_col"]) or []
        if img is None or ans is None or not (2 <= len(choices) <= len(LETTERS)) or not (0 <= int(ans) < len(choices)):
            continue
        img_id = _safe(_image_id(cfg, sp, i, row))
        split = _split(task, img_id, cfg)
        rel = save_image(img, image_root, "%s/%s%s" % (task, img_id, _ext(cfg)))
        qtext = str(row[cfg["question_col"]]).strip()
        fields = {"q": qtext}
        q = {"type": "choice", "instructions": instruction("vqa_choice", fields),
             "criteria": {LETTERS[j]: str(c) for j, c in enumerate(choices)}}
        hint = str(row.get(cfg.get("hint_col") or "", "") or "").strip()
        qid = _safe(row[cfg["id_col"]]) if cfg.get("id_col") else "%s-%d" % (sp, i)
        yield {"id": "%s/%s/%s" % (task, img_id, qid), "task": task, "split": split, "image": rel,
               "text": hint or None, "question": q,
               "target": [1.0 if j == int(ans) else 0.0 for j in range(len(choices))],
               "template": "vqa_choice", "fields": fields}


def _answers(val) -> List[str]:
    out = []
    for a in val or []:
        out.append(str(a.get("answer", "") if isinstance(a, dict) else a).strip().lower())
    return out


def convert_vqa_yesno(task: str, cfg: Dict[str, Any], rows, image_root: str, seed: int = 0,
                      n_distractors: int = 0) -> Iterator[Dict[str, Any]]:
    """VQAv2 yes/no -> noul with a soft target (share of annotators answering yes). With
    ``balance`` the running count of yes-majority vs. no-majority items never differs by more than
    ``balance_slack`` per split."""
    slack = cfg.get("balance_slack", 50)
    counts: Dict[str, List[int]] = {}
    for sp, i, row, _ in rows:
        if cfg.get("answer_type_col") and row.get(cfg["answer_type_col"]) != "yes/no":
            continue
        ans = _answers(row.get(cfg["answers_col"]))
        n_yes, n_no = ans.count("yes"), ans.count("no")
        if n_yes + n_no == 0:
            continue
        p_yes = n_yes / (n_yes + n_no)
        img_id = _safe(_image_id(cfg, sp, i, row))
        split = _split(task, img_id, cfg)
        cls = int(p_yes >= 0.5)
        c = counts.setdefault(split, [0, 0])
        if cfg.get("balance", True) and c[cls] >= c[1 - cls] + slack:
            continue
        c[cls] += 1
        rel = save_image(row[cfg["image_col"]], image_root, "%s/%s%s" % (task, img_id, _ext(cfg)))
        qtext = str(row[cfg["question_col"]]).strip()
        fields = {"q": qtext}
        q = {"type": "noul", "instructions": instruction("vqa_noul", fields),
             "criteria": {"false": "no", "true": "yes"}}
        qid = _safe(row[cfg["id_col"]]) if cfg.get("id_col") else "%s-%d" % (sp, i)
        yield {"id": "%s/%s/%s" % (task, img_id, qid), "task": task, "split": split, "image": rel,
               "text": None, "question": q, "target": [n_no / (n_yes + n_no), p_yes],
               "template": "vqa_noul", "fields": fields}


def score_target(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[float]:
    """Soft target over ``n_levels`` buckets of ``mos_range``: from vote counts when ``dist_cols``
    is set, else a Gaussian N(MOS, SD) integrated per bucket when ``std_col`` is set, else one-hot."""
    n = int(cfg.get("n_levels", 5))
    lo, hi = (float(x) for x in cfg["mos_range"])

    def bucket(v: float) -> int:
        return min(n - 1, max(0, int((v - lo) / (hi - lo) * n))) if v < hi else n - 1

    if cfg.get("dist_cols"):
        cols = cfg["dist_cols"]
        vals = [lo + (hi - lo) * j / (len(cols) - 1) for j in range(len(cols))]
        t = [0.0] * n
        for c, v in zip(cols, vals):
            t[bucket(v)] += float(row[c])
        if sum(t) > 0:
            return normalize(t)
    mos = float(row[cfg["mos_col"]])
    sd = float(row[cfg["std_col"]]) if cfg.get("std_col") and row.get(cfg["std_col"]) not in (None, "") else 0.0
    if sd > 0:
        edges = [lo + (hi - lo) * j / n for j in range(n + 1)]
        cdf = [0.5 * (1 + math.erf((e - mos) / (sd * math.sqrt(2)))) for e in edges]
        cdf[0], cdf[-1] = 0.0, 1.0  # fold the tails into the end buckets
        return normalize([cdf[j + 1] - cdf[j] for j in range(n)])
    t = [0.0] * n
    t[bucket(mos)] = 1.0
    return t


def convert_score(task: str, cfg: Dict[str, Any], rows, image_root: str, seed: int = 0,
                  n_distractors: int = 0, data_root: str = "") -> Iterator[Dict[str, Any]]:
    n = int(cfg.get("n_levels", 5))
    levels = (AESTHETIC_LEVELS if cfg.get("levels") == "aesthetic" else QUALITY_LEVELS).get(n)
    if levels is None:
        raise ValueError("%s: no level descriptions for n_levels=%d" % (task, n))
    for sp, i, row, _ in rows:
        img = row[cfg["image_col"]]
        if isinstance(img, str) and cfg.get("image_dir"):
            name = cfg.get("image_template", "{}").format(img)
            img_id = _safe(os.path.splitext(name)[0])
            img = os.path.join(data_root, cfg["image_dir"], name)
            if not os.path.exists(img):
                continue
        else:
            img_id = _safe(_image_id(cfg, sp, i, row))
        try:
            target = score_target(row, cfg)
        except (KeyError, ValueError, TypeError):
            continue
        split = _split(task, img_id, cfg)
        rel = save_image(img, image_root, "%s/%s%s" % (task, img_id, _ext(cfg)))
        tpl = cfg.get("template", "quality_score")
        q = {"type": "score", "instructions": instruction(tpl), "criteria": list(levels)}
        yield {"id": "%s/%s/q0" % (task, img_id), "task": task, "split": split, "image": rel,
               "text": None, "question": q, "target": target, "template": tpl}


CONVERTERS: Dict[str, Callable] = {"classify": convert_classify, "mc": convert_mc,
                                   "vqa_yesno": convert_vqa_yesno, "score": convert_score}


def convert_task(task: str, image_root: str, overrides: Optional[Dict[str, Any]] = None, seed: int = 0,
                 n_distractors: int = 0, max_n: Optional[int] = None, streaming: bool = False,
                 data_root: str = "", rows: Optional[Iterable] = None) -> Iterator[Dict[str, Any]]:
    """Records for one task. ``rows`` (``(hf_split, index, row, features)`` tuples) replaces the
    download, which is how the tests feed synthetic data."""
    cfg = dict(SOURCES.get(task, {}))
    cfg.update(overrides or {})
    if "kind" not in cfg:
        raise ValueError("unknown task %r; known: %s" % (task, sorted(SOURCES)))
    if rows is None:
        rows = csv_rows(cfg, data_root) if cfg.get("csv_path") else hf_rows(cfg, seed, streaming)
    kw = {"data_root": data_root} if cfg["kind"] == "score" else {}
    gen = CONVERTERS[cfg["kind"]](task, cfg, rows, image_root, seed=seed, n_distractors=n_distractors, **kw)
    for n, rec in enumerate(gen):
        if max_n is not None and n >= max_n:
            break
        yield rec


__all__ = ["SOURCES", "convert_task", "score_target", "save_image", "hf_rows"]
