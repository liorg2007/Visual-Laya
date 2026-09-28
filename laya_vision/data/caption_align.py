"""Stage-1 alignment records from an image-caption source (ARCHITECTURE §5.2).

Per image, ``questions_per_image`` records alternate between:

- ``choice`` "which caption describes the image": 4-8 options (keys A, B, ...; captions as
  descriptions), the true caption at a random position, >= 1 hard negative (nearest captions of
  *other* images) and random negatives for the rest; one-hot target.
- ``noul`` "this caption describes the image": each image gets one positive and one negative per
  pair of noul questions, so the set is balanced 50/50 by construction. Negatives are hard (50%)
  or random.

Hard negatives use TF-IDF cosine over caption words via an inverted index (pure Python, no extra
deps), or SigLIP text embeddings with ``hard_negatives="siglip"`` (GPU recommended).
"""
import logging
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .augment import instruction
from .schema import split_for
from .vision_tasks import _safe, save_image

log = logging.getLogger(__name__)

# VERIFY: jxie/coco_captions (checked via the HF datasets-server: columns image, filename, cocoid,
# caption; one caption per row, ~5 rows per image; splits train/validation/test = Karpathy splits).
COCO = dict(dataset_id="jxie/coco_captions", config=None, hf_splits=["train"], image_col="image",
            caption_col="caption", group_col="cocoid", revision=None)

STOP = set("a an the of in on at to and with is are its it this that there for by from as some two "
           "one his her their near next into while".split())
LETTERS = "ABCDEFGH"


def tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


class TfidfNeighbors:
    """Nearest captions by TF-IDF cosine. Tokens present in more than ``max_df`` captions are
    ignored for candidate generation (they would touch most of the corpus)."""

    def __init__(self, captions: Sequence[str], max_df: float = 0.02, max_postings: int = 5000):
        self.toks = [Counter(tokenize(c)) for c in captions]
        df = Counter(t for c in self.toks for t in c)
        n = len(captions)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self.norm = [math.sqrt(sum((tf * self.idf[t]) ** 2 for t, tf in c.items())) or 1.0 for c in self.toks]
        limit = max(50, int(max_df * n))
        self.post: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.toks):
            for t in c:
                if df[t] <= limit and len(self.post[t]) < max_postings:
                    self.post[t].append(i)

    def nearest(self, i: int, k: int = 16) -> List[int]:
        q = self.toks[i]
        scores: Dict[int, float] = defaultdict(float)
        for t, tf in q.items():
            w = tf * self.idf[t] ** 2
            for j in self.post.get(t, ()):
                if j != i:
                    scores[j] += w * self.toks[j][t]
        best = sorted(scores.items(), key=lambda x: (-x[1] / self.norm[x[0]], x[0]))[:k]
        return [j for j, _ in best]


class SiglipNeighbors:
    """Nearest captions by SigLIP text-embedding cosine; neighbours are precomputed in chunks."""

    def __init__(self, captions: Sequence[str], model_id: str = "google/siglip-base-patch16-224", k: int = 32,
                 batch_size: int = 512, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        embs = []
        with torch.no_grad():
            for s in range(0, len(captions), batch_size):
                enc = tok(list(captions[s:s + batch_size]), padding="max_length", truncation=True,
                          max_length=64, return_tensors="pt").to(device)
                e = model.get_text_features(**enc)
                embs.append(torch.nn.functional.normalize(e.float(), dim=-1).half())
            E = torch.cat(embs)
            self.nn: List[List[int]] = []
            for s in range(0, len(E), 1024):
                sims = E[s:s + 1024] @ E.T
                idx = torch.arange(s, min(s + 1024, len(E)), device=E.device)
                sims[torch.arange(len(idx)), idx] = -2
                self.nn.extend(sims.topk(min(k, len(E) - 1), dim=-1).indices.tolist())

    def nearest(self, i: int, k: int = 16) -> List[int]:
        return self.nn[i][:k]


def collect_images(rows: Iterable[Dict[str, Any]], image_root: str, src: Dict[str, Any], max_images: int,
                   task: str = "coco", patience: int = 2000
                   ) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """Group caption rows by image, saving each image once to ``image_root/{task}/{id}.jpg``.
    Returns ``(image_ids in first-seen order, image_id -> rel path, image_id -> captions)``.
    Once ``max_images`` are taken, scanning continues only until ``patience`` consecutive rows
    brought no caption for a taken image (caption rows of one image are usually adjacent)."""
    order: List[str] = []
    rel: Dict[str, str] = {}
    caps: Dict[str, List[str]] = defaultdict(list)
    idle = 0
    for row in rows:
        gid = _safe(row[src["group_col"]])
        if gid not in rel:
            if len(order) >= max_images:
                idle += 1
                if idle >= patience:
                    break
                continue
            idle = 0
            try:
                rel[gid] = save_image(row[src["image_col"]], image_root, "%s/%s.jpg" % (task, gid))
            except Exception as e:  # noqa: BLE001 - broken/unreachable images are skipped, not fatal
                log.warning("image %s skipped: %s", gid, e)
                continue
            order.append(gid)
        else:
            idle = 0
        c = row[src["caption_col"]]
        for cap in (c if isinstance(c, (list, tuple)) else [c]):
            cap = str(cap).strip()
            if cap and cap not in caps[gid]:
                caps[gid].append(cap)
    order = [g for g in order if caps.get(g)]
    return order, rel, dict(caps)


def generate(order: List[str], rel: Dict[str, str], caps: Dict[str, List[str]], n: Optional[int] = None,
             questions_per_image: int = 4, options: Tuple[int, int] = (4, 8), seed: int = 0,
             hard_negatives: str = "tfidf", task: str = "coco", calib: float = 0.01, test: float = 0.01,
             siglip_model: str = "google/siglip-base-patch16-224") -> Iterator[Dict[str, Any]]:
    """Caption-matching records; stops after ``n`` records (None = all images)."""
    pool: List[str] = []
    owner: List[str] = []
    for g in order:
        for c in caps[g]:
            pool.append(c)
            owner.append(g)
    first = {}
    for j, g in enumerate(owner):
        first.setdefault(g, j)
    if hard_negatives == "siglip":
        index = SiglipNeighbors(pool, siglip_model)
    elif hard_negatives == "tfidf":
        index = TfidfNeighbors(pool)
    else:
        raise ValueError("hard_negatives must be 'tfidf' or 'siglip'")
    lo, hi = options
    count = 0
    for g in order:
        rng = random.Random("%d:%s:%s" % (seed, task, g))
        split = split_for("%s:%s" % (task, g), calib=calib, test=test)
        own = caps[g]
        own_low = {c.lower() for c in own}
        idx0 = first[g]

        def negatives(gold_idx: int, n_hard: int, n_rand: int) -> List[str]:
            out: List[str] = []
            seen = set(own_low)
            for j in index.nearest(gold_idx, 16):
                if len(out) >= n_hard:
                    break
                if owner[j] != g and pool[j].lower() not in seen:
                    out.append(pool[j])
                    seen.add(pool[j].lower())
            tries = 0
            while len(out) < n_hard + n_rand and tries < 100:
                tries += 1
                j = rng.randrange(len(pool))
                if owner[j] != g and pool[j].lower() not in seen:
                    out.append(pool[j])
                    seen.add(pool[j].lower())
            return out

        noul_pos = rng.random() < 0.5  # which polarity the first noul question gets
        for qi in range(questions_per_image):
            if n is not None and count >= n:
                return
            ci = rng.randrange(len(own))
            gold_idx = idx0 + ci
            rid = "%s/%s/q%d" % (task, g, qi)
            base = {"id": rid, "task": task, "split": split, "image": rel[g], "text": None}
            if qi % 2 == 0:
                k = rng.randint(lo, hi)
                negs = negatives(gold_idx, rng.randint(1, max(1, k // 2)), k)[:k - 1]
                if len(negs) < 1:
                    continue
                opts = negs + [own[ci]]
                rng.shuffle(opts)
                q = {"type": "choice", "instructions": instruction("caption_choice"),
                     "criteria": {LETTERS[j]: c for j, c in enumerate(opts)}}
                target = [1.0 if c == own[ci] else 0.0 for c in opts]
                rec = dict(base, question=q, target=target, template="caption_choice")
            else:
                pos = noul_pos if (qi // 2) % 2 == 0 else not noul_pos
                if pos:
                    cap = own[ci]
                else:
                    hard = rng.random() < 0.5
                    negs = negatives(gold_idx, 1 if hard else 0, 0 if hard else 1)
                    if not negs:
                        continue
                    cap = negs[0]
                fields = {"caption": cap}
                q = {"type": "noul", "instructions": instruction("caption_noul", fields),
                     "criteria": {"false": "the caption does not match the image",
                                  "true": "the caption matches the image"}}
                rec = dict(base, question=q, target=[0.0, 1.0] if pos else [1.0, 0.0],
                           template="caption_noul", fields=fields)
            count += 1
            yield rec


def build_stage1(image_root: str, n: int = 500_000, source: Optional[Dict[str, Any]] = None,
                 questions_per_image: int = 4, seed: int = 0, hard_negatives: str = "tfidf",
                 streaming: bool = False, rows: Optional[Iterable[Dict[str, Any]]] = None,
                 task: str = "coco", **kw) -> Iterator[Dict[str, Any]]:
    """Download (or take ``rows``), save images, and yield ``n`` caption-matching records."""
    src = dict(COCO)
    src.update(source or {})
    if rows is None:
        from datasets import load_dataset

        from datasets import Image

        def _rows():
            for sp in src["hf_splits"]:
                ds = load_dataset(src["dataset_id"], src.get("config"), split=sp, streaming=streaming,
                                  revision=src.get("revision"))
                if src.get("image_col") in (ds.features or {}):
                    ds = ds.cast_column(src["image_col"], Image(decode=False))  # raw bytes: no decode for skipped rows
                yield from ds
        rows = _rows()
    max_images = -(-n // questions_per_image)
    order, rel, caps = collect_images(rows, image_root, src, max_images, task=task)
    log.info("stage1: %d images, %d captions", len(order), sum(len(v) for v in caps.values()))
    yield from generate(order, rel, caps, n=n, questions_per_image=questions_per_image, seed=seed,
                        hard_negatives=hard_negatives, task=task, **kw)
