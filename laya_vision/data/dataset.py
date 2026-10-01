"""Torch dataset, collate function and ratio-mixing sampler over JSONL training records.

Records that do not fit (``sequence.build_item`` raises ``ValueError``, e.g. too many options for
``head_max_len`` or no room for the image block) are handled like this:

- ``__getitem__`` returns ``None`` and ``make_collate`` drops ``None`` items. This is deterministic:
  the build depends only on ``(seed, epoch, index)``, so the same items are dropped on every run.
  A batch can therefore be smaller than ``batch_size`` (and ``None`` when every item was dropped;
  callers skip ``None`` batches).
- ``prescan=True`` builds every record once *without augmentation* at construction time and
  removes the ones that do not fit, so ``len()`` is exact (use it for calib/test sets). Augmented
  rows can still, rarely, stop fitting; those are dropped by the collate as above.
"""
import logging
import os
import random
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Union

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from ..sequence import build_item, collate_vision, to_internal
from .schema import read_jsonl

log = logging.getLogger(__name__)


def record_state(rec: Dict[str, Any], image_root: Optional[str]):
    """``(state, image_ref)`` of a record: an image state ``{"image": abs_path[, "text": ...]}``
    for image rows, the plain text state for text-only rows."""
    if rec.get("image") is None:
        return rec.get("text"), None
    ref = os.path.abspath(os.path.join(image_root or "", rec["image"]))
    state = {"image": ref}
    if rec.get("text") not in (None, ""):
        state["text"] = rec["text"]
    return state, ref


class DecisionDataset(Dataset):
    """Tokenizes lazily in ``__getitem__``; items are ``sequence.build_item`` output plus
    ``target``, ``label`` (argmax), ``image_ref`` (absolute path or None), ``task`` and ``id``.

    ``augment(record, rng) -> record`` (e.g. ``augment.Augmenter()``) is seeded per
    ``(seed, epoch, index)``; call ``set_epoch`` each epoch (with ``persistent_workers=True`` the
    workers keep their own copy, so prefer ``persistent_workers=False`` when augmenting).
    """

    def __init__(self, jsonl_paths: Union[str, Sequence[str]], image_root: Optional[str], tok,
                 max_len: int = 512, head_max_len: int = 192, n_img: int = 49,
                 augment: Optional[Callable] = None, seed: int = 0, prescan: bool = False,
                 records: Optional[List[Dict[str, Any]]] = None, teacher: bool = False):
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        self.records = list(records) if records is not None else [r for p in jsonl_paths for r in read_jsonl(p)]
        self.image_root = image_root
        self.tok = tok
        self.max_len, self.head_max_len, self.n_img = max_len, head_max_len, n_img
        self.augment = augment
        self.teacher = teacher
        self.seed = seed
        self.epoch = 0
        self.n_skipped = 0
        if prescan:
            keep = [r for r in self.records if self._build(r) is not None]
            self.n_skipped = len(self.records) - len(keep)
            if self.n_skipped:
                log.warning("DecisionDataset: skipped %d/%d records that do not fit (max_len=%d, "
                            "head_max_len=%d, n_img=%d)", self.n_skipped, len(self.records),
                            max_len, head_max_len, n_img)
            self.records = keep

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _build(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state, ref = record_state(rec, self.image_root)
        try:
            item = build_item(self.tok, state, to_internal(rec["question"], rec.get("id", "q")),
                              self.n_img, self.max_len, self.head_max_len)
        except ValueError as e:
            log.debug("record %s skipped: %s", rec.get("id"), e)
            return None
        target = [float(v) for v in rec["target"]]
        if len(target) != len(item["markers"]):
            raise ValueError("record %r: %d target entries for %d options"
                             % (rec.get("id"), len(target), len(item["markers"])))
        item.update(target=target, label=max(range(len(target)), key=target.__getitem__),
                    image_ref=ref, task=rec.get("task"), id=rec.get("id"))
        if self.teacher and ref is not None:
            item["teacher"] = self._teacher_item(rec, item)
        return item

    def _teacher_item(self, rec: Dict[str, Any], item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The same question asked about the image's true caption as a text-only state, for
        caption-teacher distillation (``train.distill_weight``). None when the record has no true
        caption (e.g. a noul row whose caption is a negative) or the markers do not line up."""
        text = teacher_text(rec)
        if not text:
            return None
        try:
            t = build_item(self.tok, text, to_internal(rec["question"], rec.get("id", "q")),
                           self.n_img, self.max_len, self.head_max_len)
        except ValueError:
            return None
        if t["markers"] != item["markers"]:
            return None
        return t

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        rec = self.records[i]
        if self.augment is not None:
            rec = self.augment(rec, random.Random("%d:%d:%d" % (self.seed, self.epoch, i)))
        return self._build(rec)


def teacher_text(rec: Dict[str, Any]) -> Optional[str]:
    """The true caption of an image record: the gold option of a caption ``choice`` row, or the
    caption of a ``noul`` caption row whose answer is true."""
    q, target = rec["question"], rec["target"]
    tmpl = rec.get("template") or ""
    if not tmpl.startswith("caption_"):
        return None
    gold = max(range(len(target)), key=target.__getitem__)
    if q["type"] == "choice":
        crit = q["criteria"]
        vals = list(crit.values()) if isinstance(crit, dict) else list(crit)
        return str(vals[gold]) if vals[gold] not in (None, "") else None
    if q["type"] == "noul" and gold == 1:
        return (rec.get("fields") or {}).get("caption")
    return None


class LRUCache(OrderedDict):
    """A tiny LRU mapping for preprocessed image tensors (``make_collate(image_cache=...)``)."""

    def __init__(self, maxsize: int = 256):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, k):
        v = super().__getitem__(k)
        self.move_to_end(k)
        return v

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.move_to_end(k)
        while len(self) > self.maxsize:
            self.popitem(last=False)


class VisionCollate:
    """Picklable collate callable; see ``make_collate``."""

    def __init__(self, pad_id: int, image_size: int, image_cache: Optional[MutableMapping] = None,
                 loader: Optional[Callable] = None):
        self.pad_id, self.image_size, self.image_cache, self.loader = pad_id, image_size, image_cache, loader

    def __call__(self, items: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        items = [it for it in items if it is not None]
        if not items:
            return None
        batch = collate_vision(items, self.pad_id)
        rows = [i for i, it in enumerate(items) if it.get("teacher") is not None]
        if rows:  # text-only teacher rows for caption distillation, aligned by ``teacher_rows``
            batch["teacher"] = collate_vision([items[i]["teacher"] for i in rows], self.pad_id)
            batch["teacher_rows"] = torch.tensor(rows, dtype=torch.long)
        refs = batch["image_refs"]
        if not refs:
            batch["pixel_values"] = None
            return batch
        from ..images import load_image, preprocess  # lazy: data code must not need the model deps

        load, cache = self.loader or load_image, self.image_cache
        out: List[Optional[torch.Tensor]] = [None] * len(refs)
        miss = []
        for j, r in enumerate(refs):
            if cache is not None and r in cache:
                out[j] = cache[r]
            else:
                miss.append(j)
        if miss:
            px = preprocess([load(refs[j]) for j in miss], self.image_size)
            for j, t in zip(miss, px):
                out[j] = t
                if cache is not None:
                    cache[refs[j]] = t
        batch["pixel_values"] = torch.stack(out)
        return batch


def make_collate(pad_id: int, image_size: int, image_cache: Optional[MutableMapping] = None,
                 loader: Optional[Callable] = None) -> VisionCollate:
    """``fn(items) -> batch``: ``collate_vision`` output (incl. ``target`` [B,Kmax] and ``label``)
    plus ``pixel_values`` [M,3,S,S] aligned with ``batch["image_refs"]`` (None when the batch has no
    images). Each unique image is loaded and preprocessed once per batch; ``image_cache`` (e.g.
    ``LRUCache(512)``, per worker process) additionally reuses tensors across batches. ``None``
    items are dropped and an all-``None`` batch collates to ``None``. ``loader(ref) -> PIL.Image``
    overrides ``images.load_image``."""
    return VisionCollate(pad_id, image_size, image_cache, loader)


class MixtureSampler(Sampler[int]):
    """Samples indices into ``ConcatDataset(datasets)`` so that each source gets ``ratio`` of the
    ``num_samples`` draws per epoch (``ratios`` normalised; sources in ``sizes`` order).

    Within a source, indices come from successive random permutations (no repeat before the
    source is exhausted; small sources are cycled). Deterministic for ``(seed, epoch)``; every rank
    computes the same global order and keeps ``order[rank::world_size]`` after padding the order to
    a multiple of ``world_size`` (or truncating it with ``drop_last``). Call ``set_epoch``.
    """

    def __init__(self, sizes: Mapping[str, int], ratios: Mapping[str, float], num_samples: Optional[int] = None,
                 seed: int = 0, rank: int = 0, world_size: int = 1, drop_last: bool = False):
        names = list(sizes)
        if set(ratios) - set(names):
            raise ValueError("ratios for unknown sources: %s" % sorted(set(ratios) - set(names)))
        w = [float(ratios.get(n, 0.0)) for n in names]
        if any(v < 0 for v in w) or sum(w) <= 0:
            raise ValueError("ratios must be non-negative with a positive sum")
        for n, v in zip(names, w):
            if v > 0 and sizes[n] <= 0:
                raise ValueError("source %r has ratio %g but no items" % (n, v))
        self.names, self.sizes = names, [int(sizes[n]) for n in names]
        self.weights = [v / sum(w) for v in w]
        if num_samples is None:  # one pass over the largest source relative to its share
            num_samples = int(max(s / p for s, p in zip(self.sizes, self.weights) if p > 0))
        self.num_samples = int(num_samples)
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.seed, self.rank, self.world_size, self.drop_last = seed, rank, world_size, drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def counts(self) -> List[int]:
        raw = [p * self.num_samples for p in self.weights]
        c = [int(x) for x in raw]
        # largest-remainder rounding so counts sum to num_samples
        for i in sorted(range(len(raw)), key=lambda i: raw[i] - c[i], reverse=True)[:self.num_samples - sum(c)]:
            c[i] += 1
        return c

    def global_order(self) -> List[int]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        order: List[int] = []
        for size, off, n in zip(self.sizes, self.offsets, self.counts()):
            picks: List[int] = []
            while len(picks) < n:
                picks.extend(torch.randperm(size, generator=g).tolist())
            order.extend(off + i for i in picks[:n])
        perm = torch.randperm(len(order), generator=g).tolist()
        return [order[i] for i in perm]

    def _per_rank(self) -> int:
        if self.drop_last:
            return self.num_samples // self.world_size
        return -(-self.num_samples // self.world_size)

    def __iter__(self) -> Iterator[int]:
        order = self.global_order()
        total = self._per_rank() * self.world_size
        if len(order) < total:
            order = order + order[:total - len(order)]
        return iter(order[:total][self.rank::self.world_size])

    def __len__(self) -> int:
        return self._per_rank()


def build_mixture(groups: Mapping[str, Dict[str, Any]], image_root: Optional[str], tok, max_len: int = 512,
                  head_max_len: int = 192, n_img: int = 49, augment: Optional[Callable] = None,
                  num_samples: Optional[int] = None, seed: int = 0, rank: int = 0, world_size: int = 1):
    """``groups = {"image": {"files": [...], "ratio": 0.75}, "text": {"files": [...], "ratio": 0.25}}``
    (the ``mixture`` block of ``configs/data_stage*.yaml``) -> ``(ConcatDataset, MixtureSampler)``.
    Use ``DataLoader(ds, batch_size=..., sampler=sampler, collate_fn=make_collate(...))`` and call
    ``ds.set_epoch(e); sampler.set_epoch(e)`` each epoch."""
    dsets = {name: DecisionDataset(g["files"], image_root, tok, max_len, head_max_len, n_img,
                                   augment=augment, seed=seed + i)
             for i, (name, g) in enumerate(groups.items())}
    sampler = MixtureSampler({n: len(d) for n, d in dsets.items()},
                             {n: g.get("ratio", 1.0) for n, g in groups.items()},
                             num_samples=num_samples, seed=seed, rank=rank, world_size=world_size)
    return MixtureDataset(dsets), sampler


class MixtureDataset(ConcatDataset):
    """``ConcatDataset`` over named sources; ``set_epoch`` reaches every source (picklable)."""

    def __init__(self, datasets_by_name: Mapping[str, DecisionDataset]):
        super().__init__(list(datasets_by_name.values()))
        self.names = list(datasets_by_name)

    def set_epoch(self, epoch: int) -> None:
        for d in self.datasets:
            d.set_epoch(epoch)
