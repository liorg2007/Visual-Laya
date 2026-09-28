"""Training record schema: one JSONL row = one (state, question) pair.

```json
{"id": "oxford_pets/000123/q0", "task": "oxford_pets", "split": "train|calib|test",
 "image": "oxford_pets/000123.jpg",   // relative to image_root, or null for text-only rows
 "text": null,                         // optional text state (str | dict | list)
 "question": {"type": "choice", "instructions": "...", "criteria": {...}},  // public Laya question
 "target": [0.0, 1.0, 0.0],           // distribution in option order (noul: [false, true])
 "template": "oxford_pets",            // optional: augment.TEMPLATES key for paraphrases
 "fields": {"label": "..."}}           // optional: format fields for those templates
```

The target has one entry per option *as Laya renders them* (``laya.common.render_options`` on the
internal question), so choice targets follow ``criteria`` key order, score targets follow the level
list and noul targets are always ``[false, true]``.
"""
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from laya.common import render_options

from ..sequence import to_internal

SPLITS = ("train", "calib", "test")
TOL = 1e-4


@dataclass
class TrainRecord:
    id: str
    task: str
    split: str
    question: Dict[str, Any]
    target: List[float]
    image: Optional[str] = None
    text: Optional[Union[str, dict, list]] = None
    # Optional, used by augment.paraphrase: template key (defaults to task) and format fields.
    template: Optional[str] = None
    fields: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("template", "fields"):
            if d[k] is None:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


def n_options(question: Dict[str, Any]) -> int:
    """Number of options Laya renders for a public question dict."""
    return len(render_options(to_internal(question)))


def normalize(target: Iterable[float]) -> List[float]:
    """Normalise to a distribution; an all-zero target becomes uniform (the notebook's rule)."""
    t = [float(v) for v in target]
    s = sum(t)
    return [v / s for v in t] if s > 0 else [1.0 / len(t)] * len(t)


def validate_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Raise ``ValueError`` naming the record and the problem; return ``rec`` unchanged."""
    rid = rec.get("id")
    for k in ("id", "task", "split", "question", "target"):
        if k not in rec or rec[k] is None:
            raise ValueError("record %r: missing %r" % (rid, k))
    if rec["split"] not in SPLITS:
        raise ValueError("record %r: split must be one of %s, got %r" % (rid, SPLITS, rec["split"]))
    img = rec.get("image")
    if img is not None and (not isinstance(img, str) or os.path.isabs(img) or not img):
        raise ValueError("record %r: image must be a relative path or null, got %r" % (rid, img))
    try:
        k = n_options(rec["question"])
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError("record %r: invalid question: %s" % (rid, e)) from e
    t = rec["target"]
    if not isinstance(t, list) or len(t) != k:
        raise ValueError("record %r: target has %s entries but the question renders %d options"
                         % (rid, len(t) if isinstance(t, list) else type(t).__name__, k))
    if any((not isinstance(v, (int, float))) or math.isnan(v) or v < 0 for v in t):
        raise ValueError("record %r: target entries must be non-negative numbers: %r" % (rid, t))
    if abs(sum(t) - 1.0) > TOL:
        raise ValueError("record %r: target sums to %.6f, not 1" % (rid, sum(t)))
    return rec


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Union[Dict[str, Any], TrainRecord]], validate: bool = True,
                append: bool = False) -> int:
    """Write rows (dicts or TrainRecords); returns the number written."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            r = r.to_dict() if isinstance(r, TrainRecord) else r
            if validate:
                validate_record(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_for(key: str, calib: float = 0.02, test: float = 0.05, salt: str = "laya_vision") -> str:
    """Deterministic split from a stable key (an *image* id, never a question id)."""
    h = int.from_bytes(hashlib.sha1(("%s:%s" % (salt, key)).encode()).digest()[:8], "big") / 2.0 ** 64
    if h < test:
        return "test"
    if h < test + calib:
        return "calib"
    return "train"
