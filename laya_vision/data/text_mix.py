"""Text-only rows from ``LocalLLaMA/typed-decisions``, built exactly as Laya's fine-tuning notebook
builds its training targets (``build_training_item`` in docs/reference/laya_finetune_notebook.py):

- choice: ``[p.get(k, 0.0) for k in criteria]`` in criteria key order
- noul:   ``[p.get("false", 0.5), p.get("true", 0.5)]``
- score:  ``n = len(criteria)`` (4 when criteria is not a list), ``[p.get(str(i), 0.0) for i in range(n)]``
- normalised by the sum (uniform when the sum is 0); label = argmax.

Rows are ``image=None, text=state`` with the public question dict. The notebook's "skip when the
marker count differs from the option count" is enforced by ``DecisionDataset`` (build_item raises
ValueError -> row dropped). One sequence difference: the notebook calls ``build_sequence`` with
``truncate_left=False`` for every state, ``build_item`` left-truncates *list* states like
``laya.Agent`` does; typed-decisions states are dicts, so the sequences are identical.
"""
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schema import normalize, split_for, validate_record

DATASET_ID = "LocalLLaMA/typed-decisions"
CONFIG = "all"
TASK = "typed_decisions"


def gold_target(q: Dict[str, Any], gold_q: Dict[str, Any]) -> List[float]:
    t = q["type"]
    crit = q.get("criteria", {})
    probs = gold_q["probabilities"]
    if t == "choice":
        keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        target = [probs.get(str(k), 0.0) for k in keys]
    elif t == "noul":
        target = [probs.get("false", 0.5), probs.get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [probs.get(str(i), 0.0) for i in range(n_levels)]
    else:
        raise ValueError("unknown question type %r" % t)
    return normalize(target)


def _loads(x):
    return json.loads(x) if isinstance(x, str) else x


def convert_row(row: Dict[str, Any], split: Optional[str] = None, calib: float = 0.0, test: float = 0.0
                ) -> Iterator[Dict[str, Any]]:
    """One typed-decisions case -> one record per question that has a gold answer.

    ``split`` forces the split; otherwise it is hashed from the case id (``calib``/``test`` fractions),
    so all questions of a case land in the same split.
    """
    state, questions, gold = _loads(row["state"]), _loads(row["questions"]), _loads(row["gold"])
    sp = split or split_for(str(row["id"]), calib=calib, test=test, salt=TASK)
    for qid, q in questions.items():
        if qid not in gold:
            continue
        rec = {"id": "%s/%s/%s" % (TASK, row["id"], qid), "task": TASK, "split": sp, "image": None,
               "text": state, "question": q, "target": gold_target(q, gold[qid])}
        try:
            validate_record(rec)
        except ValueError:
            continue
        yield rec


def convert_rows(rows: Iterable[Dict[str, Any]], **kw) -> Iterator[Dict[str, Any]]:
    for row in rows:
        yield from convert_row(row, **kw)


def load_text_mix(hf_split: str = "train", dataset_id: str = DATASET_ID, config: str = CONFIG,
                  split: Optional[str] = None, calib: float = 0.0, test: float = 0.0,
                  revision: Optional[str] = None) -> List[Dict[str, Any]]:
    """Download ``dataset_id``/``config``/``hf_split`` and convert it. For the notebook's training
    set use ``hf_split="train"`` (optionally ``calib=0.05`` to hold out text calibration cases);
    the HF ``test`` split is the text-regression eval set (``split="test"``)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=hf_split, revision=revision)
    return list(convert_rows(ds, split=split, calib=calib, test=test))
