"""Text-only regression sets (plan.md §4 Phase 3 item 7): AG News (choice) and BoolQ (noul).

Written as ``data/{ag_news,boolq}.test.jsonl`` with ``image=None``. They measure whether stage 2
cost text accuracy: run both stock Laya (``run_eval --baseline stock_laya``) and the trained
checkpoint on the same records. The question wording is ours, not the one behind Laya's published
AG News / BoolQ numbers, so compare the two models on these files, not against the README.
"""
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schema import validate_record

# HF ids and columns checked against the Hub on 2026-09-28; licences: see LICENSES.md.
SOURCES = {
    # label: 0 World, 1 Sports, 2 Business, 3 Sci/Tech
    "ag_news": {"dataset_id": "fancyzhx/ag_news", "hf_split": "test"},
    # columns: question, answer (bool), passage
    "boolq": {"dataset_id": "google/boolq", "hf_split": "validation"},
}

AG_NEWS_CRITERIA = {
    "world": "world news, politics, international affairs",
    "sports": "sports",
    "business": "business, economy, finance",
    "science_tech": "science and technology",
}


def convert_ag_news(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(AG_NEWS_CRITERIA)
    for i, row in enumerate(rows):
        target = [0.0] * len(keys)
        target[int(row["label"])] = 1.0
        rec = {"id": "ag_news/test/%06d" % i, "task": "ag_news", "split": "test", "image": None,
               "text": row["text"],
               "question": {"type": "choice", "instructions": "What is the topic of this news article?",
                            "criteria": dict(AG_NEWS_CRITERIA)},
               "target": target}
        validate_record(rec)
        yield rec


def convert_boolq(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for i, row in enumerate(rows):
        q = str(row["question"]).strip()
        q = q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?")
        rec = {"id": "boolq/test/%06d" % i, "task": "boolq", "split": "test", "image": None,
               "text": row["passage"],
               "question": {"type": "noul", "instructions": "According to the passage: %s" % q},
               "target": [0.0, 1.0] if bool(row["answer"]) else [1.0, 0.0]}
        validate_record(rec)
        yield rec


CONVERTERS = {"ag_news": convert_ag_news, "boolq": convert_boolq}


def load_text_eval(task: str, max_items: Optional[int] = None, revision: Optional[str] = None,
                   dataset_id: Optional[str] = None, hf_split: Optional[str] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    src = SOURCES[task]
    ds = load_dataset(dataset_id or src["dataset_id"], split=hf_split or src["hf_split"], revision=revision)
    if max_items:
        ds = ds.shuffle(seed=0).select(range(min(max_items, len(ds))))
    return list(CONVERTERS[task](ds))
