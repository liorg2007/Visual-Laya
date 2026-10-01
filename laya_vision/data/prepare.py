"""Build training JSONL files: ``{out}/{task}.{split}.jsonl`` plus images under ``--image-root``.

Stage 2 (vision tasks + text mix):
    python -m laya_vision.data.prepare --tasks cifar10,oxford_pets,food101,eurosat,aokvqa,scienceqa,vqav2_yesno \\
        --out data --image-root data/images --max-per-task 50000 --n-distractors 1 --text-mix
    python -m laya_vision.data.prepare --config configs/data_stage2.yaml

Stage 1 (caption alignment):
    python -m laya_vision.data.prepare --stage1 coco --n 500000 --out data --image-root data/images
    python -m laya_vision.data.prepare --config configs/data_stage1.yaml

Records written: ``data/{task}.{train,calib,test}.jsonl`` (stage 1: ``data/coco.*.jsonl``, text mix:
``data/typed_decisions.*.jsonl``) and ``data/{task}.stats.json`` with counts per split/type.
"""
import argparse
import json
import logging
import os
from collections import Counter
from typing import Any, Dict, Iterable, Optional

from .schema import validate_record

log = logging.getLogger("laya_vision.data.prepare")


class SplitWriter:
    """Validating writer for ``{out}/{name}.{split}.jsonl``; files are truncated on first write."""

    def __init__(self, out: str, name: str):
        self.out, self.name = out, name
        self.files: Dict[str, Any] = {}
        self.stats: Counter = Counter()
        os.makedirs(out, exist_ok=True)

    def path(self, split: str) -> str:
        return os.path.join(self.out, "%s.%s.jsonl" % (self.name, split))

    def write(self, rec: Dict[str, Any]) -> None:
        validate_record(rec)
        sp = rec["split"]
        if sp not in self.files:
            self.files[sp] = open(self.path(sp), "w", encoding="utf-8")
        self.files[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.stats[(sp, rec["question"]["type"])] += 1

    def write_all(self, recs: Iterable[Dict[str, Any]]) -> "SplitWriter":
        for r in recs:
            self.write(r)
        return self

    def close(self) -> Dict[str, Any]:
        for f in self.files.values():
            f.close()
        summary = {"files": {sp: self.path(sp) for sp in self.files},
                   "counts": {"%s/%s" % k: v for k, v in sorted(self.stats.items())},
                   "total": sum(self.stats.values())}
        with open(os.path.join(self.out, "%s.stats.json" % self.name), "w") as f:
            json.dump(summary, f, indent=2)
        log.info("%s: %s", self.name, summary["counts"])
        return summary


def run_stage2(tasks, out, image_root, overrides=None, max_per_task=None, seed=0, n_distractors=0,
               streaming=False, data_root=""):
    from .vision_tasks import SOURCES, convert_task

    res = {}
    for t in tasks:
        ov = dict((overrides or {}).get(t) or {})
        if not ov.pop("enabled", SOURCES.get(t, {}).get("enabled", True)):
            log.warning("task %s is disabled (licence/availability); set enabled: true to build it", t)
            continue
        n = ov.pop("max_per_task", max_per_task)
        w = SplitWriter(out, t)
        try:
            w.write_all(convert_task(t, image_root, ov, seed=seed, n_distractors=ov.pop("n_distractors", n_distractors),
                                     max_n=n, streaming=streaming, data_root=data_root))
        finally:
            res[t] = w.close()
    return res


def run_text_mix(out, hf_split="train", calib=0.05, test=0.0, dataset_id=None, config=None, test_split=True,
                 revision=None):
    from . import text_mix as tm

    kw = {"dataset_id": dataset_id or tm.DATASET_ID, "config": config or tm.CONFIG, "revision": revision}
    w = SplitWriter(out, tm.TASK)
    try:
        w.write_all(tm.load_text_mix(hf_split, calib=calib, test=test, **kw))
        if test_split:  # the HF test split is the text regression set
            w.write_all(tm.load_text_mix("test", split="test", **kw))
    finally:
        summary = w.close()
    return summary


def run_text_eval(out, tasks=("ag_news", "boolq"), max_items=5000, revisions=None):
    """Text regression sets -> data/{task}.test.jsonl (see text_eval.py). ``revisions``: {task: sha}."""
    from . import text_eval as te

    summaries = {}
    for task in tasks:
        w = SplitWriter(out, task)
        try:
            w.write_all(te.load_text_eval(task, max_items=max_items, revision=(revisions or {}).get(task)))
        finally:
            summaries[task] = w.close()
    return summaries


def run_stage1(out, image_root, n=500_000, source=None, seed=0, streaming=False, name="coco", **kw):
    from .caption_align import build_stage1

    w = SplitWriter(out, name)
    try:
        w.write_all(build_stage1(image_root, n=n, source=source, seed=seed, streaming=streaming, task=name, **kw))
    finally:
        summary = w.close()
    return summary


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="configs/data_stage{1,2}.yaml; CLI flags override it")
    p.add_argument("--tasks", help="comma-separated stage-2 tasks (see vision_tasks.SOURCES)")
    p.add_argument("--stage1", choices=["coco"], help="build stage-1 caption-matching data")
    p.add_argument("--n", type=int, help="stage-1 number of records (default 500000)")
    p.add_argument("--out", help="output dir for JSONL (default data)")
    p.add_argument("--image-root", help="where images are saved (default data/images)")
    p.add_argument("--data-root", help="root for local csv sources (KonIQ, AVA)")
    p.add_argument("--max-per-task", type=int)
    p.add_argument("--n-distractors", type=int, help="0-2 extra questions per image (classification)")
    p.add_argument("--questions-per-image", type=int, help="stage 1 (default 4)")
    p.add_argument("--hard-negatives", choices=["tfidf", "siglip"])
    p.add_argument("--text-mix", action="store_true", help="also write typed_decisions.*.jsonl")
    p.add_argument("--text-eval", action="store_true", help="also write ag_news/boolq.test.jsonl")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--seed", type=int)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg: Dict[str, Any] = {}
    if a.config:
        import yaml

        with open(a.config) as f:
            cfg = yaml.safe_load(f) or {}

    def opt(name, default=None):
        v = getattr(a, name.replace("-", "_"), None)
        return v if v is not None else cfg.get(name.replace("-", "_"), default)

    out, image_root, seed = opt("out", "data"), opt("image_root", "data/images"), opt("seed", 0)
    streaming = a.streaming or bool(cfg.get("streaming", False))
    did = False
    s1 = cfg.get("stage1") or {}
    if a.stage1 or s1:
        kw = {k: v for k, v in s1.items() if k in ("questions_per_image", "hard_negatives", "calib", "test",
                                                   "siglip_model", "max_hard", "noul_hard")}
        if s1.get("options"):
            kw["options"] = tuple(s1["options"])
        if a.questions_per_image:
            kw["questions_per_image"] = a.questions_per_image
        if a.hard_negatives:
            kw["hard_negatives"] = a.hard_negatives
        run_stage1(out, image_root, n=a.n or s1.get("n", 500_000), source=s1.get("source"), seed=seed,
                   streaming=streaming, name=s1.get("name", "coco"), **kw)
        did = True
    tasks = a.tasks.split(",") if a.tasks else list(cfg.get("tasks") or {})
    if tasks:
        run_stage2([t.strip() for t in tasks if t.strip()], out, image_root, overrides=cfg.get("tasks"),
                   max_per_task=opt("max_per_task"), seed=seed, n_distractors=opt("n_distractors", 0),
                   streaming=streaming, data_root=opt("data_root", ""))
        did = True
    tm = cfg.get("text_mix")
    if a.text_mix or tm:
        run_text_mix(out, **(tm if isinstance(tm, dict) else {}))
        did = True
    te = cfg.get("text_eval")
    if a.text_eval or te:
        run_text_eval(out, **(te if isinstance(te, dict) else {}))
        did = True
    if not did:
        p.error("nothing to do: pass --tasks, --stage1, --text-mix, --text-eval or --config")


if __name__ == "__main__":
    main()
