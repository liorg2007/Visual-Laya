"""Run a checkpoint or a baseline over eval JSONL and write metrics, predictions and a summary.

    python -m laya_vision.eval.run_eval --checkpoint DIR --data a.jsonl b.jsonl --out results.json
    python -m laya_vision.eval.run_eval --config configs/eval.yaml --baseline caption_laya \
        --record-baseline eval/results/baselines.json --out eval/results/caption_laya.json
    python -m laya_vision.eval.run_eval --config configs/eval.yaml --checkpoint DIR \
        --compare eval/results/baselines.json --latency eval/results/latency.json --out eval/results/model.json

Records (docs/interfaces.md) are grouped by (image, text) state, so every question about one state
goes into one ``predict`` call, the way the model is meant to be used. Outputs: ``--out`` (metrics
JSON), ``<out stem>.predictions.jsonl`` and ``<out stem>.md``.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from laya_vision.eval.metrics import answer_probs, compute_metrics, markdown_table

BASELINES = ("caption_laya", "siglip_zeroshot", "stock_laya")
ACCEPTANCE = {"min_image_task_wins_frac": 0.75, "max_text_drop": 0.02, "max_image_ece10": 0.10,
              "max_latency_ratio": 2.0}


def read_jsonl(paths: Iterable[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        with open(p) as f:
            for i, line in enumerate(f):
                if line.strip():
                    r = json.loads(line)
                    r.setdefault("id", "%s:%d" % (os.path.basename(p), i))
                    out.append(r)
    return out


def group_records(records: Sequence[Dict[str, Any]], image_root: Optional[str] = None) -> "OrderedDict":
    """``{(image_path, text_json): [record, ...]}`` in first-seen order."""
    groups: "OrderedDict" = OrderedDict()
    for r in records:
        img = r.get("image")
        if img is not None and image_root and not os.path.isabs(img):
            img = os.path.join(image_root, img)
        key = (img, json.dumps(r.get("text"), sort_keys=True, ensure_ascii=False))
        groups.setdefault(key, []).append(r)
    return groups


def make_state(image: Optional[str], text: Any) -> Any:
    if image is None:
        return text
    return {"image": image} if text is None else {"image": image, "text": text}


def run_records(runner, records: Sequence[Dict[str, Any]], image_root: Optional[str] = None,
                max_questions: int = 32, log_every: int = 200) -> List[Dict[str, Any]]:
    """Prediction rows (see ``metrics``) for every record; failures are rows with ``probs=None``."""
    rows: List[Dict[str, Any]] = []
    groups = group_records(records, image_root)
    t0 = time.time()
    for g, ((image, _), recs) in enumerate(groups.items()):
        state = make_state(image, recs[0].get("text"))
        for c in range(0, len(recs), max_questions):
            chunk = recs[c:c + max_questions]
            questions = {"q%d" % j: r["question"] for j, r in enumerate(chunk)}
            err = None
            try:
                answers = runner.predict(state, questions)["answers"]
            except (ValueError, OSError) as e:  # e.g. image does not fit, unreadable image
                answers, err = {}, "%s: %s" % (type(e).__name__, e)
            for j, r in enumerate(chunk):
                a = answers.get("q%d" % j)
                row = {"id": r["id"], "task": r.get("task", "unknown"),
                       "modality": "image" if r.get("image") is not None else "text",
                       "qtype": r["question"]["type"], "target": r["target"],
                       "probs": answer_probs(a, r["question"]) if a is not None else None}
                if a is not None:
                    row["answer_confidence"] = a.get("answer_confidence")
                if err:
                    row["error"] = err
                rows.append(row)
        if log_every and (g + 1) % log_every == 0:
            print("[run_eval] %d/%d states, %.1fs" % (g + 1, len(groups), time.time() - t0), file=sys.stderr)
    return rows


def _acc(metrics: Dict[str, Any], task: str) -> Optional[float]:
    m = metrics.get("by_task", {}).get(task, {})
    return m.get("accuracy")


def task_modalities(rows: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in rows:
        out.setdefault(r["task"], r["modality"])
    return out


def acceptance(result: Dict[str, Any], baselines: Dict[str, Any], latency: Optional[Dict[str, Any]] = None,
               thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """plan.md §4 Phase 5 exit criteria. ``baselines`` = ``{name: results-json}`` (see --record-baseline)."""
    th = dict(ACCEPTANCE, **(thresholds or {}))
    metrics, mods = result["metrics"], result["task_modality"]
    checks: Dict[str, Any] = {}

    cap = baselines.get("caption_laya", {}).get("metrics")
    image_tasks = sorted(t for t, m in mods.items() if m == "image")
    if cap is not None and image_tasks:
        per = {t: {"model": _acc(metrics, t), "caption_laya": _acc(cap, t)} for t in image_tasks}
        wins = sum(1 for v in per.values() if v["caption_laya"] is not None and v["model"] is not None
                   and v["model"] > v["caption_laya"])
        need = math.ceil(th["min_image_task_wins_frac"] * len(image_tasks))
        checks["beats_caption_laya"] = {"pass": wins >= need, "wins": wins, "needed": need, "per_task": per}

    stock = baselines.get("stock_laya", {}).get("metrics")
    text_tasks = sorted(t for t, m in mods.items() if m == "text")
    if stock is not None and text_tasks:
        per = {}
        for t in text_tasks:
            a, b = _acc(metrics, t), _acc(stock, t)
            per[t] = {"model": a, "stock_laya": b, "drop": None if a is None or b is None else b - a}
        ok = all(v["drop"] is not None and v["drop"] <= th["max_text_drop"] for v in per.values())
        checks["text_regression"] = {"pass": ok, "max_drop": th["max_text_drop"], "per_task": per}

    ece = metrics.get("by_modality", {}).get("image", {}).get("ece10")
    if ece is not None:
        checks["image_ece10"] = {"pass": ece <= th["max_image_ece10"], "value": ece, "max": th["max_image_ece10"]}

    if latency:
        r = latency.get("results", {})
        img, txt = r.get("image_q1", {}).get("p50_ms"), r.get("text_q1", {}).get("p50_ms")
        if img and txt:
            ratio = img / txt
            checks["latency_p50_q1"] = {"pass": ratio <= th["max_latency_ratio"], "ratio": ratio,
                                        "device": latency.get("device"),
                                        "note": "criterion is defined on GPU" if latency.get("device") == "cpu" else ""}
    return {"pass": bool(checks) and all(c["pass"] for c in checks.values()), "checks": checks}


def build_runner(args) -> Any:
    from laya_vision.eval import baselines as B

    if args.baseline is None:
        import laya_vision

        return laya_vision.load(args.checkpoint, device=args.device)

    import laya

    laya_id = args.checkpoint or args.laya
    if args.baseline == "stock_laya":
        return B.StockLaya(laya.Agent(laya_id, device=args.device, revision=args.laya_revision))
    if args.baseline == "caption_laya":
        return B.CaptionLaya(laya.Agent(laya_id, device=args.device, revision=args.laya_revision),
                             cache_dir=args.caption_cache, captioner_id=args.captioner,
                             captioner_revision=args.captioner_revision, device=args.device)
    return B.SiglipZeroShot.from_pretrained(args.siglip, revision=args.siglip_revision, device=args.device)


def _stem(path: str) -> str:
    return path[:-5] if path.endswith(".json") else path


def evaluate(runner, records, out: str, name: str, image_root: Optional[str] = None, max_questions: int = 32,
             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t0 = time.time()
    rows = run_records(runner, records, image_root, max_questions)
    result = {"runner": name, "n_records": len(records), "seconds": round(time.time() - t0, 2),
              "task_modality": task_modalities(rows), "metrics": compute_metrics(rows), **(extra or {})}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(_stem(out) + ".predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(_stem(out) + ".md", "w") as f:
        f.write(markdown_table(result["metrics"], name))
    return result


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="configs/eval.yaml; supplies data paths and defaults")
    ap.add_argument("--checkpoint", help="laya_vision checkpoint (or the stock Laya id/dir for a baseline)")
    ap.add_argument("--data", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None, help="subset of config tasks")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--baseline", choices=BASELINES, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="baselines.json ({name: results}) for the acceptance check")
    ap.add_argument("--latency", help="latency JSON from laya_vision.eval.latency")
    ap.add_argument("--record-baseline", help="merge this run into a baselines.json under its baseline name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="max records (smoke runs)")
    for k, d in (("laya", None), ("laya-revision", None), ("captioner", None),
                 ("captioner-revision", None), ("caption-cache", None), ("siglip", None), ("siglip-revision", None)):
        ap.add_argument("--" + k, default=d)
    args = ap.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    bcfg = cfg.get("baselines", {})
    from laya_vision.eval.baselines import DEFAULT_CAPTIONER, DEFAULT_SIGLIP

    args.captioner = args.captioner or bcfg.get("captioner") or DEFAULT_CAPTIONER
    args.captioner_revision = args.captioner_revision or bcfg.get("captioner_revision")
    args.caption_cache = args.caption_cache or bcfg.get("caption_cache")
    args.siglip = args.siglip or bcfg.get("siglip") or DEFAULT_SIGLIP
    args.siglip_revision = args.siglip_revision or bcfg.get("siglip_revision")
    args.laya_revision = args.laya_revision or bcfg.get("laya_revision")
    args.laya = args.laya or bcfg.get("laya") or "convaiinnovations/laya"
    args.device = args.device or cfg.get("device")
    image_root = args.image_root or cfg.get("image_root")
    max_q = args.max_questions or cfg.get("max_questions_per_call", 32)

    paths = list(args.data or [])
    if not paths:
        tasks = {**cfg.get("image_tasks", {}), **cfg.get("text_tasks", {})}
        paths = [p for t, p in tasks.items() if not args.tasks or t in args.tasks]
    if not paths:
        ap.error("no data: pass --data or --config")
    if args.baseline is None and not args.checkpoint:
        ap.error("--checkpoint is required unless --baseline is given")

    records = read_jsonl(paths)
    if args.limit:
        records = records[:args.limit]
    name = args.baseline or "laya_vision"
    runner = build_runner(args)
    result = evaluate(runner, records, args.out, name, image_root, max_q,
                      extra={"checkpoint": args.checkpoint, "data": paths})

    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)
        lat = None
        if args.latency:
            with open(args.latency) as f:
                lat = json.load(f)
        result["acceptance"] = acceptance(result, base, lat, cfg.get("acceptance"))
        with open(_stem(args.out) + ".md", "a") as f:
            f.write("\n## Acceptance: %s\n\n" % ("PASS" if result["acceptance"]["pass"] else "FAIL"))
            for k, c in result["acceptance"]["checks"].items():
                f.write("- %s: %s\n" % (k, "pass" if c["pass"] else "FAIL"))
    _write_json(args.out, result)
    if args.record_baseline and args.baseline:
        base = {}
        if os.path.isfile(args.record_baseline):
            with open(args.record_baseline) as f:
                base = json.load(f)
        base[args.baseline] = {k: v for k, v in result.items() if k != "acceptance"}
        _write_json(args.record_baseline, base)
    with open(_stem(args.out) + ".md") as f:
        print(f.read())
    return result


if __name__ == "__main__":
    main()
