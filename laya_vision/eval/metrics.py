"""Decision metrics shared by the model and every baseline (plan.md §1.1, ARCH §6).

A *prediction row* is a flat dict:

    {"id", "task", "modality": "image"|"text", "qtype": "choice"|"score"|"noul",
     "probs": [p_0, ...] | None,     # option order = record target order; None = skipped
     "target": [y_0, ...]}           # distribution in option order (soft targets allowed)

``answer_probs`` turns a Laya answer dict into ``probs``, so Laya, ``VisionAgent`` and the
baselines all go through the same code. Confidence for ECE is ``answer_confidence = max(p)``,
binned by ``laya.common.ece_score`` for parity with Laya's own figures.
"""
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from laya.common import ece_score

EPS = 1e-12


def answer_probs(answer: Dict[str, Any], qdef: Dict[str, Any]) -> List[float]:
    """Probability vector of a Laya-schema answer, in the question's option order."""
    t = answer.get("type", qdef.get("type"))
    if t == "noul":
        p = float(answer["noul"])
        return [1.0 - p, p]
    probs = answer["probabilities"]
    if t == "score":
        return [float(probs[str(i)]) for i in range(len(qdef["criteria"]))]
    crit = qdef["criteria"]
    keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
    return [float(probs[k]) for k in keys]


def _norm(p: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1.0 / len(p))


def row_stats(row: Dict[str, Any]) -> Dict[str, float]:
    """Per-row correct / Brier / NLL / confidence (and score error for score questions)."""
    p, y = _norm(row["probs"]), np.asarray(row["target"], dtype=np.float64)
    if len(p) != len(y):
        raise ValueError("row %s: %d probabilities for %d target options" % (row.get("id"), len(p), len(y)))
    y = y / y.sum()
    out = {
        "correct": float(int(p.argmax()) == int(y.argmax())),
        "brier": float(((p - y) ** 2).sum()),
        "nll": float(-(y * np.log(np.clip(p, EPS, 1.0))).sum()),
        "conf": float(p.max()),
    }
    if row.get("qtype") == "score":
        lv = np.arange(len(p))
        out["score_err"] = float(abs((lv * p).sum() - (lv * y).sum()))
    return out


def reliability_table(conf: Sequence[float], correct: Sequence[float], bins: int = 10) -> List[Dict[str, Any]]:
    """Reliability-diagram bins, same edges and edge rule as ``laya.common.ece_score``."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        n = int(sel.sum())
        rows.append({"lo": float(lo), "hi": float(hi), "count": n,
                     "confidence": float(conf[sel].mean()) if n else None,
                     "accuracy": float(correct[sel].mean()) if n else None})
    return rows


def summarize(rows: Iterable[Dict[str, Any]], reliability_bins: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate metrics over prediction rows; skipped rows (``probs`` None) are only counted."""
    rows = list(rows)
    done = [r for r in rows if r.get("probs") is not None]
    out: Dict[str, Any] = {"n": len(done), "skipped": len(rows) - len(done)}
    if not done:
        return out
    st = [row_stats(r) for r in done]
    conf = np.array([s["conf"] for s in st])
    corr = np.array([s["correct"] for s in st])
    out.update({
        "accuracy": float(corr.mean()),
        "brier": float(np.mean([s["brier"] for s in st])),
        "nll": float(np.mean([s["nll"] for s in st])),
        "ece10": ece_score(conf, corr, bins=10),
        "ece15": ece_score(conf, corr, bins=15),
        "mean_confidence": float(conf.mean()),
    })
    errs = [s["score_err"] for s in st if "score_err" in s]
    if errs:
        out["score_mae"] = float(np.mean(errs))
    if reliability_bins:
        out["reliability"] = reliability_table(conf, corr, reliability_bins)
    return out


def group_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[str(r.get(key))].append(r)
    return dict(sorted(g.items()))


def compute_metrics(rows: Sequence[Dict[str, Any]], reliability_bins: int = 10) -> Dict[str, Any]:
    """Overall, per-task, per-modality, per-qtype and per-(modality, qtype) metrics."""
    rows = list(rows)
    return {
        "overall": summarize(rows, reliability_bins),
        "by_task": {k: summarize(v) for k, v in group_by(rows, "task").items()},
        "by_modality": {k: summarize(v, reliability_bins) for k, v in group_by(rows, "modality").items()},
        "by_qtype": {k: summarize(v) for k, v in group_by(rows, "qtype").items()},
        "by_modality_qtype": {k: summarize(v) for k, v in group_by(
            [dict(r, mq="%s/%s" % (r.get("modality"), r.get("qtype"))) for r in rows], "mq").items()},
    }


def markdown_table(metrics: Dict[str, Any], title: str = "") -> str:
    """Per-task summary table (plus overall) in Markdown."""
    cols = ["n", "skipped", "accuracy", "brier", "nll", "ece10", "ece15", "score_mae"]

    def fmt(v):
        return "" if v is None else ("%d" % v if isinstance(v, int) else "%.4f" % v)

    lines = (["## %s" % title, ""] if title else []) + [
        "| task | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    items = list(metrics["by_task"].items()) + [("**overall**", metrics["overall"])]
    items += [("modality=%s" % k, v) for k, v in metrics.get("by_modality", {}).items()]
    for name, m in items:
        lines.append("| %s | " % name + " | ".join(fmt(m.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"
