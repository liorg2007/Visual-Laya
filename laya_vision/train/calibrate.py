"""Stage 3: per-modality temperature scaling on a held-out calib split (the notebook's ``fit_one_temp``).

    python -m laya_vision.train.calibrate --checkpoint runs/stage2_a/best --data data/calib/*.jsonl \
        --image-root data [--out runs/stage2_a/calibrated] [--refit-text]

For each modality, one temperature per question type and one per ``(qtype, temp_bucket)`` bucket
with at least ``--min-bucket`` items (fewer: the bucket falls back to the per-type value). Values
are clamped to Laya's runtime range [0.5, 5.0]; pre-clamp values are logged. Image temperatures
go to ``cfg["vision"]["temperature" / "temperature_by_options"]``; text temperatures replace the
stock fields only with ``--refit-text``. ECE@10 / ECE@15 on ``answer_confidence`` (max p) are
reported before (current config) and after.
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from laya.common import QTYPE_NAMES, TEMP_MAX, TEMP_MIN, clamp_temperature, ece_score, temp_bucket


def fit_one_temp(sel: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> Optional[float]:
    """Notebook ``fit_one_temp``: NLL-optimal T by LBFGS on log T. Unclamped; None if < 10 items."""
    if len(sel) < 10:
        return None
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z, dtype=torch.float32)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().item())


def fit_temperatures(rows: List[Dict], min_bucket: int = 100, min_type: int = 10,
                     fallback: Sequence[float] = (1.0, 1.0, 1.0)) -> Dict:
    """``{"temperature": [3], "temperature_by_options": {bucket: T}, "raw": {...}, "counts": {...}}``."""
    by_type, by_bucket = defaultdict(list), defaultdict(list)
    for r in rows:
        pair = (r["logits"], r["target"])
        by_type[r["qtype"]].append(pair)
        by_bucket[temp_bucket(r["qtype"], len(r["logits"]))].append(pair)
    temps, raw = [float(t) for t in fallback], {}
    for qt in range(3):
        if len(by_type[qt]) >= min_type:
            t = fit_one_temp(by_type[qt])
            raw[QTYPE_NAMES[qt]] = t
            temps[qt] = clamp_temperature(t)
    tbo = {}
    for b, sel in sorted(by_bucket.items()):
        if len(sel) >= min_bucket:
            t = fit_one_temp(sel)
            raw[b] = t
            tbo[b] = clamp_temperature(t)
    return {"temperature": temps, "temperature_by_options": tbo, "raw": raw,
            "counts": {**{QTYPE_NAMES[q]: len(v) for q, v in by_type.items()},
                       **{b: len(v) for b, v in by_bucket.items()}}}


def calib_metrics(rows: List[Dict], temps: Sequence[float], tbo: Dict[str, float]) -> Dict[str, float]:
    """Accuracy, NLL, ECE@10/@15 on max p after applying (clamped) temperatures the way the agent does."""
    if not rows:
        return {}
    conf, correct, nll = [], [], []
    for r in rows:
        z, y = np.asarray(r["logits"], dtype=np.float64), np.asarray(r["target"], dtype=np.float64)
        t = clamp_temperature(tbo.get(temp_bucket(r["qtype"], len(z)), temps[r["qtype"]]))
        p = np.exp(z / t - (z / t).max())
        p /= p.sum()
        conf.append(p.max())
        correct.append(float(p.argmax() == y.argmax()))
        nll.append(float(-(y / y.sum() * np.log(np.clip(p, 1e-12, 1))).sum()))
    conf, correct = np.array(conf), np.array(correct)
    return {"n": len(rows), "accuracy": float(correct.mean()), "nll": float(np.mean(nll)),
            "ece10": ece_score(conf, correct, bins=10), "ece15": ece_score(conf, correct, bins=15),
            "mean_conf": float(conf.mean())}


def current_temps(cfg: Dict, modality: str) -> Tuple[List[float], Dict[str, float]]:
    """Temperatures the agent applies today (same fallback rule as ``VisionAgent``)."""
    text_t, text_b = cfg.get("temperature", [1.0, 1.0, 1.0]), cfg.get("temperature_by_options", {}) or {}
    if modality == "text":
        return list(text_t), dict(text_b)
    v = cfg.get("vision", {})
    vt, vb = v.get("temperature"), v.get("temperature_by_options") or {}
    return list(vt if vt is not None else text_t), dict(vb if (vt is not None or vb) else text_b)


def run(checkpoint: str, data: Sequence[str], image_root: str = ".", out: Optional[str] = None,
        refit_text: bool = False, min_bucket: int = 100, batch_size: int = 32, device: str = "auto",
        amp: str = "auto", num_workers: int = 2, report: Optional[str] = None) -> Dict:
    from ..checkpoint import load_checkpoint, resolve_dir
    from ..data.dataset import DecisionDataset, make_collate
    from .loop import amp_setup, collect_predictions, pick_device

    dev = pick_device(device)
    src = resolve_dir(checkpoint)
    model, tok, cfg, vcfg = load_checkpoint(src, device="cpu")
    model.to(dev).eval()
    ds = DecisionDataset(list(data), image_root, tok, cfg["max_len"], cfg["head_max_len"], model.n_image_tokens)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                         collate_fn=make_collate(tok.pad_token_id, vcfg.image_size))
    amp_dtype, _ = amp_setup(amp, dev)
    rows = collect_predictions(model, loader, dev, amp_dtype)

    result: Dict = {"checkpoint": checkpoint, "data": list(data), "clamp": [TEMP_MIN, TEMP_MAX]}
    for mod in ("image", "text"):
        sel = [r for r in rows if r["modality"] == mod]
        if not sel:
            continue
        cur_t, cur_b = current_temps(cfg, mod)
        fit = fit_temperatures(sel, min_bucket=min_bucket, fallback=cur_t)
        result[mod] = {
            "fit": fit,
            "raw_T1": calib_metrics(sel, [1.0] * 3, {}),
            "before": calib_metrics(sel, cur_t, cur_b),
            "after": calib_metrics(sel, fit["temperature"], fit["temperature_by_options"]),
            "written": mod == "image" or refit_text,
        }
        for k, t in fit["raw"].items():
            if t is not None and clamp_temperature(t) != t:
                print("[calibrate] %s %s: fitted T=%.4f clamped to %.2f" % (mod, k, t, clamp_temperature(t)))
        b, a = result[mod]["before"], result[mod]["after"]
        print("[calibrate] %-5s n=%d acc=%.4f | ECE@10 %.4f -> %.4f | ECE@15 %.4f -> %.4f | NLL %.4f -> %.4f | T=%s %s%s"
              % (mod, b["n"], b["accuracy"], b["ece10"], a["ece10"], b["ece15"], a["ece15"], b["nll"], a["nll"],
                 [round(t, 3) for t in fit["temperature"]],
                 {k: round(v, 3) for k, v in fit["temperature_by_options"].items()},
                 "" if result[mod]["written"] else "  (not written; use --refit-text)"))

    out = out or src
    if os.path.abspath(out) != os.path.abspath(src):
        shutil.copytree(src, out, dirs_exist_ok=True)
    path = os.path.join(out, "rl_agent_config.json")
    with open(path) as f:
        new_cfg = json.load(f)
    if "image" in result:
        new_cfg["vision"]["temperature"] = result["image"]["fit"]["temperature"]
        new_cfg["vision"]["temperature_by_options"] = result["image"]["fit"]["temperature_by_options"]
    if refit_text and "text" in result:
        new_cfg["temperature"] = result["text"]["fit"]["temperature"]
        new_cfg["temperature_by_options"] = result["text"]["fit"]["temperature_by_options"]
    with open(path, "w") as f:
        json.dump(new_cfg, f, indent=2)
    report = report or os.path.join(out, "calibration_report.json")
    with open(report, "w") as f:
        json.dump(result, f, indent=2)
    print("[calibrate] wrote %s and %s" % (path, report))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="YAML whose keys are the long option names (e.g. configs/calibrate.yaml)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--image-root", dest="image_root", default=".")
    ap.add_argument("--out", help="output checkpoint dir (default: update --checkpoint in place)")
    ap.add_argument("--refit-text", dest="refit_text", action="store_true",
                    help="also replace the stock text temperatures (only if the encoder changed)")
    ap.add_argument("--min-bucket", dest="min_bucket", type=int, default=100)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto")
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    ap.add_argument("--report")
    args, _ = ap.parse_known_args(argv)
    if args.config:
        with open(args.config) as f:
            ap.set_defaults(**{k.replace("-", "_"): v for k, v in (yaml.safe_load(f) or {}).items()})
    args = ap.parse_args(argv)
    if isinstance(args.data, str):
        args.data = [args.data]
    if not args.checkpoint or not args.data:
        ap.error("--checkpoint and --data are required (on the command line or in --config)")
    kw = {k: v for k, v in vars(args).items() if k != "config"}
    return run(**kw)


if __name__ == "__main__":
    main()
