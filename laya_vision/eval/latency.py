"""Latency harness: p50/p95 for 1 and 10 questions per state, with and without an image.

    python -m laya_vision.eval.latency --checkpoint DIR [--device cuda] [--iters 50] [--stock]
        [--out eval/results/latency.json]

``text_q*`` rows use the VisionAgent text path, which is stock Laya's path; ``--stock`` also times
stock ``laya.Agent`` on the same checkpoint. ``overhead_p50_q1 = image_q1 / text_q1`` is the
number the Phase 5 exit criterion (≤ 2×, on GPU) is checked against.
"""
import argparse
import json
import os
import platform
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

TEXT = ("Customer writes: the blender I ordered arrived with a cracked jar and the motor smells burnt "
        "after one use. I want my money back, not a replacement.")


def questions(n: int) -> Dict[str, Any]:
    qs: Dict[str, Any] = {}
    for i in range(n):
        if i % 3 == 0:
            qs["q%d" % i] = {"type": "noul", "instructions": "Is the customer unhappy? (%d)" % i}
        elif i % 3 == 1:
            qs["q%d" % i] = {"type": "choice", "instructions": "What does the request concern? (%d)" % i,
                             "criteria": {"refund": "asks for money back", "replace": "asks for a new item",
                                          "info": "asks a question"}}
        else:
            qs["q%d" % i] = {"type": "score", "instructions": "How severe is the issue? (%d)" % i,
                             "criteria": ["minor", "moderate", "severe"]}
    return qs


def random_image(size: int = 640, seed: int = 0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB")


def time_calls(fn: Callable[[], Any], iters: int, warmup: int, sync: Optional[Callable[[], None]] = None
               ) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    ts: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n": iters}


def measure(agent, iters: int = 30, warmup: int = 5, image=None, stock=None, sync=None) -> Dict[str, Any]:
    image = image if image is not None else random_image()
    res: Dict[str, Any] = {}
    for nq in (1, 10):
        qs = questions(nq)
        res["text_q%d" % nq] = time_calls(lambda: agent.predict(TEXT, qs), iters, warmup, sync)
        res["image_q%d" % nq] = time_calls(lambda: agent.predict({"image": image, "text": TEXT}, qs),
                                           iters, warmup, sync)
        res["image_only_q%d" % nq] = time_calls(lambda: agent.predict({"image": image}, qs), iters, warmup, sync)
        if stock is not None:
            res["stock_text_q%d" % nq] = time_calls(lambda: stock.predict(TEXT, qs), iters, warmup, sync)
    return res


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=640, help="side of the random test image (pre-resize)")
    ap.add_argument("--stock", action="store_true", help="also time stock laya.Agent text-only")
    ap.add_argument("--out", default="eval/results/latency.json")
    args = ap.parse_args(argv)

    import torch

    import laya_vision

    agent = laya_vision.load(args.checkpoint, device=args.device)
    stock = None
    if args.stock:
        import laya

        stock = laya.Agent(args.checkpoint, device=args.device)
    device = str(getattr(agent, "device", args.device or "cpu"))
    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    res = measure(agent, args.iters, args.warmup, random_image(args.image_size), stock, sync)
    out = {"checkpoint": args.checkpoint, "device": device.split(":")[0], "device_full": device,
           "torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(),
           "iters": args.iters, "warmup": args.warmup, "results": res,
           "overhead_p50_q1": res["image_q1"]["p50_ms"] / res["text_q1"]["p50_ms"],
           "overhead_p50_q10": res["image_q10"]["p50_ms"] / res["text_q10"]["p50_ms"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for k, v in res.items():
        print("%-16s p50 %8.1f ms  p95 %8.1f ms" % (k, v["p50_ms"], v["p95_ms"]))
    print("image/text p50 (1 q): %.2fx" % out["overhead_p50_q1"])
    return out


if __name__ == "__main__":
    main()
