"""Shared trainer for stage 1 (alignment) and stage 2 (decisions).

One YAML config drives everything (see ``configs/stage1_a.yaml``). The optimisation recipe is
the notebook's (``docs/reference/laya_finetune_notebook.py``): RLCD + soft CE, AdamW, cosine,
grad clip 1.0, sigma 0.4 -> 0.1, fp16/bf16 autocast, optional DDP via ``torchrun``.

Run through ``python -m laya_vision.train.stage1|stage2 --config ... [--override k=v ...]``.
"""
import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml

from laya.common import QTYPE_NAMES

from ..config import VisionConfig
from .rlcd import rlcd_loss, sigma_at

GROUPS = ("vision", "projector", "encoder", "head")

DEFAULTS: Dict[str, Any] = {
    "stage": 1,
    "seed": 0,
    "output_dir": "runs/stage1",
    "base": "convaiinnovations/laya",      # Hub id / dir, or "tiny" (random tiny Laya for smoke tests)
    "base_revision": None,
    "init_from": None,                     # laya_vision checkpoint dir to start from (stage 2)
    "resume": "auto",                      # auto | false | <checkpoint dir with trainer_state.pt>
    "device": "auto",
    "vision": {"tower": "google/siglip-base-patch16-224", "tower_revision": None,
               "feature_layer": -2, "pool_mode": "avg", "pool_k": 2},
    "data": {
        "train": [],            # [{path, weight?, image_root?}] or plain paths
        "eval": [],             # held-out JSONL(s); rows with and without images are scored separately
        "image_root": "data",
        "samples_per_epoch": None,     # default: one pass over the source largest relative to its weight
        "max_eval_items": 2000,
        "num_workers": 2,
        "augment": None,
        "max_len": None, "head_max_len": None,  # default: the base Laya config
        "synthetic": None,      # {dir, n_train, n_eval, n_calib}: generate a toy dataset (smoke runs)
    },
    "train": {
        "epochs": 1, "max_steps": None,
        "micro_batch": 8, "grad_accum": 4,
        # lr 0 / null = group frozen. Stage 1: projector only; stage 2: everything but vision.
        "lr": {"projector": 1e-3, "encoder": 0.0, "head": 0.0, "vision": 0.0},
        "unfreeze_vision_blocks": 0,   # >0 trains only the last N tower blocks at lr.vision
        "weight_decay": 0.0,
        "warmup_ratio": 0.03, "min_lr_ratio": 0.0,
        "grad_clip": 1.0,
        "group_size": 4, "sigma_start": 0.4, "sigma_end": 0.1,
        "w_sph": 0.75, "w_rps": 1.0, "ce_weight": 1.0,
        "amp": "auto",                 # auto | bf16 | fp16 | fp32
        "gradient_checkpointing": True,
        "detach_encoder": False,
        "lora": {"enabled": False, "r": 16, "alpha": 32, "dropout": 0.05, "targets": ["Wqkv", "Wo", "Wi"]},
        "log_every": 20, "eval_every": 500, "save_every": 500, "eval_at_start": True,
        "best_metric": "image.accuracy", "best_mode": "max",
        "early_stop_text_drop": None,  # stop when text accuracy falls this far below its step-0 value
        "early_stop_patience": 1,
    },
    "wandb": {"enabled": False, "project": "laya-vision", "name": None},
}


# ----------------------------------------------------------------------------- config

def deep_update(base: Dict, upd: Dict) -> Dict:
    out = dict(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_override(cfg: Dict, spec: str) -> None:
    """``a.b.c=value`` with a YAML-parsed value."""
    key, sep, raw = spec.partition("=")
    if not sep:
        raise ValueError("override must be key=value, got %r" % spec)
    d = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = yaml.safe_load(raw)


def load_config(path: Optional[str], overrides: Sequence[str] = (), stage_defaults: Optional[Dict] = None) -> Dict:
    cfg = deep_update(DEFAULTS, stage_defaults or {})
    if path:
        with open(path) as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    for o in overrides:
        apply_override(cfg, o)
    return cfg


# ----------------------------------------------------------------------------- runtime

def dist_info():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return world > 1, int(os.environ.get("RANK", "0")), world, int(os.environ.get("LOCAL_RANK", "0"))


def pick_device(name: str = "auto", local_rank: int = 0) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_setup(mode: str, device: torch.device):
    """``(autocast dtype or None, use GradScaler)``: bf16 where supported, else fp16 + scaler (CUDA)."""
    if mode == "fp32" or (mode == "auto" and device.type != "cuda"):
        return None, False
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, device.type == "cuda"


def make_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device: torch.device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, out_dir: str, enabled: bool, wandb_cfg: Optional[Dict] = None, run_cfg: Optional[Dict] = None):
        self.enabled = enabled
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self.wandb = None
        if enabled and wandb_cfg and wandb_cfg.get("enabled"):
            try:
                import wandb

                self.wandb = wandb.init(project=wandb_cfg.get("project"), name=wandb_cfg.get("name"), config=run_cfg)
            except Exception as e:  # optional dependency / offline box
                print("[train] wandb disabled: %s" % e)

    def log(self, rec: Dict[str, Any], echo: bool = True) -> None:
        if not self.enabled:
            return
        rec = dict(rec, time=round(time.time(), 1))
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if echo:
            print("[train] " + " ".join("%s=%s" % (k, ("%.4g" % v) if isinstance(v, float) else v)
                                        for k, v in rec.items() if k != "time" and not isinstance(v, (dict, list))), flush=True)
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in _flatten(rec).items() if isinstance(v, (int, float))},
                           step=rec.get("step"))


def _flatten(d: Dict, prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------- data

def _sources(spec) -> List[Dict[str, Any]]:
    """Normalise ``data.train`` / ``data.eval``: a path, a list of paths / ``{files|path, weight|ratio,
    image_root}`` dicts, or a data-agent ``mixture`` mapping ``{name: {files, ratio}}``."""
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, dict):
        spec = [dict(v, name=k) for k, v in spec.items()]
    out = []
    for s in spec or []:
        s = {"files": [s]} if isinstance(s, str) else dict(s)
        files = s.pop("files", None) or s.pop("path")
        s["files"] = [files] if isinstance(files, str) else list(files)
        if "ratio" in s:
            s["weight"] = s.pop("ratio")
        out.append(s)
    return out


class EpochSampler(torch.utils.data.Sampler):
    """Deterministic (seed, epoch) mixture sampler, sharded across ranks, resumable mid-epoch.

    ``sizes``/``weights`` describe consecutive sub-datasets of a ``ConcatDataset``. Each epoch draws
    ``round(w_i / sum(w) * num_samples)`` rows from source i (cycling fresh permutations when a
    source is smaller than its share), shuffles, trims to a multiple of ``world`` so every rank gets
    the same number of batches, and keeps every ``world``-th index for this rank.
    """

    def __init__(self, sizes: List[int], weights: Optional[List[float]], num_samples: Optional[int],
                 seed: int = 0, rank: int = 0, world: int = 1):
        self.sizes, self.seed, self.rank, self.world = sizes, seed, rank, world
        self.weights = weights if weights else [float(n) for n in sizes]
        # default: one pass over the source that is largest relative to its share
        wsum = sum(self.weights)
        self.num_samples = int(num_samples or max(n / (w / wsum) for n, w in zip(sizes, self.weights) if w > 0))
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def indices(self) -> List[int]:
        g = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch)
        wsum = sum(self.weights)
        out, offset = [], 0
        for n, w in zip(self.sizes, self.weights):
            want = int(round(w / wsum * self.num_samples)) if n else 0
            got: List[int] = []
            while len(got) < want:
                got += (torch.randperm(n, generator=g) + offset).tolist()
            out += got[:want]
            offset += n
        out = [out[i] for i in torch.randperm(len(out), generator=g).tolist()]
        out = out[: len(out) - len(out) % self.world] if len(out) >= self.world else out
        return out[self.rank::self.world]

    def __iter__(self):
        return iter(self.indices()[self.start:])

    def __len__(self) -> int:
        return max(0, len(self.indices()) - self.start)


def build_datasets(cfg: Dict, tok, laya_cfg: Dict, n_img: int, with_eval: bool = True):
    from ..data.dataset import DecisionDataset

    d = cfg["data"]
    max_len = d.get("max_len") or laya_cfg["max_len"]
    head_max_len = d.get("head_max_len") or laya_cfg["head_max_len"]

    def make(src, augment, prescan, seed):
        return DecisionDataset(src["files"], src.get("image_root", d["image_root"]), tok, max_len, head_max_len,
                               n_img, augment=augment, seed=seed, prescan=prescan)

    augment = None
    if d.get("augment"):
        from ..data.augment import Augmenter

        augment = Augmenter(**(d["augment"] if isinstance(d["augment"], dict) else {}))
    train_src = _sources(d["train"])
    if not train_src:
        raise ValueError("data.train is empty")
    train = [make(s, augment, bool(d.get("prescan_train")), cfg["seed"] + i) for i, s in enumerate(train_src)]
    weights = [float(s.get("weight", 1.0)) for s in train_src] if any("weight" in s for s in train_src) else None
    # eval sets are pre-scanned (exact len, unfit rows logged); only rank 0 evaluates
    evals = [make(s, None, True, cfg["seed"]) for s in _sources(d["eval"])] if with_eval else []
    return train, weights, evals


def make_loader(datasets, sampler, batch_size: int, collate, num_workers: int, seed: int):
    ds = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collate,
                                       num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=False, drop_last=False,
                                       generator=torch.Generator().manual_seed(seed))


def eval_subset(datasets, max_items: Optional[int], seed: int):
    ds = torch.utils.data.ConcatDataset(datasets)
    idx = list(range(len(ds)))
    if max_items and len(idx) > max_items:
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:max_items])
    return torch.utils.data.Subset(ds, idx)


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(model, b: Dict, detach_encoder: bool = False):
    kw = {}
    if b.get("pixel_values") is not None:
        kw = {"pixel_values": b["pixel_values"], "image_index": b["image_index"], "image_start": b["image_start"]}
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                 detach_encoder=detach_encoder, **kw)


# ----------------------------------------------------------------------------- synthetic toy data

_COLORS = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 60, 220), "yellow": (230, 220, 30)}


def make_synthetic_data(out_dir: str, n_train: int = 96, n_eval: int = 48, n_calib: int = 48,
                        seed: int = 0, image_size: int = 32) -> Dict[str, str]:
    """Toy colour task in the data JSONL format: choice / noul / score rows with and without images.

    Writes ``images/*.png`` and ``{train,eval,calib}.jsonl`` under ``out_dir`` (image paths relative
    to ``out_dir``). Idempotent: existing files are reused.
    """
    from PIL import Image

    paths = {s: os.path.join(out_dir, "%s.jsonl" % s) for s in ("train", "eval", "calib")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(_COLORS)
    for split, n in (("train", n_train), ("eval", n_eval), ("calib", n_calib)):
        with open(paths[split], "w") as f:
            for i in range(n):
                c = names[i % len(names)]
                bright = int(rng.integers(0, 3))
                arr = np.clip(np.array(_COLORS[c]) * (0.4 + 0.3 * bright)
                              + rng.normal(0, 12, (image_size, image_size, 3)), 0, 255).astype(np.uint8)
                rel = "images/%s_%04d.png" % (split, i)
                Image.fromarray(arr).save(os.path.join(out_dir, rel))
                opts = {k: "the image is mostly %s" % k for k in names}
                rows = [
                    ({"type": "choice", "instructions": "Which colour dominates?", "criteria": opts},
                     [float(k == c) for k in names]),
                    ({"type": "noul", "instructions": "Is the image %s?" % names[(i // 4) % 4]},
                     [float(names[(i // 4) % 4] != c), float(names[(i // 4) % 4] == c)]),
                    ({"type": "score", "instructions": "How bright is the image?",
                      "criteria": ["dark", "medium", "bright"]}, [float(bright == j) for j in range(3)]),
                ]
                for j, (q, t) in enumerate(rows):
                    text_row = i % 4 == 3
                    rec = {"id": "synth/%s/%04d/q%d" % (split, i, j), "task": "synthetic_colour", "split": split,
                           "image": None if text_row else rel,
                           "text": ("A %s picture." % c) if text_row else None, "question": q, "target": t}
                    f.write(json.dumps(rec) + "\n")
    return paths


# ----------------------------------------------------------------------------- model

def build_model(cfg: Dict, out_dir: str, rank: int):
    """``(model, tok, laya_cfg, vcfg, resumed_state_or_None)``."""
    from ..checkpoint import init_from_laya, load_checkpoint

    resume = cfg.get("resume")
    latest = os.path.join(out_dir, "checkpoint_latest")
    if resume == "auto":
        resume = latest if os.path.exists(os.path.join(latest, "trainer_state.pt")) else None
    if resume:
        model, tok, laya_cfg, vcfg = load_checkpoint(resume, device="cpu")
        state = torch.load(os.path.join(resume, "trainer_state.pt"), map_location="cpu", weights_only=False)
        if rank == 0:
            print("[train] resuming from %s (step %d)" % (resume, state["step"]))
        return model, tok, laya_cfg, vcfg, state
    if cfg.get("init_from"):
        model, tok, laya_cfg, vcfg = load_checkpoint(cfg["init_from"], device="cpu")
        return model, tok, laya_cfg, vcfg, None

    v = dict(cfg["vision"])
    tower_config = None
    if v.get("tower") == "tiny":
        from ..testing import tiny_siglip_config

        tower_config = tiny_siglip_config()
    base = cfg["base"]
    if base == "tiny":
        from ..testing import make_tiny_laya_dir

        base = os.path.join(out_dir, "_tiny_laya")
        if rank == 0 and not os.path.exists(os.path.join(base, "model.safetensors")):
            make_tiny_laya_dir(base, seed=cfg["seed"])
        barrier()
    vcfg = VisionConfig.from_dict(v)
    model, tok, laya_cfg = init_from_laya(base, vcfg, tower_config=tower_config, revision=cfg.get("base_revision"))
    return model, tok, laya_cfg, getattr(model, "vcfg", vcfg), None


def _vision_blocks(vision: torch.nn.Module):
    """The tower's transformer blocks (``...encoder.layers`` in HF SigLIP)."""
    best = None
    for name, m in vision.named_modules():
        if isinstance(m, torch.nn.ModuleList) and name.endswith("layers"):
            best = m
    return best


def setup_trainable(model, tcfg: Dict, vcfg) -> List[Dict[str, Any]]:
    """Set ``requires_grad`` per group and return AdamW param groups (frozen = lr 0/None)."""
    from .lora import apply_lora

    lrs = tcfg["lr"]
    for p in model.parameters():
        p.requires_grad_(False)
    groups = model.trainable_groups()
    param_groups = []
    for g in GROUPS:
        lr = float(lrs.get(g) or 0.0)
        if lr <= 0:
            continue
        params = list(groups[g])
        if g == "vision":
            n = int(tcfg.get("unfreeze_vision_blocks") or 0)
            blocks = _vision_blocks(model.vision)
            if n > 0 and blocks is not None:
                keep = {id(p) for b in list(blocks)[-n:] for p in b.parameters()}
                params = [p for p in params if id(p) in keep]
            vcfg.tower_trained = True
        if g == "encoder" and tcfg["lora"].get("enabled"):
            lc = tcfg["lora"]
            params = apply_lora(model.laya.encoder, r=lc["r"], alpha=lc["alpha"], dropout=lc["dropout"],
                                targets=lc["targets"])
        for p in params:
            p.requires_grad_(True)
        param_groups.append({"params": params, "lr": lr, "name": g})
    if not param_groups:
        raise ValueError("nothing to train: every train.lr entry is 0")
    return param_groups


def lr_lambda(total: int, warmup_ratio: float, min_ratio: float):
    warm = int(round(warmup_ratio * total))

    def f(step: int) -> float:
        if warm and step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ----------------------------------------------------------------------------- eval / save

@torch.no_grad()
def collect_predictions(model, loader, device, amp_dtype) -> List[Dict[str, Any]]:
    """Raw logits per row: ``{id, task, modality, qtype (int), logits [k], target [k]}``."""
    was = model.training
    model.eval()
    rows = []
    for b in loader:
        if b is None:  # every item in the batch was dropped by the dataset (does not fit)
            continue
        b = to_device(b, device)
        with autocast(device, amp_dtype):
            logits, _ = forward_batch(model, b)
        logits = logits.float().cpu()
        k = b["marker_mask"].sum(-1).cpu()
        img = b["image_start"].cpu() if torch.is_tensor(b.get("image_start")) else torch.full((len(k),), -1)
        target = b["target"].cpu()
        for i in range(len(k)):
            meta = b["meta"][i] if "meta" in b else {}
            ki = int(k[i])
            rows.append({"id": meta.get("id"), "task": meta.get("task"),
                         "modality": "image" if int(img[i]) >= 0 else "text", "qtype": int(b["qtype"][i]),
                         "logits": logits[i, :ki].tolist(), "target": target[i, :ki].tolist()})
    model.train(was)
    return rows


def evaluate(model, loader, device, amp_dtype) -> Dict[str, Any]:
    from ..eval.metrics import summarize

    rows = collect_predictions(model, loader, device, amp_dtype)
    for r in rows:
        z = torch.tensor(r["logits"])
        r["probs"] = torch.softmax(z, -1).tolist()
        r["qtype_name"] = QTYPE_NAMES[r["qtype"]]
    out: Dict[str, Any] = {}
    for mod in ("image", "text"):
        sel = [dict(r, qtype=r["qtype_name"]) for r in rows if r["modality"] == mod]
        if not sel:
            continue
        s = summarize(sel)
        zs = [z for r in sel for z in r["logits"]]
        s["logit_mean"] = float(np.mean(zs))
        s["logit_std"] = float(np.std(zs))
        s["logit_max_mean"] = float(np.mean([max(r["logits"]) for r in sel]))
        out[mod] = s
    return out


def _metric(metrics: Dict, key: str) -> Optional[float]:
    d: Any = metrics
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return float(d)


def save_all(model, tok, laya_cfg, vcfg, out: str, trainer_state: Optional[Dict] = None) -> None:
    """Atomic ``save_checkpoint`` (LoRA merged) + optional ``trainer_state.pt``."""
    from ..checkpoint import save_checkpoint
    from .lora import merged_copy

    tmp = out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    save_checkpoint(merged_copy(model), tok, laya_cfg, vcfg, tmp)
    if trainer_state is not None:
        torch.save(trainer_state, os.path.join(tmp, "trainer_state.pt"))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)


# ----------------------------------------------------------------------------- train

def train(cfg: Dict) -> Dict[str, Any]:
    is_dist, rank, world, local_rank = dist_info()
    device = pick_device(cfg.get("device", "auto"), local_rank)
    if is_dist and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    main = rank == 0
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    tcfg = cfg["train"]
    seed_everything(cfg["seed"])
    if main:
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    syn = cfg["data"].get("synthetic")
    if syn:
        sdir = syn.get("dir") or os.path.join(out_dir, "synthetic")
        if main:
            make_synthetic_data(sdir, syn.get("n_train", 96), syn.get("n_eval", 48), syn.get("n_calib", 48),
                                seed=cfg["seed"])
        barrier()
        cfg["data"].update(train=[os.path.join(sdir, "train.jsonl")], eval=[os.path.join(sdir, "eval.jsonl")],
                           image_root=sdir)

    model, tok, laya_cfg, vcfg, state = build_model(cfg, out_dir, rank)
    param_groups = setup_trainable(model, tcfg, vcfg)
    if tcfg.get("gradient_checkpointing"):
        model.laya.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.laya.head_checkpointing = True
    model.to(device)
    model.train()
    amp_dtype, use_scaler = amp_setup(tcfg.get("amp", "auto"), device)

    from ..data.dataset import make_collate

    train_sets, weights, eval_sets = build_datasets(cfg, tok, laya_cfg, model.n_image_tokens, with_eval=main)
    has_eval = bool(_sources(cfg["data"]["eval"]))
    collate = make_collate(tok.pad_token_id, vcfg.image_size)
    sampler = EpochSampler([len(d) for d in train_sets], weights, cfg["data"].get("samples_per_epoch"),
                           seed=cfg["seed"], rank=rank, world=world)
    loader = make_loader(train_sets, sampler, tcfg["micro_batch"], collate, cfg["data"]["num_workers"], cfg["seed"])
    eval_loader = None
    if eval_sets:
        eval_loader = torch.utils.data.DataLoader(
            eval_subset(eval_sets, cfg["data"].get("max_eval_items"), cfg["seed"]),
            batch_size=tcfg["micro_batch"] * 2, shuffle=False, collate_fn=collate,
            num_workers=cfg["data"]["num_workers"])

    accum = int(tcfg["grad_accum"])
    micro_per_epoch = math.ceil(len(sampler.indices()) / tcfg["micro_batch"])
    steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum))
    total = int(tcfg.get("max_steps") or steps_per_epoch * tcfg["epochs"])
    epochs = math.ceil(total / steps_per_epoch)

    optimizer = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in param_groups],
                                  weight_decay=float(tcfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(total, float(tcfg["warmup_ratio"]), float(tcfg["min_lr_ratio"])))
    scaler = make_scaler(use_scaler)

    step, start_epoch, start_micro = 0, 0, 0
    best, text_base, bad_evals = None, None, 0
    if state is not None:
        if tcfg["lora"].get("enabled"):
            print("[train] LoRA resume: checkpoint is merged; adapters restart from zero, optimizer state reset")
        else:
            optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step, start_epoch, start_micro = state["step"], state["epoch"], state["micro_in_epoch"]
        best, text_base, bad_evals = state.get("best"), state.get("text_base"), state.get("bad_evals", 0)
    seed_everything(cfg["seed"] + 1000 * rank + step)

    ddp_model = model
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # find_unused: text-only batches leave the projector unused, feature_layer -2 the last tower block.
        ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                        find_unused_parameters=True)

    logger = Logger(out_dir, main, cfg.get("wandb"), cfg)
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    if main:
        print("[train] stage %s | device %s x%d | amp %s | trainable %.2fM (%s) | %d rows/epoch | "
              "%d steps (%d/epoch) | effective batch %d"
              % (cfg["stage"], device, world, amp_dtype, n_train / 1e6, ",".join(g["name"] for g in param_groups),
                 len(sampler.indices()) * world, total, steps_per_epoch, tcfg["micro_batch"] * accum * world),
              flush=True)

    def run_eval() -> bool:
        """Evaluate, track best / early stop. Returns True to stop."""
        nonlocal best, text_base, bad_evals
        stop = False
        if main and eval_loader is not None:
            m = evaluate(model, eval_loader, device, amp_dtype)
            logger.log({"event": "eval", "step": step, **{"%s_%s" % (mod, k): v for mod, s in m.items()
                                                           for k, v in s.items() if not isinstance(v, (list, dict))}})
            val = _metric(m, tcfg["best_metric"])
            if val is not None and step > 0 and (best is None or (val > best if tcfg["best_mode"] == "max" else val < best)):
                best = val
                save_all(model, tok, laya_cfg, vcfg, os.path.join(out_dir, "best"))
                logger.log({"event": "best", "step": step, tcfg["best_metric"]: val})
            text_acc = _metric(m, "text.accuracy")
            if text_acc is not None:
                if text_base is None:
                    text_base = text_acc
                drop = tcfg.get("early_stop_text_drop")
                if drop is not None and text_base - text_acc > float(drop):
                    bad_evals += 1
                    stop = bad_evals >= int(tcfg.get("early_stop_patience", 1))
                    logger.log({"event": "text_regression", "step": step, "base": text_base, "now": text_acc,
                                "stop": stop})
                else:
                    bad_evals = 0
        if is_dist:
            flag = torch.tensor([int(stop)], device=device)
            dist.broadcast(flag, 0)
            stop = bool(flag.item())
        return stop

    def checkpoint_state(epoch: int, micro: int) -> Dict:
        return {"step": step, "epoch": epoch, "micro_in_epoch": micro, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if use_scaler else None,
                "best": best, "text_base": text_base, "bad_evals": bad_evals, "config": cfg}

    # every rank calls run_eval (rank 0 evaluates, the others wait on the stop-flag broadcast)
    if state is None and tcfg.get("eval_at_start", True) and has_eval:
        run_eval()

    params = [p for g in param_groups for p in g["params"]]
    stopped, window, t0, grads_in_step = False, [], time.time(), False
    pos = (start_epoch, start_micro)  # (epoch, micro-batches done in it) after the last optimizer step
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        skip = start_micro if epoch == start_epoch else 0
        sampler.set_epoch(epoch, skip * tcfg["micro_batch"])
        for d in train_sets:
            d.set_epoch(epoch)
        n_micro = skip + len(loader)
        for i, batch in enumerate(loader, start=skip):
            if step >= total:
                break
            have = batch is not None
            if is_dist:  # all ranks must skip together, or DDP's gradient all-reduce would hang
                flag = torch.tensor([int(have)], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                have = bool(flag.item())
            last_micro = (i + 1) % accum == 0 or i + 1 == n_micro
            if have:
                b = to_device(batch, device)
                sigma = sigma_at(step / max(1, total - 1), tcfg["sigma_start"], tcfg["sigma_end"])
                sync = contextlib.nullcontext() if (not is_dist or last_micro) else ddp_model.no_sync()
                with sync:
                    with autocast(device, amp_dtype):
                        logits, act = forward_batch(ddp_model, b, tcfg.get("detach_encoder", False))
                    loss, stats = rlcd_loss(logits, b["target"], b["marker_mask"], b["qtype"], sigma,
                                            group_size=tcfg["group_size"], w_sph=tcfg["w_sph"], w_rps=tcfg["w_rps"],
                                            ce_weight=tcfg["ce_weight"])
                    # notebook: loss / GRAD_ACCUM (+ 0 * act so DDP sees the act head used)
                    n_acc = accum if (i // accum + 1) * accum <= n_micro else n_micro - (i // accum) * accum
                    scaler.scale(loss / n_acc + 0.0 * act.float().sum()).backward()
                window.append(stats)
                grads_in_step = True
                if not math.isfinite(stats["loss"]):
                    raise FloatingPointError("non-finite loss at step %d: %s" % (step, stats))
            if not last_micro or not grads_in_step:
                continue
            grads_in_step = False
            if is_dist and not have:  # the synced last micro-batch was skipped: average grads by hand
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad)
                    p.grad /= world
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(params, float(tcfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            pos = (epoch, i + 1)
            if window and (step % tcfg["log_every"] == 0 or step == 1 or step == total):
                avg = {k: float(np.mean([s[k] for s in window])) for k in window[0]}
                logger.log({"event": "train", "step": step, "epoch": epoch, **avg, "grad_norm": float(gnorm),
                            **{"lr_%s" % g["name"]: optimizer.param_groups[j]["lr"]
                               for j, g in enumerate(param_groups)},
                            "sec": round(time.time() - t0, 1)})
                window = []
            if step % tcfg["eval_every"] == 0 and step < total:
                stopped = run_eval()
            if (step % tcfg["save_every"] == 0 or stopped) and main:
                save_all(model, tok, laya_cfg, vcfg, os.path.join(out_dir, "checkpoint_latest"), checkpoint_state(*pos))
            if stopped:
                break
        if stopped or step >= total:
            break

    if not stopped:
        stopped = run_eval()
    if main:
        save_all(model, tok, laya_cfg, vcfg, os.path.join(out_dir, "checkpoint_latest"), checkpoint_state(*pos))
        save_all(model, tok, laya_cfg, vcfg, os.path.join(out_dir, "final"))
        logger.log({"event": "done", "step": step, "early_stopped": stopped, "best": best,
                    "sec": round(time.time() - t0, 1)})
    barrier()
    if is_dist:
        dist.destroy_process_group()
    return {"step": step, "best": best, "early_stopped": stopped, "output_dir": out_dir}


def cli(stage: int, stage_defaults: Dict, argv=None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="laya_vision stage %d training" % stage)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", nargs="*", default=[], help="dotted key=value, e.g. train.max_steps=100")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override, stage_defaults)
    cfg["stage"] = stage
    return train(cfg)
