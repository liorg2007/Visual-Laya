"""Build, save and load Laya-Vision checkpoints (plan.md §2.9).

Layout::

    rl_agent_config.json   stock Laya config + "vision": VisionConfig.to_dict()
    model.safetensors      DecisionModel keys only (no "laya." prefix) -> stock laya.Agent loads it
    vision.safetensors     pooler.* / projector.* / modality_emb (+ vision.* when the tower is saved)
    encoder/ tokenizer/    as in Laya
    vision_tower/          config.json of the tower, present only when the tower is saved

Tower rule: the tower weights are saved (``vision.*`` in vision.safetensors, config in
``vision_tower/``) when ``vcfg.tower_trained`` is true or the tower was not loaded from the Hub
(built from a config, e.g. tiny test towers). Otherwise the tower is re-fetched from
``vcfg.tower`` at ``vcfg.tower_revision``. On load, the presence of ``vision_tower/config.json``
decides.
"""
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from .config import VisionConfig
from .model import LayaVisionModel, build_vision_side
from .vision import VisionEncoder

TOWER_DIR = "vision_tower"
_LAYA_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
_VISION_FILES = ["vision.safetensors", TOWER_DIR + "/*"]


def resolve_dir(path_or_hub_id: str, revision: Optional[str] = None, vision: bool = True,
                token: Optional[str] = None) -> str:
    """Local directory for a checkpoint path or Hub id (downloads only checkpoint files)."""
    if os.path.isdir(path_or_hub_id):
        return path_or_hub_id
    if path_or_hub_id.startswith(("/", "./", "../")) or os.path.isabs(path_or_hub_id):
        raise FileNotFoundError("Local checkpoint path not found: %r" % path_or_hub_id)
    from huggingface_hub import snapshot_download

    kw = {"allow_patterns": _LAYA_FILES + (_VISION_FILES if vision else [])}
    if revision:
        kw["revision"] = revision
    if token or os.environ.get("HF_TOKEN"):
        kw["token"] = token or os.environ.get("HF_TOKEN")
    return snapshot_download(path_or_hub_id, **kw)


def _load_laya(model_dir: str):
    """(DecisionModel, tok, cfg) exactly the way ``laya.Agent.__init__`` loads them."""
    from safetensors.torch import load_file
    from laya.agent import _fix_tokenizer_config, _load_tokenizer
    from laya.common import build_model

    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = _load_tokenizer(os.path.join(model_dir, "tokenizer"), cfg)
    enc_dir = os.path.join(model_dir, "encoder")
    laya = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None, pretrained=False)
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    laya.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
    laya.encoder.config.reference_compile = False
    return laya, tok, cfg


def _laya_version() -> Optional[str]:
    try:
        import laya

        return laya.__version__
    except Exception:
        return None


def build_tower(vcfg: VisionConfig, tower_config=None) -> VisionEncoder:
    if tower_config is not None:
        return VisionEncoder.from_config(tower_config, feature_layer=vcfg.feature_layer)
    return VisionEncoder.from_pretrained(vcfg.tower, revision=vcfg.tower_revision, feature_layer=vcfg.feature_layer)


def init_from_laya(laya_id_or_dir: str, vcfg: VisionConfig, tower_config=None,
                   revision: Optional[str] = None) -> Tuple[LayaVisionModel, Any, Dict[str, Any]]:
    """A fresh LayaVisionModel from a Laya checkpoint plus a SigLIP tower (random projector).

    ``tower_config`` (a ``SiglipVisionConfig``) builds a randomly initialised tower instead of
    downloading ``vcfg.tower``; such a tower is always saved with the checkpoint.
    ``vcfg`` is updated in place with the tower's image_size / patch_size / width.
    """
    model_dir = resolve_dir(laya_id_or_dir, revision, vision=False)
    laya, tok, cfg = _load_laya(model_dir)
    vision = build_tower(vcfg, tower_config)
    vision._laya_vision_saved_tower = tower_config is not None
    pooler, projector = build_vision_side(vision, vcfg, laya.encoder.config.hidden_size)
    model = LayaVisionModel(laya, vision, pooler, projector)
    if vcfg.laya_base is None:
        from laya.revisions import snapshot_revision

        rev = snapshot_revision(model_dir) if not os.path.isdir(laya_id_or_dir) else None
        vcfg.laya_base = laya_id_or_dir + ("@" + rev if rev else "")
    if vcfg.laya_version is None:
        vcfg.laya_version = _laya_version()
    cfg = dict(cfg)
    cfg["vision"] = vcfg.to_dict()
    return model, tok, cfg


def _save_tower(model: LayaVisionModel, vcfg: VisionConfig) -> bool:
    return bool(vcfg.tower_trained or getattr(model.vision, "_laya_vision_saved_tower", False))


def save_checkpoint(model: LayaVisionModel, tok, cfg: Dict[str, Any], vcfg: VisionConfig, out_dir: str,
                    dtype: torch.dtype = torch.float16) -> str:
    """Write ``out_dir`` in the layout above. ``dtype`` applies to float tensors (fp16 like Laya)."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)

    def cast(sd):
        return {k: (v.detach().to("cpu", dtype) if v.is_floating_point() else v.detach().cpu()).contiguous()
                for k, v in sd.items()}

    save_file(cast(model.laya.state_dict()), os.path.join(out_dir, "model.safetensors"))
    vsd = {"modality_emb": model.modality_emb}
    vsd.update({"pooler." + k: v for k, v in model.pooler.state_dict().items()})
    vsd.update({"projector." + k: v for k, v in model.projector.state_dict().items()})
    tower_saved = _save_tower(model, vcfg)
    if tower_saved:
        vsd.update({"vision." + k: v for k, v in model.vision.state_dict().items()})
        model.vision.config.save_pretrained(os.path.join(out_dir, TOWER_DIR))
    save_file(cast(vsd), os.path.join(out_dir, "vision.safetensors"))

    model.laya.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg = dict(cfg)
    cfg["vision"] = vcfg.to_dict()
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return out_dir


def build_vision_from_dir(model_dir: str, vcfg: VisionConfig, laya, device="cpu") -> LayaVisionModel:
    """Attach the vision side stored in ``model_dir`` to an already-loaded ``DecisionModel``."""
    from safetensors.torch import load_file

    tower_dir = os.path.join(model_dir, TOWER_DIR)
    saved_tower = os.path.isfile(os.path.join(tower_dir, "config.json"))
    if saved_tower:
        from transformers import SiglipVisionConfig

        vision = VisionEncoder.from_config(SiglipVisionConfig.from_pretrained(tower_dir),
                                           feature_layer=vcfg.feature_layer)
    else:
        vision = build_tower(vcfg)
    vision._laya_vision_saved_tower = saved_tower
    pooler, projector = build_vision_side(vision, vcfg, laya.encoder.config.hidden_size)
    model = LayaVisionModel(laya, vision, pooler, projector)

    sd = load_file(os.path.join(model_dir, "vision.safetensors"))
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    expected = {"modality_emb"} | {"pooler." + k for k in pooler.state_dict()} \
        | {"projector." + k for k in projector.state_dict()}
    if saved_tower or any(k.startswith("vision.") for k in sd):
        expected |= {"vision." + k for k in vision.state_dict()}
    missing, unexpected = expected - set(sd), set(sd) - expected
    if missing or unexpected:
        raise ValueError("vision.safetensors does not match the vision config: missing %s, unexpected %s"
                         % (sorted(missing)[:5], sorted(unexpected)[:5]))
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def load_checkpoint(path_or_hub_id: str, device="cpu", revision: Optional[str] = None):
    """``(model, tok, cfg, vcfg)`` from a checkpoint dir or Hub id; weights in fp32, eval mode."""
    model_dir = resolve_dir(path_or_hub_id, revision)
    laya, tok, cfg = _load_laya(model_dir)
    if "vision" not in cfg:
        raise ValueError("%r is a text-only Laya checkpoint (no 'vision' block in rl_agent_config.json)"
                         % path_or_hub_id)
    vcfg = VisionConfig.from_dict(cfg["vision"])
    model = build_vision_from_dir(model_dir, vcfg, laya, device)
    return model, tok, cfg, vcfg
