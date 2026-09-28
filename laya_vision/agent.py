"""``VisionAgent``: ``laya.Agent`` that also accepts ``{"image": ..., "text": ...}`` states (plan.md §2.10).

Only the per-state encode, the forward and the per-state decode are overridden; ``predict_batch``
(hooks, batching, usage, ``min_confidence``) is Laya's own. A request with no image state takes
the stock methods end to end, so its output is exactly ``laya.Agent``'s.

Image rows: one vision forward per distinct image (sha256) per forward pass, then Laya's decode
with the image temperatures from ``cfg["vision"]`` (text temperatures when those are unset).
Unsupported with an image state: ``fast=True``, ``compile=True``, ``predict_long``,
``laya.shortlist.predict_shortlist`` and ONNX.
"""
import os
import types
from typing import Any, Dict, List, Optional, Union

import torch

from laya.agent import Agent, _amp_context
from laya.common import clamp_temperature

from .checkpoint import TOWER_DIR, build_vision_from_dir
from .config import VisionConfig
from .images import load_image_and_hash, preprocess
from .sequence import _state_token_ids, build_item, is_image_state, split_state

_CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
                     "vision.safetensors", TOWER_DIR + "/*")


def _is_image_row(item: Dict[str, Any]) -> bool:
    return int(item.get("image_start", -1)) >= 0


class VisionAgent(Agent):
    """Laya-Vision runtime. Takes every ``laya.Agent`` argument; see the module docstring."""

    def __init__(self, model_id_or_path: str, device: Optional[str] = None, token: Optional[str] = None,
                 subfolder: Optional[str] = None, revision: Optional[str] = None, **kw):
        model_dir, snapshot_rev = self._resolve(model_id_or_path, token, subfolder, revision)
        super().__init__(model_dir, device=device, token=token, **kw)
        self.model_id = model_id_or_path
        self.revision = snapshot_rev
        self.model_dir = model_dir

        if "vision" not in self.cfg:
            raise ValueError("%r has no 'vision' block in rl_agent_config.json; load it with laya.Agent"
                             % model_id_or_path)
        self.vcfg = VisionConfig.from_dict(self.cfg["vision"])
        laya_model = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
        enc_cfg = laya_model.encoder.config
        keep_compile = getattr(enc_cfg, "reference_compile", None)
        self.vlm = build_vision_from_dir(model_dir, self.vcfg, laya_model, device=self.device)
        enc_cfg.reference_compile = keep_compile  # the text path keeps whatever Agent chose

        vt = self.vcfg.temperature if self.vcfg.temperature is not None else self.temperature
        if len(vt) != 3:
            raise ValueError("vision temperature must be a list of 3 floats, got %r" % (vt,))
        vtbo = (self.vcfg.temperature_by_options if self.vcfg.temperature is not None
                or self.vcfg.temperature_by_options else self.temperature_by_options)
        # `_decode_answers` reads only these three attributes, so Laya's decode runs unchanged on it.
        self._image_temps = types.SimpleNamespace(
            temperature=[clamp_temperature(t) for t in vt],
            temperature_by_options={k: clamp_temperature(v) for k, v in vtbo.items()},
            lang_temperatures={},
        )

    @staticmethod
    def _resolve(model_id_or_path, token, subfolder, revision):
        if os.path.exists(model_id_or_path):
            path = os.path.join(model_id_or_path, subfolder) if subfolder else model_id_or_path
            return path, None
        if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
            raise FileNotFoundError("Local model path not found: %r" % model_id_or_path)
        from huggingface_hub import snapshot_download
        from laya.revisions import snapshot_revision

        prefix = subfolder + "/" if subfolder else ""
        kw = {"allow_patterns": [prefix + p for p in _CHECKPOINT_FILES]}
        if revision:
            kw["revision"] = revision
        if token or os.environ.get("HF_TOKEN"):
            kw["token"] = token or os.environ.get("HF_TOKEN")
        root = snapshot_download(model_id_or_path, **kw)
        return (os.path.join(root, subfolder) if subfolder else root), snapshot_revision(root) or revision

    @property
    def n_image_tokens(self) -> int:
        return self.vlm.n_image_tokens

    def _encode_state(self, state: Union[str, dict, list], ids: List[str], internal: Dict[str, Dict],
                      max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> List[Dict]:
        if not is_image_state(state):
            return super()._encode_state(state, ids, internal, max_len=max_len, head_max_len=head_max_len)
        max_len = self.cfg.get("max_len", 512) if max_len is None else max_len
        head_max_len = self.cfg.get("head_max_len", 192) if head_max_len is None else head_max_len
        image, text = split_state(state)
        img, sha = load_image_and_hash(image)
        pixel = preprocess([img], self.vcfg.image_size)[0]
        text_ids = _state_token_ids(self.tok, text) if text is not None and text != "" else None
        items = []
        for qid in ids:
            try:
                item = build_item(self.tok, state, internal[qid], self.n_image_tokens, max_len, head_max_len,
                                  state_ids=text_ids)
            except ValueError as e:
                raise ValueError("question %r: %s" % (qid, e)) from None
            # Shared by reference across the state's rows; collate_items carries both into b["meta"].
            item["image_ref"] = sha
            item["pixel_values"] = pixel
            items.append(item)
        return items

    def _check_image_path(self):
        if self._fast is not None:
            raise ValueError("image states are not supported on the TileLang fast path; "
                             "load without fast=True (or call deaccelerate()) to send images")
        if self._compiled:
            raise ValueError("image states are not supported with compile=True; load without it to send images")

    def _image_batch(self, meta: List[Dict[str, Any]]):
        """``(pixel_values [M,3,S,S], image_index [B], image_start [B])`` with one entry per distinct image."""
        refs: Dict[Any, int] = {}
        pixels, index, start = [], [], []
        for m in meta:
            if not _is_image_row(m):
                index.append(-1)
                start.append(-1)
                continue
            if m["image_ref"] not in refs:
                refs[m["image_ref"]] = len(pixels)
                pixels.append(m["pixel_values"])
            index.append(refs[m["image_ref"]])
            start.append(int(m["image_start"]))
        return torch.stack(pixels), torch.tensor(index), torch.tensor(start)

    def _forward(self, b: Dict):
        if not any(_is_image_row(m) for m in b["meta"]):
            return super()._forward(b)
        self._check_image_path()
        pixel_values, image_index, image_start = self._image_batch(b["meta"])
        dev = self.device
        if self.vlm.modality_emb.device != dev:  # e.g. after Laya's scoped CPU fallback moved self.model
            self.vlm.to(dev)
        with _amp_context(dev, self.dtype, self._amp_enabled_for(b["input_ids"].shape[0])):
            tokens = self.vlm.encode_images(pixel_values.to(dev))
            logits, act = self.vlm(
                b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                b["marker_mask"].to(dev), b["qtype"].to(dev),
                image_tokens=tokens, image_index=image_index.to(dev), image_start=image_start.to(dev))
        return logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()

    def _decode_answers(self, logits, act, items: List[Dict], ids: List[str],
                        internal: Dict[str, Dict], offset: int, lang: Optional[str] = None) -> Dict[str, Any]:
        # One call covers one state, so its rows are all image rows or all text rows.
        if items and _is_image_row(items[0]):
            return Agent._decode_answers(self._image_temps, logits, act, items, ids, internal, offset)
        return super()._decode_answers(logits, act, items, ids, internal, offset,
                                       **({"lang": lang} if lang else {}))

    def predict_long(self, state, questions, *args, **kwargs):
        if is_image_state(state):
            raise ValueError("predict_long does not support image states; use predict()")
        return super().predict_long(state, questions, *args, **kwargs)

    def __repr__(self) -> str:
        return "VisionAgent(model_id=%r, device=%s)" % (self.model_id, getattr(self, "device", None))


def load(model_id_or_path: str, device: Optional[str] = None, **kw) -> VisionAgent:
    """Load a Laya-Vision checkpoint (directory or Hub id). Keyword arguments go to ``laya.Agent``."""
    return VisionAgent(model_id_or_path, device=device, **kw)
