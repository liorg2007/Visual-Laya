"""Baselines to beat (ARCH §6, plan.md §4 Phase 3 item 6).

Every runner has ``predict(state, questions) -> {"model", "answers", "usage"}`` with answers in
Laya's schema, so ``run_eval`` and ``metrics`` treat them exactly like ``VisionAgent``. A question a
runner cannot answer is left out of ``answers`` and is reported as skipped.

- ``CaptionLaya``: caption (+ optional OCR) the image, then ask stock ``laya.Agent`` about the text.
- ``SiglipZeroShot``: image-text similarity against ``"a photo of {option}"``, ``choice`` only.
- ``StockLaya``: stock ``laya.Agent`` on text-only states (text regression baseline).
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from laya.common import answer_confidence, confidence_from_probs
from laya_vision.images import load_image, load_image_and_hash
from laya_vision.sequence import split_state

DEFAULT_CAPTIONER = "Salesforce/blip-image-captioning-base"
DEFAULT_SIGLIP = "google/siglip-base-patch16-224"


def choice_answer(keys: List[str], p: np.ndarray) -> Dict[str, Any]:
    """A Laya-schema ``choice`` answer from a probability vector."""
    k = len(keys)
    return {"type": "choice", "choice": keys[int(p.argmax())],
            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
            "confidence": round(confidence_from_probs(p, k), 4),
            "answer_confidence": round(answer_confidence(p, k), 4)}


def _result(model: str, answers: Dict[str, Any]) -> Dict[str, Any]:
    return {"model": model, "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


class BlipCaptioner:
    """Pinned transformers image captioner (BLIP by default). ``__call__(PIL) -> str``."""

    def __init__(self, model_id: str = DEFAULT_CAPTIONER, revision: Optional[str] = None,
                 device: Optional[str] = None, max_new_tokens: int = 30, num_beams: int = 3):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.model_id, self.revision = model_id, revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_id, revision=revision)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.gen = {"max_new_tokens": max_new_tokens, "num_beams": num_beams}

    def __call__(self, img) -> str:
        import torch

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self.gen)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class CaptionLaya:
    """Caption → stock Laya. Text-only states go to Laya unchanged.

    The text state is the caption string, or ``{"image_caption", "ocr_text"?, "text"?}`` when
    OCR or a record text is present. Captions are cached on disk by image sha256.
    """

    name = "caption_laya"

    def __init__(self, laya_agent, captioner: Optional[Callable] = None, cache_dir: Optional[str] = None,
                 ocr: Optional[Callable] = None, captioner_id: str = DEFAULT_CAPTIONER,
                 captioner_revision: Optional[str] = None, device: Optional[str] = None):
        self.laya = laya_agent
        self._captioner = captioner
        self._cap_kw = {"model_id": captioner_id, "revision": captioner_revision, "device": device}
        self.captioner_id = captioner_id if captioner is None else getattr(captioner, "model_id", "custom")
        self.ocr = ocr
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", self.captioner_id))
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def captioner(self):
        if self._captioner is None:
            self._captioner = BlipCaptioner(**self._cap_kw)
        return self._captioner

    def caption(self, image: Any) -> str:
        img, sha = load_image_and_hash(image)
        path = os.path.join(self.cache_dir, sha + ".json") if self.cache_dir else None
        if path and os.path.isfile(path):
            with open(path) as f:
                return json.load(f)["caption"]
        cap = self.captioner(img)
        if path:
            with open(path, "w") as f:
                json.dump({"caption": cap, "captioner": self.captioner_id}, f)
        return cap

    def text_state(self, state: Any) -> Any:
        image, text = split_state(state)
        if image is None:
            return state
        cap = self.caption(image)
        ocr = self.ocr(load_image(image)) if self.ocr else None
        if not ocr and (text is None or text == ""):
            return cap
        out: Dict[str, Any] = {"image_caption": cap}
        if ocr:
            out["ocr_text"] = ocr
        if text is not None and text != "":
            out["text"] = text
        return out

    def predict(self, state: Any, questions: Dict[str, Any], **kw) -> Dict[str, Any]:
        return self.laya.predict(self.text_state(state), questions, **kw)


class StockLaya:
    """Stock ``laya.Agent`` on text-only states; image states are skipped."""

    name = "stock_laya"

    def __init__(self, laya_agent):
        self.laya = laya_agent

    def predict(self, state: Any, questions: Dict[str, Any], **kw) -> Dict[str, Any]:
        if split_state(state)[0] is not None:
            return _result("stock_laya", {})
        return self.laya.predict(state, questions, **kw)


class SiglipZeroShot:
    """SigLIP zero-shot ``choice``: softmax over ``logit_scale·cos(img, "a photo of {option}") + bias``.

    ``model`` is a ``transformers`` ``SiglipModel`` (both towers); ``processor`` maps
    ``text=[...]`` / ``images=[...]`` to model inputs (``AutoProcessor`` for real checkpoints).
    Non-choice questions and text-only states are skipped.
    """

    name = "siglip_zeroshot"

    def __init__(self, model, processor, template: str = "a photo of {}", device: str = "cpu"):
        self.model, self.processor, self.template, self.device = model.eval().to(device), processor, template, device
        self._text_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_SIGLIP, revision: Optional[str] = None,
                        device: Optional[str] = None, **kw) -> "SiglipZeroShot":
        import torch
        from transformers import AutoModel, AutoProcessor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_id, revision=revision)
        return cls(model, AutoProcessor.from_pretrained(model_id, revision=revision), device=device, **kw)

    @staticmethod
    def _features(out):
        return out if hasattr(out, "norm") else out.pooler_output  # transformers 5 may return a ModelOutput

    def _normed(self, feats) -> np.ndarray:
        x = self._features(feats).float()
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()

    def encode_image(self, img) -> np.ndarray:
        import torch

        inp = self.processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            return self._normed(self.model.get_image_features(pixel_values=inp["pixel_values"].to(self.device)))[0]

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        import torch

        todo = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if todo:
            # SigLIP was trained with max_length padding (64 tokens).
            inp = self.processor(text=todo, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                emb = self._normed(self.model.get_text_features(input_ids=inp["input_ids"].to(self.device)))
            self._text_cache.update(zip(todo, emb))
        return np.stack([self._text_cache[t] for t in texts])

    def option_prompts(self, qdef: Dict[str, Any]) -> List[str]:
        crit = qdef["criteria"]
        items = crit.items() if isinstance(crit, dict) else ((c, None) for c in crit)
        return [self.template.format(desc if isinstance(desc, str) and desc.strip() else key) for key, desc in items]

    def scale_bias(self):
        scale = float(self.model.logit_scale.exp()) if hasattr(self.model, "logit_scale") else 1.0
        bias = float(self.model.logit_bias) if getattr(self.model, "logit_bias", None) is not None else 0.0
        return scale, bias

    def predict(self, state: Any, questions: Dict[str, Any], **kw) -> Dict[str, Any]:
        image, _ = split_state(state)
        answers: Dict[str, Any] = {}
        if image is None or not any(q.get("type") == "choice" for q in questions.values()):
            return _result("siglip_zeroshot", answers)
        img = self.encode_image(load_image(image))
        scale, bias = self.scale_bias()
        for qid, q in questions.items():
            if q.get("type") != "choice":
                continue
            crit = q["criteria"]
            keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            z = scale * (self.encode_texts(self.option_prompts(q)) @ img) + bias
            p = np.exp(z - z.max())
            answers[qid] = choice_answer(keys, p / p.sum())
        return _result("siglip_zeroshot", answers)

