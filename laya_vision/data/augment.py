"""Record-level augmentation, applied before tokenizing and seeded by the caller.

Rules (ARCHITECTURE §5.3, mirroring Laya's text pipeline):

- option order shuffle, target permuted with it: ``choice`` only. ``score`` is ordinal and ``noul``
  has the fixed semantic order ``[false, true]``, so neither is ever reordered.
- label renaming for ``choice`` keys; never ``yes/no/true/false`` as a choice key.
- instruction paraphrase from per-task templates, formatted with the record's ``fields``.
- distractor questions (0-2 extra questions per image) are a generation-time option of the
  converters (``n_distractors``); at dataset level they are just extra records sharing an image.
"""
import copy
import random
import string
from typing import Any, Dict, List, Optional

BANNED_CHOICE_KEYS = {"yes", "no", "true", "false", "y", "n", "t", "f"}

# Per-task instruction templates, formatted with ``record["fields"]``. The first template is the
# canonical instruction the converters write; the rest are paraphrases.
TEMPLATES: Dict[str, List[str]] = {
    "caption_choice": [
        "Which caption describes the image?",
        "Pick the caption that best matches the picture.",
        "Which of these captions was written for this image?",
        "Select the description that fits the image.",
        "Choose the caption that correctly describes what is shown.",
        "Which sentence describes this photo?",
    ],
    "caption_noul": [
        "This caption describes the image: \"{caption}\"",
        "The picture matches the description \"{caption}\".",
        "\"{caption}\" is an accurate caption for this image.",
        "The following caption was written for this image: \"{caption}\"",
        "The image shows what this caption says: \"{caption}\"",
    ],
    "classify": [
        "What does the image show?",
        "Which category does this image belong to?",
        "Classify the image.",
        "What is in this picture?",
        "Pick the label that fits the image.",
        "Identify the main subject of the image.",
    ],
    "cifar10": [
        "What object is shown in the image?",
        "Which class does this small photo belong to?",
        "Classify the object in the picture.",
        "What is the main object in this image?",
        "Pick the category of the pictured object.",
    ],
    "oxford_pets": [
        "Which breed is the animal in the image?",
        "What breed of cat or dog is shown?",
        "Identify the pet's breed.",
        "Which cat or dog breed is in this photo?",
        "Pick the breed that matches the animal pictured.",
    ],
    "food101": [
        "Which dish is shown in the image?",
        "What food is in this photo?",
        "Identify the dish.",
        "Which of these dishes is pictured?",
        "What kind of food is this?",
    ],
    "eurosat": [
        "What land use or land cover does this satellite image show?",
        "Classify this satellite image.",
        "Which land cover type is visible in this aerial image?",
        "What kind of terrain does this satellite tile show?",
        "Pick the land-use category of this satellite photo.",
    ],
    "class_noul": [
        "The image shows {label}.",
        "This is a picture of {label}.",
        "{label} is visible in the image.",
        "The main subject of this photo is {label}.",
        "This image contains {label}.",
    ],
    "vqa_choice": [
        "{q}",
        "Answer the question about the image: {q}",
        "Look at the image. {q}",
        "Based on the picture: {q}",
        "Question about this image: {q}",
    ],
    "vqa_noul": [
        "{q}",
        "Answer yes or no about the image: {q}",
        "Look at the image. {q}",
        "Based on the picture: {q}",
        "Question about this image: {q}",
    ],
    "quality_score": [
        "How good is the technical quality of this image?",
        "Rate the overall quality of this photo.",
        "How would a viewer rate the quality of this picture?",
        "Judge the image quality (sharpness, exposure, noise, artifacts).",
        "What is the perceived quality of this image?",
    ],
    "aesthetic_score": [
        "How aesthetically pleasing is this image?",
        "Rate the aesthetic quality of this photo.",
        "How beautiful is this picture?",
        "Judge the composition and visual appeal of this image.",
        "How would photographers rate the aesthetics of this image?",
    ],
}


def instruction(template_key: str, fields: Optional[Dict[str, Any]] = None, i: int = 0) -> str:
    return TEMPLATES[template_key][i].format(**(fields or {}))


def _choice_keys(n: int, style: str) -> List[str]:
    # letters that read as yes/no/true/false (e.g. "F") are skipped, so 6+ options stay valid
    if style == "upper":
        keys = [c for c in string.ascii_uppercase if c.lower() not in BANNED_CHOICE_KEYS]
    elif style == "lower":
        keys = [c for c in string.ascii_lowercase if c not in BANNED_CHOICE_KEYS]
    elif style == "number":
        keys = [str(i + 1) for i in range(n)]
    elif style == "option":
        keys = ["option %d" % (i + 1) for i in range(n)]
    elif style == "paren":
        keys = ["(%s)" % c for c in string.ascii_lowercase]
    else:
        raise ValueError("unknown key style %r" % style)
    if n > len(keys):
        keys = ["option %d" % (i + 1) for i in range(n)]
    keys = keys[:n]
    assert not BANNED_CHOICE_KEYS & {k.lower() for k in keys}
    return keys


KEY_STYLES = ("upper", "lower", "number", "option", "paren")


def shuffle_options(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Shuffle choice options and permute the target the same way. Other types unchanged."""
    q = rec["question"]
    if q["type"] != "choice":
        return rec
    crit = q["criteria"]
    items = list(crit.items()) if isinstance(crit, dict) else [(c, None) for c in crit]
    perm = list(range(len(items)))
    rng.shuffle(perm)
    rec = dict(rec)
    q = dict(q)
    if isinstance(crit, dict):
        q["criteria"] = {items[p][0]: items[p][1] for p in perm}
    else:
        q["criteria"] = [crit[p] for p in perm]
    rec["question"] = q
    rec["target"] = [rec["target"][p] for p in perm]
    return rec


def rename_labels(rec: Dict[str, Any], rng: random.Random, style: Optional[str] = None) -> Dict[str, Any]:
    """Rename choice keys (A/B/..., 1/2/..., option 1, ...). The old key moves into the description
    when there was none, so no information is lost. Target order is unchanged."""
    q = rec["question"]
    if q["type"] != "choice":
        return rec
    crit = q["criteria"]
    items = list(crit.items()) if isinstance(crit, dict) else [(c, None) for c in crit]
    keys = _choice_keys(len(items), style or rng.choice(KEY_STYLES))
    rec = dict(rec)
    q = dict(q)
    q["criteria"] = {nk: (str(k) if v is None or v == "" else v) for nk, (k, v) in zip(keys, items)}
    rec["question"] = q
    return rec


def paraphrase(rec: Dict[str, Any], rng: random.Random, templates: Optional[Dict[str, List[str]]] = None
               ) -> Dict[str, Any]:
    """Replace the instructions with a random template of the record's ``template`` key."""
    key = rec.get("template") or rec.get("task")
    tpl = (templates or TEMPLATES).get(key)
    if not tpl:
        return rec
    try:
        ins = rng.choice(tpl).format(**(rec.get("fields") or {}))
    except (KeyError, IndexError):
        return rec
    rec = dict(rec)
    rec["question"] = dict(rec["question"], instructions=ins)
    return rec


class Augmenter:
    """``Augmenter()(record, rng) -> record``; never mutates its input.

    Text-only rows (``image`` is null) are left untouched by default, so the text mix keeps the
    original typed-decisions training distribution.
    """

    def __init__(self, p_shuffle: float = 1.0, p_rename: float = 0.3, p_paraphrase: float = 0.5,
                 templates: Optional[Dict[str, List[str]]] = None, augment_text: bool = False):
        self.p_shuffle, self.p_rename, self.p_paraphrase = p_shuffle, p_rename, p_paraphrase
        self.templates = templates
        self.augment_text = augment_text

    def __call__(self, rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        if rec.get("image") is None and not self.augment_text:
            return rec
        rec = copy.deepcopy(rec)
        # Draw every random number unconditionally so the stream does not depend on the record.
        u_shuf, u_ren, u_par = rng.random(), rng.random(), rng.random()
        if u_par < self.p_paraphrase:
            rec = paraphrase(rec, rng, self.templates)
        if u_ren < self.p_rename:
            rec = rename_labels(rec, rng)
        if u_shuf < self.p_shuffle:
            rec = shuffle_options(rec, rng)
        return rec
