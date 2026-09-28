"""Token sequences with an image block at the start of Laya's state region.

Layout (plan.md §2.4), produced by calling ``laya.common.build_sequence`` unchanged with
pre-tokenized ``state_ids``:

    [CLS] <t> question: ... [SEP] [MASK] o0 ... [MASK] oK [SEP] [PAD]*N [SEP] text ... [SEP]
                                                                ^ image_start

The ``[PAD]`` placeholders are overwritten with projector outputs in ``LayaVisionModel.forward``;
their positions come from ``image_start``, never from searching for ``pad_id``.
"""
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import torch

from laya.agent import Agent
from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    encode_text,
    render_options,
    serialize_state,
)

State = Union[str, dict, list]
IMAGE_KEY = "image"
TEXT_KEY = "text"


def is_image_state(state: Any) -> bool:
    """Only a dict with the reserved ``"image"`` key takes the vision path."""
    return isinstance(state, dict) and IMAGE_KEY in state


def split_state(state: State) -> Tuple[Any, Optional[State]]:
    """``(image, text_state)`` for an image state, ``(None, state)`` for any other state."""
    if not is_image_state(state):
        return None, state
    extra = set(state) - {IMAGE_KEY, TEXT_KEY}
    if extra:
        raise ValueError("image state accepts only the keys 'image' and 'text'; got extra keys %s"
                         % sorted(extra))
    if isinstance(state[IMAGE_KEY], (list, tuple)):
        raise ValueError("only one image per state is supported in v1")
    return state[IMAGE_KEY], state.get(TEXT_KEY)


def to_internal(qdef: Dict[str, Any], qid: str = "q") -> Dict[str, Any]:
    """Validate a public Laya question dict and convert it to the internal ``{t, ins, crit}`` form."""
    Agent._check_question(qid, qdef)
    return Agent._to_internal(qdef)


def _state_token_ids(tok, state: State) -> List[int]:
    return encode_text(tok, serialize_state(state).replace(tok.mask_token, " "),
                       add_special_tokens=False)["input_ids"]


def build_item(
    tok,
    state: State,
    q: Dict[str, Any],
    n_img: int,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    state_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """One question row for one state.

    ``q`` is an internal question (see ``to_internal``). ``n_img`` is the number of image tokens
    (``VisionConfig.n_tokens``); it is ignored for text-only states. ``state_ids`` optionally
    passes the already-tokenized *text* part of the state so multi-question requests tokenize it
    once.

    Returns ``{"ids", "markers", "qtype", "options", "image_start"}`` where ``image_start`` is
    -1 for text-only rows. Text-only rows are built exactly as ``laya.Agent._encode_state`` builds
    them, which is what makes text-only outputs identical to stock Laya.
    """
    image, text = split_state(state)
    n_opts = len(render_options(q))

    if image is None:
        truncate_left = isinstance(state, list)
        if state_ids is None:
            state_ids = _state_token_ids(tok, state)
        ids, markers, stats = build_sequence(tok, state, q, max_len, head_max_len,
                                             option_order=option_order, truncate_left=truncate_left,
                                             state_ids=state_ids, return_stats=True)
        if len(markers) != n_opts:
            raise ValueError("options exceed head_max_len=%d" % head_max_len)
        return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "options": stats,
                "image_start": -1}

    # Head length: build with an empty state; the result is head + [SEP].
    head, _ = build_sequence(tok, None, q, max_len, head_max_len, option_order=option_order,
                             state_ids=[])
    image_start = len(head) - 1
    room = max(0, max_len - image_start - 1)
    if room < n_img:
        raise ValueError(
            "image does not fit: %d image tokens but only %d state tokens left after the question "
            "(max_len=%d). Shorten the options or use a longer-context checkpoint."
            % (n_img, room, max_len))

    block = [tok.pad_token_id] * n_img
    text_budget = room - n_img - 1  # one [SEP] between the image block and the text
    if text is not None and text != "" and text_budget > 0:
        if state_ids is None:
            state_ids = _state_token_ids(tok, text)
        # Conversation lists keep their newest turns, matching Laya's truncate_left for lists.
        text_ids = (state_ids[max(0, len(state_ids) - text_budget):] if isinstance(text, list)
                    else state_ids[:text_budget])
        block = block + [tok.sep_token_id] + text_ids

    ids, markers, stats = build_sequence(tok, None, q, max_len, head_max_len,
                                         option_order=option_order, truncate_left=False,
                                         state_ids=block, return_stats=True)
    if len(markers) != n_opts:
        raise ValueError("options exceed head_max_len=%d" % head_max_len)
    placeholder = ids[image_start:image_start + n_img]
    if len(placeholder) != n_img or any(t != tok.pad_token_id for t in placeholder):
        raise AssertionError("image block misplaced: expected %d placeholders at %d" % (n_img, image_start))
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "options": stats,
            "image_start": image_start}


def collate_vision(items: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    """Collate rows into a batch: ``laya.common.collate_items`` plus the image routing tensors.

    Each item may carry ``"image_ref"`` (any hashable identifying its image, e.g. a sha256 or a
    dataset path) when ``image_start >= 0``. The batch gets:

    - ``image_refs``: unique refs in first-seen order; the caller turns these into pixel values
    - ``image_index`` [B] long: index into ``image_refs``, -1 for text-only rows
    - ``image_start`` [B] long: first placeholder position, -1 for text-only rows
    """
    batch = collate_items([list(items)], pad_id)
    refs: List[Hashable] = []
    seen: Dict[Hashable, int] = {}
    index, start = [], []
    for it in items:
        s = int(it.get("image_start", -1))
        if s < 0:
            index.append(-1)
            start.append(-1)
            continue
        ref = it.get("image_ref")
        if ref is None:
            raise ValueError("an image row needs an 'image_ref'")
        if ref not in seen:
            seen[ref] = len(refs)
            refs.append(ref)
        index.append(seen[ref])
        start.append(s)
    batch["image_refs"] = refs
    batch["image_index"] = torch.tensor(index, dtype=torch.long)
    batch["image_start"] = torch.tensor(start, dtype=torch.long)
    return batch
