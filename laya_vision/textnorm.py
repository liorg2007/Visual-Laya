"""Request text checks for the vision path: lone surrogates and all-caps text.

``check_text`` rejects strings the tokenizer cannot encode (a lone UTF-16 surrogate such as the
JSON escape ``"\\ud800"``) with a ``ValueError`` instead of the tokenizer's ``TypeError``, which
the server would otherwise report as a 500.

``unshout_question`` / ``unshout_state`` rewrite all-caps text to lower case before tokenizing.
The tokenizer is case-sensitive, so ``"APPLE PIE"`` becomes ``['APP', 'LE', 'ĠP', 'IE']``
instead of ``['apple', 'Ġpie']``. Training text is almost never in that form, and the model
collapsed on it (test_campaign/04: Food-101 0.82 -> 0.06 with all-caps prompts). Only strings
with no lower-case letters and at least one word of ``MIN_SHOUT_WORD`` letters are rewritten,
so acronyms and keys ("DNA", "CO2", "A", "USA") pass through unchanged. Callers get their
original labels back: only the token sequence sees the rewritten text.
"""
import re
from typing import Any, Dict

MIN_SHOUT_WORD = 4
_WORD = re.compile(r"[^\W\d_]+")


def check_text(obj: Any, where: str) -> None:
    """Raise ``ValueError`` if any string in ``obj`` (recursively, keys too) cannot be UTF-8 encoded."""
    if isinstance(obj, str):
        try:
            obj.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError("%s contains an invalid character %r at position %d (a lone surrogate "
                             "cannot be encoded as UTF-8)" % (where, obj[e.start], e.start)) from None
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_text(k, where)
            check_text(v, where)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            check_text(v, where)


def is_shouted(s: str) -> bool:
    return s == s.upper() and s != s.lower() and any(len(w) >= MIN_SHOUT_WORD for w in _WORD.findall(s))


def unshout(s: Any) -> Any:
    """``s.lower()`` for an all-caps string, anything else unchanged."""
    return s.lower() if isinstance(s, str) and is_shouted(s) else s


def unshout_state(state: Any) -> Any:
    """Copy of a text state (str, dict, list) with every all-caps string value lower-cased."""
    if isinstance(state, str):
        return unshout(state)
    if isinstance(state, dict):
        return {k: unshout_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [unshout_state(v) for v in state]
    return state


def unshout_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of an internal question (``{t, ins, crit[, labels]}``) with all-caps text lower-cased.

    Choice labels are rewritten only when that keeps them distinct, so the option count and
    order (which decode maps back to the caller's labels by position) never change.
    """
    out = dict(q)
    out["ins"] = unshout(q["ins"])
    crit = q.get("crit")
    if isinstance(crit, dict):
        keys = [unshout(k) for k in crit]
        if len(set(keys)) != len(keys):
            keys = list(crit)
        out["crit"] = {k2: unshout_state(v) for k2, v in zip(keys, crit.values())}
    elif isinstance(crit, list):
        out["crit"] = [unshout_state(c) for c in crit]
    return out
