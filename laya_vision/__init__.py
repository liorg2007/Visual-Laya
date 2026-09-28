"""Laya-Vision: image-conditioned typed decisions on top of Laya."""

__version__ = "0.1.0"

# Resolved lazily so `import laya_vision` (e.g. for VisionConfig) does not import torch/transformers.
_LAZY_ATTRS = {
    "load": (".agent", "load"),
    "VisionAgent": (".agent", "VisionAgent"),
    "VisionConfig": (".config", "VisionConfig"),
    "LayaVisionModel": (".model", "LayaVisionModel"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name)) from None
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value
    return value
