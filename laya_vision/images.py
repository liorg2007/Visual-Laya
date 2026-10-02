"""Safe image decoding and SigLIP preprocessing (plan.md §2.2, §2.11)."""
import base64
import binascii
import hashlib
import io
import os
import re
import warnings
from typing import Any, List, Tuple

import numpy as np
import torch

MAX_IMAGE_BYTES = 10 * 1024 * 1024     # encoded input size
MAX_PIXELS = 40_000_000                 # width * height
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP")
_DATA_URL = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,", re.I)


class ImageError(ValueError):
    """The input is not an acceptable image (bad encoding, format, or size)."""


def _to_bytes(x: Any) -> bytes:
    """Encoded image bytes from a path, bytes, base64 string or data URL."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        data = bytes(x)
    elif isinstance(x, os.PathLike):
        data = _read_file(os.fspath(x))
    elif isinstance(x, str):
        s = x.strip()
        m = _DATA_URL.match(s)
        if m:
            data = _b64(s[m.end():])
        elif s.startswith("data:"):
            raise ImageError("only base64 image data URLs (data:image/...;base64,...) are supported")
        elif len(s) < 4096 and os.path.isfile(s):
            data = _read_file(s)
        else:
            data = _b64(s)
    else:
        raise ImageError("unsupported image input type %s" % type(x).__name__)
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError("image is %d bytes; the limit is %d" % (len(data), MAX_IMAGE_BYTES))
    return data


def _read_file(path: str) -> bytes:
    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        raise ImageError("image file is %d bytes; the limit is %d" % (size, MAX_IMAGE_BYTES))
    with open(path, "rb") as f:
        return f.read()


def _b64(s: str) -> bytes:
    if len(s) > MAX_IMAGE_BYTES * 4 // 3 + 8:
        raise ImageError("base64 image exceeds %d decoded bytes" % MAX_IMAGE_BYTES)
    try:
        return base64.b64decode("".join(s.split()), validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("string is neither an existing file path nor valid base64 image data") from None


def _to_rgb(img) -> "Image.Image":  # noqa: F821
    from PIL import Image

    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "RGB" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode == "CMYK":
        return img.convert("RGB")
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # 16/32-bit grayscale: scale into 8 bits rather than letting convert() clip.
        a = np.asarray(img, dtype=np.float64)
        hi = a.max() if a.size else 1.0
        a = np.clip(a / (hi if hi > 255 else 255.0) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(a, "L").convert("RGB")
    return img.convert("RGB") if img.mode != "RGB" else img


def _decode(data: bytes):
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            if img.format not in ALLOWED_FORMATS:
                raise ImageError("image format %s is not allowed (allowed: %s)"
                                 % (img.format, ", ".join(ALLOWED_FORMATS)))
            w, h = img.size
            if w * h > MAX_PIXELS or w <= 0 or h <= 0:
                raise ImageError("image is %dx%d = %d pixels; the limit is %d" % (w, h, w * h, MAX_PIXELS))
            img.seek(0)  # first frame of an animated GIF/WebP
            img.load()
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ImageError("image rejected as a decompression bomb: %s" % e) from None
    except Exception as e:
        raise ImageError("could not decode image: %s" % e) from None
    return _to_rgb(img)


def _from_pil(img):
    w, h = img.size
    if w == 0 or h == 0:
        raise ImageError("image is %dx%d; it must have at least one pixel" % (w, h))
    if w * h > MAX_PIXELS:
        raise ImageError("image is %dx%d = %d pixels; the limit is %d" % (w, h, w * h, MAX_PIXELS))
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    return _to_rgb(img)


def _from_array(a: np.ndarray):
    from PIL import Image

    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)  # CHW -> HWC
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[-1] not in (3, 4)):
        raise ImageError("array image must be HxW, HxWx3 or HxWx4, got shape %s" % (a.shape,))
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ImageError("array image has shape %s; it must have at least one pixel" % (a.shape,))
    if a.shape[0] * a.shape[1] > MAX_PIXELS:
        raise ImageError("image has %d pixels; the limit is %d" % (a.shape[0] * a.shape[1], MAX_PIXELS))
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if a.size and a.max() <= 1.0:
            a = a * 255.0
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    mode = "L" if a.ndim == 2 else ("RGB" if a.shape[-1] == 3 else "RGBA")
    return _to_rgb(Image.fromarray(np.ascontiguousarray(a), mode))


def load_image_and_hash(x: Any) -> Tuple["Image.Image", str]:  # noqa: F821
    """``(RGB PIL image, sha256)`` decoding ``x`` once. See ``load_image`` / ``image_sha256``."""
    from PIL import Image

    if isinstance(x, Image.Image):
        img = _from_pil(x)
        return img, _pixel_hash(img)
    if isinstance(x, np.ndarray):
        img = _from_array(x)
        return img, _pixel_hash(img)
    data = _to_bytes(x)
    return _decode(data), hashlib.sha256(data).hexdigest()


def _pixel_hash(img) -> str:
    h = hashlib.sha256(("%s:%dx%d:" % (img.mode, img.size[0], img.size[1])).encode())
    h.update(img.tobytes())
    return h.hexdigest()


def load_image(x: Any):
    """RGB ``PIL.Image`` from a path, bytes, base64 string, data URL, ``PIL.Image`` or ndarray.

    Rejects inputs over 10 MB encoded or 40 MP, formats other than JPEG/PNG/WebP/GIF/BMP, and
    decompression bombs. Transparency is composited on white; animated images use frame 0.
    """
    return load_image_and_hash(x)[0]


def image_sha256(x: Any) -> str:
    """sha256 of the encoded source bytes (paths, bytes, base64), or of the RGB pixels (PIL, ndarray)."""
    return load_image_and_hash(x)[1]


def preprocess(images: List[Any], image_size: int) -> torch.Tensor:
    """``[M,3,S,S]`` float32, equal to ``SiglipImageProcessor``: bicubic resize to SxS, /255, (x-0.5)/0.5."""
    from PIL import Image

    out = []
    for img in images:
        if not isinstance(img, Image.Image) or img.mode != "RGB":
            img = load_image(img)
        a = np.asarray(img.resize((image_size, image_size), Image.BICUBIC), dtype=np.float32)
        out.append(torch.from_numpy((a / 255.0 - 0.5) / 0.5).permute(2, 0, 1))
    if not out:
        return torch.empty(0, 3, image_size, image_size)
    return torch.stack(out).contiguous()
