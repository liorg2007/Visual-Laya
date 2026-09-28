"""Safe decoding and SigLIP-equivalent preprocessing (plan.md §2.11)."""
import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from _core_utils import noise_image, png_bytes
from laya_vision import images
from laya_vision.images import ImageError, image_sha256, load_image, preprocess


def _save(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF", "BMP"])
def test_formats(fmt):
    img = load_image(_save(Image.new("RGB", (20, 10), (10, 200, 30)), fmt))
    assert img.mode == "RGB" and img.size == (20, 10)


def test_input_kinds(tmp_path):
    data = png_bytes((1, 2, 3))
    p = tmp_path / "a.png"
    p.write_bytes(data)
    b64 = base64.b64encode(data).decode()
    for x in (data, bytearray(data), str(p), p, b64, "data:image/png;base64," + b64,
              Image.open(io.BytesIO(data)), np.full((30, 40, 3), (1, 2, 3), np.uint8)):
        img = load_image(x)
        assert img.mode == "RGB" and img.size == (40, 30)
        assert img.getpixel((0, 0)) == (1, 2, 3)
    # the same bytes hash the same whichever way they arrive
    assert image_sha256(data) == image_sha256(str(p)) == image_sha256(b64) == image_sha256("data:image/png;base64," + b64)
    assert image_sha256(data) != image_sha256(png_bytes((1, 2, 4)))
    assert image_sha256(noise_image(0)) == image_sha256(noise_image(0)) != image_sha256(noise_image(1))


def test_float_and_chw_arrays():
    a = np.zeros((3, 8, 6), np.float32)
    a[0] = 1.0
    img = load_image(a)
    assert img.size == (6, 8) and img.getpixel((0, 0)) == (255, 0, 0)
    assert load_image(np.zeros((5, 5), np.uint8)).mode == "RGB"


def test_rgba_composited_on_white():
    img = load_image(png_bytes((255, 0, 0, 0), mode="RGBA"))  # fully transparent red
    assert img.getpixel((0, 0)) == (255, 255, 255)
    img = load_image(png_bytes((0, 0, 0, 255), mode="RGBA"))
    assert img.getpixel((0, 0)) == (0, 0, 0)
    la = Image.new("LA", (4, 4), (0, 0))
    assert load_image(la).getpixel((0, 0)) == (255, 255, 255)


def test_grayscale_cmyk_palette():
    assert load_image(png_bytes(128, mode="L")).getpixel((0, 0)) == (128, 128, 128)
    cmyk = load_image(_save(Image.new("CMYK", (8, 8), (0, 255, 255, 0)), "JPEG"))
    r, g, b = cmyk.getpixel((0, 0))
    assert r > 200 and g < 60 and b < 60
    p = Image.new("RGB", (8, 8), (0, 0, 255)).convert("P")
    assert load_image(_save(p, "PNG")).getpixel((0, 0)) == (0, 0, 255)
    i16 = Image.fromarray(np.full((4, 4), 65535, np.uint16))
    assert load_image(i16).getpixel((0, 0)) == (255, 255, 255)


def test_animated_gif_first_frame():
    frames = [Image.new("RGB", (10, 10), c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=50)
    r, g, b = load_image(buf.getvalue()).getpixel((5, 5))
    assert r > 200 and g < 50 and b < 50


def test_rejects(monkeypatch, tmp_path):
    with pytest.raises(ImageError):
        load_image(b"not an image at all")
    with pytest.raises(ImageError):
        load_image("definitely/not/a/file.png")
    with pytest.raises(ImageError):
        load_image("data:text/plain;base64,aGVsbG8=")
    with pytest.raises(ImageError):
        load_image(png_bytes()[:40])  # truncated/corrupt
    with pytest.raises(ImageError, match="not allowed"):
        load_image(_save(Image.new("RGB", (8, 8)), "TIFF"))
    with pytest.raises(ImageError):
        load_image(12345)
    # size limits, scaled down so the test stays fast
    monkeypatch.setattr(images, "MAX_PIXELS", 100)
    with pytest.raises(ImageError, match="pixels"):
        load_image(png_bytes(size=(11, 10)))
    with pytest.raises(ImageError, match="pixels"):
        load_image(Image.new("RGB", (11, 10)))
    monkeypatch.setattr(images, "MAX_PIXELS", 40_000_000)
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 50)
    big = png_bytes(size=(64, 64))
    with pytest.raises(ImageError, match="bytes"):
        load_image(big)
    p = tmp_path / "big.png"
    p.write_bytes(big)
    with pytest.raises(ImageError, match="bytes"):
        load_image(str(p))


def test_decompression_bomb_header_only():
    # A PNG whose header claims 10000 x 10000 (100 MP): rejected before any pixel is decoded.
    img = Image.new("1", (10000, 10000))
    data = _save(img, "PNG")
    assert len(data) < 100_000
    with pytest.raises(ImageError, match="pixels"):
        load_image(data)


@pytest.mark.parametrize("size", [16, 32, 224])
def test_preprocess_matches_siglip_processor(size):
    from transformers import SiglipImageProcessor

    proc = SiglipImageProcessor(do_resize=True, size={"height": size, "width": size}, resample=Image.BICUBIC,
                                do_rescale=True, rescale_factor=1 / 255, do_normalize=True,
                                image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    imgs = [noise_image(0, (50, 37)), load_image(png_bytes((10, 20, 30), size=(7, 90)))]
    ref = proc(images=imgs, return_tensors="pt")["pixel_values"]
    out = preprocess(imgs, size)
    assert out.shape == (2, 3, size, size) and out.dtype == torch.float32
    assert (out - ref).abs().max().item() < 1e-4
    assert preprocess([], size).shape == (0, 3, size, size)
