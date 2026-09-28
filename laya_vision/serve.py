"""HTTP server: Laya's ``/v1/systemone`` contract with image states (plan.md §2.11).

    python -m laya_vision.serve --checkpoint DIR [--host 0.0.0.0] [--port 8000] [--device cuda]

Request body, as in ``laya.serve``: ``{"state": ..., "questions": {...}, "max_len"?, "head_max_len"?}``.
``state`` may be ``{"image": "<base64 | data:image/...;base64,...>", "text": optional}``.

- The image is checked against its own limits: ≤ 10 MB decoded, ≤ 40 MP, JPEG/PNG/WebP/GIF
  (first frame). It is decoded here with ``images.load_image``. Only inline data is accepted,
  never a path, so a client cannot make the server read local files.
- The ``text`` part and the questions go through Laya's own ``_check_request_limits``.
- Text-only requests keep Laya's 2 MiB body cap and produce Laya's response unchanged.
- The body cap for image requests is ``LAYA_VISION_MAX_BODY_BYTES`` (default ≈ 16 MiB).
- ``LAYA_API_KEY`` enables the same bearer check as ``laya.serve``. ``LAYA_MAX_CONCURRENT``,
  ``LAYA_MAX_TOKEN_BUDGET`` and ``LAYA_THREADS`` behave as they do there.
"""
import argparse
import base64
import binascii
import hmac
import io
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from laya import serve as lserve
from laya_vision import images
from laya_vision.sequence import is_image_state, split_state

_log = logging.getLogger("laya_vision.serve")

SERVE_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")
MAX_IMAGE_BYTES = images.MAX_IMAGE_BYTES          # decoded
MAX_IMAGE_PIXELS = images.MAX_PIXELS
DEFAULT_MAX_BODY_BYTES = MAX_IMAGE_BYTES * 4 // 3 + lserve.MAX_BODY_BYTES + 64 * 1024
_DATA_URL = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,", re.I)


def max_body_bytes() -> int:
    raw = os.environ.get("LAYA_VISION_MAX_BODY_BYTES")
    try:
        n = int(raw) if raw else DEFAULT_MAX_BODY_BYTES
    except ValueError:
        n = DEFAULT_MAX_BODY_BYTES
    return n if n > 0 else DEFAULT_MAX_BODY_BYTES


def decode_image_field(value: Any):
    """RGB PIL image from a base64 string or data URL, or ``HTTPException`` (400 bad / 413 too large)."""
    from fastapi import HTTPException
    from PIL import Image

    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail="'state.image' must be a base64 string or a data URL")
    s = value.strip()
    m = _DATA_URL.match(s)
    if m:
        s = s[m.end():]
    elif s.startswith("data:"):
        raise HTTPException(status_code=400, detail="only base64 image data URLs (data:image/...;base64,...)")
    s = "".join(s.split())
    if len(s) > (MAX_IMAGE_BYTES + 2) // 3 * 4 + 4:
        raise HTTPException(status_code=413, detail="image too large (> %d bytes decoded)" % MAX_IMAGE_BYTES)
    try:
        data = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="'state.image' is not valid base64") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large (%d > %d bytes)" % (len(data), MAX_IMAGE_BYTES))
    try:
        head = Image.open(io.BytesIO(data))  # header only; pixels are decoded by images.load_image
        fmt, (w, h) = head.format, head.size
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="image has too many pixels") from None
    except Exception:
        raise HTTPException(status_code=400, detail="'state.image' is not a decodable image") from None
    if fmt not in SERVE_FORMATS:
        raise HTTPException(status_code=400, detail="image format %s not allowed (allowed: %s)"
                            % (fmt, ", ".join(SERVE_FORMATS)))
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="image too large (%dx%d > %d pixels)" % (w, h, MAX_IMAGE_PIXELS))
    try:
        return images.load_image(data)
    except images.ImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def check_request(state: Any, questions: Any):
    """Validate ``state``/``questions``; returns the state to predict on (image decoded to PIL)."""
    from fastapi import HTTPException

    if not is_image_state(state):
        lserve._check_request_limits(state, questions)
        return state
    try:
        image, text = split_state(state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    # Laya's text limits on the text part only ("" stands in for an absent text).
    lserve._check_request_limits("" if text is None else text, questions)
    img = decode_image_field(image)
    return {"image": img} if text is None else {"image": img, "text": text}


async def _read_body(request, cap: int) -> bytes:
    from fastapi import HTTPException

    total, chunks = 0, []
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(agent: Any = None, checkpoint: Optional[str] = None, device: Optional[str] = None):
    """FastAPI app around a ``VisionAgent`` (pass ``agent`` in tests, or a ``checkpoint`` to load)."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse

    if agent is None:
        if checkpoint is None:
            raise ValueError("create_app needs an agent or a checkpoint")
        import laya_vision

        lserve._apply_thread_limit()
        agent = laya_vision.load(checkpoint, device=device)
    api_key = os.environ.get("LAYA_API_KEY") or None
    expected_auth = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""
    body_cap = max_body_bytes()
    max_concurrent = lserve._resolve_max_concurrent()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-vision-infer")
    locks: Dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="laya-vision-serve", lifespan=lifespan,
                  summary="Laya System-1 decisions over text and images (/v1/systemone)")

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        if not hmac.compare_digest((authorization or "").encode("utf-8", "surrogateescape"), expected_auth):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        vcfg = getattr(agent, "vcfg", None)
        return {"status": "ok", "checkpoint": checkpoint, "device": str(getattr(agent, "device", "unknown")),
                "vision": vcfg is not None, "image_tokens": getattr(vcfg, "n_tokens", None),
                "limits": {"max_image_bytes": MAX_IMAGE_BYTES, "max_image_pixels": MAX_IMAGE_PIXELS,
                           "image_formats": list(SERVE_FORMATS), "max_body_bytes": body_cap,
                           "max_text_body_bytes": lserve.MAX_BODY_BYTES, "max_state_chars": lserve.MAX_STATE_CHARS}}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        if "admission" not in locks:  # created lazily: bound to the running loop
            locks["admission"], locks["gate"] = asyncio.Semaphore(max_concurrent), asyncio.Lock()
        if locks["admission"].locked():
            raise HTTPException(status_code=503, detail="server busy, try again later", headers={"Retry-After": "1"})
        async with locks["admission"]:
            return await _inner(request)

    async def _inner(request: Request):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > body_cap:
            raise HTTPException(status_code=413, detail="request body too large")
        raw = await _read_body(request, body_cap)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state, questions = body.get("state"), body["questions"]
        if not is_image_state(state) and len(raw) > lserve.MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")  # Laya's cap for text
        state = check_request(state, questions)
        cap = lserve._resolve_max_token_budget()
        kw = {k: v for k in ("max_len", "head_max_len")
              if (v := lserve._validate_budget_param(body, k, cap)) is not None}
        try:
            async with locks["gate"]:
                t0 = time.perf_counter()
                result = await asyncio.get_running_loop().run_in_executor(
                    pool, lambda: agent.predict(state, questions, **kw))
                ms = (time.perf_counter() - t0) * 1000.0
        except HTTPException:
            raise
        except ValueError as e:  # question / image validation: safe to show
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001 -- never leak internals
            _log.exception("inference failed")
            raise HTTPException(status_code=500, detail="inference failed")
        return JSONResponse(content=result, headers={"Server-Timing": "inference;dur=%.2f" % ms,
                                                     "X-Inference-Time-Ms": "%.2f" % ms})

    return app


def main(argv=None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="laya_vision HTTP server (/v1/systemone)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default=os.environ.get("LAYA_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8000")))
    ap.add_argument("--device", default=os.environ.get("LAYA_DEVICE") or None)
    ap.add_argument("--log-level", default=os.environ.get("LAYA_LOG_LEVEL", "info"))
    args = ap.parse_args(argv)
    uvicorn.run(create_app(checkpoint=args.checkpoint, device=args.device), host=args.host, port=args.port,
                log_level=args.log_level)


if __name__ == "__main__":
    main()
