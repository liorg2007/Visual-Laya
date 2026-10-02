"""Local web UI for laya_vision: the /v1/systemone server plus a single-page front end at ``/``.

    python -m laya_vision.ui --checkpoint runs/stage2_c/wise085_calibrated [--port 7860] [--device cuda]

Binds to 127.0.0.1 by default. The page posts to ``/v1/systemone`` on the same origin, so
``LAYA_API_KEY`` is not supported here (the page has no way to send it).
"""
import argparse
import os
from pathlib import Path

from laya_vision import serve

PAGE = Path(__file__).with_name("ui.html")


def create_ui_app(agent=None, checkpoint=None, device=None):
    from fastapi.responses import HTMLResponse

    app = serve.create_app(agent=agent, checkpoint=checkpoint, device=device)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    return app


def main(argv=None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="laya_vision local web UI")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=os.environ.get("LAYA_DEVICE") or None)
    args = ap.parse_args(argv)
    os.environ.pop("LAYA_API_KEY", None)
    uvicorn.run(create_ui_app(checkpoint=args.checkpoint, device=args.device), host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
