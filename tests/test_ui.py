from fastapi.testclient import TestClient

from laya_vision import ui


def test_index_served_and_api_still_mounted():
    class A:
        device = "cpu"
    with TestClient(ui.create_ui_app(agent=A(), checkpoint="x")) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Laya Vision" in r.text
        assert c.get("/health").json()["status"] == "ok"
