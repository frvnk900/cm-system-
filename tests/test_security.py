from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main
from app.api.routes.pages import pages
from app.core import security


def _client(monkeypatch, api_key: str | None) -> TestClient:
	monkeypatch.setattr(security, "get_settings", lambda: SimpleNamespace(api_key=api_key))
	monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
	monkeypatch.setattr(
		pages, "get_pages",
		lambda token: [{"id": "1", "name": "Shop", "access_token": "SECRET-PAGE-TOKEN"}],
	)
	return TestClient(main.app)  # no lifespan: these routes don't touch the database


def test_api_stays_open_until_a_key_is_configured(monkeypatch):
	client = _client(monkeypatch, api_key=None)

	assert client.get("/pages").status_code == 200


def test_api_routes_need_the_key_once_configured(monkeypatch):
	client = _client(monkeypatch, api_key="right-key")

	assert client.get("/pages").status_code == 401
	assert client.get("/pages", headers={"X-API-Key": "wrong"}).status_code == 401
	assert client.get("/pages", headers={"X-API-Key": "right-key"}).status_code == 200
	# Destructive and costly routes are covered too.
	assert client.delete("/comments/123?page_id=1").status_code == 401
	assert client.post("/api/ai/classify/1").status_code == 401
	assert client.get("/comments/page/1/sentiment").status_code == 401


def test_public_pages_do_not_need_the_key(monkeypatch):
	client = _client(monkeypatch, api_key="right-key")

	assert client.get("/").status_code == 200
	assert client.get("/logo.png").status_code == 200
	assert client.get("/robots.txt").status_code == 200
	# The admin portal is behind its own login, not the API key.
	assert client.get(main.get_settings().admin_path).status_code == 200


def test_page_tokens_are_never_returned(monkeypatch):
	client = _client(monkeypatch, api_key=None)

	response = client.get("/pages")

	assert response.json() == [{"id": "1", "name": "Shop"}]
	assert "SECRET-PAGE-TOKEN" not in response.text
