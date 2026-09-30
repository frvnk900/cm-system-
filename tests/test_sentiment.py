from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.db.db import get_db
from app.api.routes.comments import comments
from app.api.schema.ai_model import (
	ClassificationRequest,
	SentimentBatch,
	SentimentOutput,
)
from app.core.settings import default_runtime_settings
from app.services import sentiment
from app.services.ai_classification import AIProviderError


def _request() -> ClassificationRequest:
	return ClassificationRequest(
		location="Lilongwe",
		posts=[{"post_id": "post_1", "page_id": "page_1", "body": "New store opening"}],
		comments=[
			{"comment_id": "c_love", "post_id": "post_1", "page_id": "page_1", "body": "Love this shop!"},
			{"comment_id": "c_scam", "post_id": "post_1", "page_id": "page_1", "body": "You are scammers"},
			{"comment_id": "c_cached", "post_id": "post_1", "page_id": "page_1", "body": "When do you open?"},
			{"comment_id": "c_image", "post_id": "post_1", "page_id": "page_1", "body": "  "},
		],
	)


class _FakeSession:
	def scalar(self, *_args, **_kwargs):
		return None


def _client(monkeypatch, classify) -> TestClient:
	application = FastAPI()
	application.include_router(comments.router)
	application.dependency_overrides[get_db] = lambda: _FakeSession()
	monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
	monkeypatch.setattr(comments, "load_runtime_settings", lambda _db: default_runtime_settings())
	monkeypatch.setattr(comments, "get_page", lambda page_id, token: {"name": "Lilongwe", "access_token": "p"})
	monkeypatch.setattr(comments, "_build_page_request", lambda *args, **kwargs: _request())
	cached_row = SimpleNamespace(sentiment="neutral", confidence=0.9, reason="A question.")
	monkeypatch.setattr(comments, "load_sentiments", lambda _db, _c: {"c_cached": cached_row})
	monkeypatch.setattr(comments, "save_sentiments", lambda *args: None)
	monkeypatch.setattr(comments, "classify_sentiment", classify)
	return TestClient(application)


def test_sentiment_route_labels_positive_and_negative(monkeypatch):
	sent_ids = []

	def classify(request, settings, on_batch=None):
		sent_ids.extend(c.comment_id for c in request.comments)
		return {
			"c_love": SentimentOutput(comment_id="c_love", sentiment="positive", confidence=0.95, reason="Praise."),
			"c_scam": SentimentOutput(comment_id="c_scam", sentiment="negative", confidence=0.9, reason="Accusation."),
		}

	client = _client(monkeypatch, classify)
	response = client.get("/comments/page/page_1/sentiment")

	assert response.status_code == 200
	labels = {row["comment_id"]: row["sentiment"] for row in response.json()}
	assert labels == {
		"c_love": "positive",
		"c_scam": "negative",
		"c_cached": "neutral",   # from the cache, not re-sent
		"c_image": "neutral",    # no text, never sent
	}
	# Only uncached comments with text go to the AI.
	assert sent_ids == ["c_love", "c_scam"]
	# Every row carries what DELETE /comments/{id}?page_id= needs.
	assert all(row["page_id"] == "page_1" for row in response.json())


def test_sentiment_filter_returns_only_requested_kind(monkeypatch):
	def classify(request, settings, on_batch=None):
		return {
			"c_love": SentimentOutput(comment_id="c_love", sentiment="positive", confidence=0.95, reason="Praise."),
			"c_scam": SentimentOutput(comment_id="c_scam", sentiment="negative", confidence=0.9, reason="Accusation."),
		}

	client = _client(monkeypatch, classify)
	response = client.get("/comments/page/page_1/sentiment?sentiment=positive")

	assert [row["comment_id"] for row in response.json()] == ["c_love"]


def test_sentiment_service_ignores_unknown_ids_and_skips_failed_batches(monkeypatch):
	calls = {"n": 0}

	def fake_complete(settings, prompt, content, response_format, task):
		calls["n"] += 1
		if calls["n"] == 2:
			raise AIProviderError("down")
		return SentimentBatch(results=[
			SentimentOutput(comment_id="c_love", sentiment="positive", confidence=0.9, reason="Praise."),
			SentimentOutput(comment_id="made_up", sentiment="negative", confidence=0.9, reason="Invented."),
		])

	monkeypatch.setattr(sentiment, "complete_with_fallback", fake_complete)
	settings = default_runtime_settings().model_copy(update={"batch_size": 2})
	request = _request().model_copy(update={"comments": _request().comments[:3]})

	results = sentiment.classify_sentiment(request, settings)

	# Batch 1 labelled c_love (made-up id dropped); batch 2 failed and was skipped.
	assert set(results) == {"c_love"}
