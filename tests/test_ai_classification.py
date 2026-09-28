from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
from datetime import datetime, timedelta, timezone

from app.api.db.db import get_db
from app.api.routes import ai
from app.api.schema.ai_model import (
	ClassificationRequest,
	ClassificationResult,
)
from app.core.settings import default_runtime_settings
from app.services.ai_classification import AIOutputError, _validate_results


class _FakeSession:
	"""Stands in for the database: no stored pages, settings or cache."""

	def scalar(self, *_args, **_kwargs):
		return None


def _client(monkeypatch) -> TestClient:
	application = FastAPI()
	application.include_router(ai.router)
	application.dependency_overrides[get_db] = lambda: _FakeSession()
	monkeypatch.setattr(ai, "load_runtime_settings", lambda _db: default_runtime_settings())
	monkeypatch.setattr(ai, "load_classified", lambda _db, _comments: {})
	monkeypatch.setattr(ai, "load_system_prompt", lambda _db: "test prompt")
	return TestClient(application)


def _request() -> ClassificationRequest:
	return ClassificationRequest(
		location="Lilongwe",
		posts=[
			{"post_id": "post_1", "page_id": "page_1", "body": "Original post"}
		],
		comments=[
			{
				"comment_id": "comment_1",
				"post_id": "post_1",
				"author": "A",
				"body": "This is bad",
			},
			{"post_id": "post_1", "body": ""},
		],
	)


def test_classification_endpoint_collects_page_content(monkeypatch):
	client = _client(monkeypatch)
	results = [
		{
			"comment_id": "comment_1",
			"post_id": "post_1",
			"page_id": "page_1",
			"location": "Lilongwe",
			"flagged": True,
			"confidence": 0.45,
			"type": "negative_sentiment",
			"reason": "The comment expresses criticism.",
		},
		{
			"comment_id": None,
			"post_id": "post_1",
			"page_id": "page_1",
			"location": "Lilongwe",
			"flagged": False,
			"confidence": 0.0,
			"type": "none",
			"reason": "unparsable input",
		},
	]
	captured_request = None

	monkeypatch.setenv("META_ACCESS_TOKEN", "server-token")
	monkeypatch.setattr(
		ai,
		"get_page",
		lambda page_id, token: {
			"name": "Lilongwe",
			"access_token": "page-token",
		},
	)
	post_tokens = []
	recent_post_time = (
		datetime.now(timezone.utc) - timedelta(hours=1)
	).isoformat()
	monkeypatch.setattr(
		ai,
		"get_page_posts",
		lambda page_id, token, max_items=None, since=None, until=None: (
			post_tokens.append(token)
			or [
				{
					"id": "post_1",
					"message": "Original post",
					"created_time": recent_post_time,
				}
			]
		),
	)
	comment_tokens = []
	monkeypatch.setattr(
		ai,
		"get_all_comments_from_a_post",
		lambda post_id, token, max_items=None: (
			comment_tokens.append(token)
			or [
				{
					"id": "comment_1",
					"message": "This is bad",
					"from": {"name": "A"},
				}
			]
		),
	)

	def fake_classify(request, **_kwargs):
		nonlocal captured_request
		captured_request = request
		return [ClassificationResult.model_validate(result) for result in results]

	monkeypatch.setattr(ai, "classify_comments", fake_classify)

	response = client.post("/api/ai/classify/page_1")

	assert response.status_code == 200
	response_result = response.json()[0]
	assert response_result["comment"] == "This is bad"
	assert response_result["author"] == "A"
	assert response_result["platform"] == "facebook"
	assert response_result["page_name"] == "Lilongwe"
	assert response_result["category"] == "negative_sentiment"
	assert response_result["severity"] == "low"
	assert response_result["status"] == "pending"
	assert response_result["parent_post_id"] == "post_1"
	assert captured_request.location == "Lilongwe"
	assert captured_request.comments[0].comment_id == "comment_1"
	assert captured_request.comments[0].author == "A"
	assert post_tokens == ["page-token"]
	assert comment_tokens == ["page-token"]


def test_invalid_classification_type_is_rejected():
	with pytest.raises(ValidationError):
		ClassificationResult(
			location="Lilongwe",
			flagged=False,
			confidence=0.5,
			type="unknown",
			reason="not allowed",
		)


def test_confidence_outside_range_is_rejected():
	with pytest.raises(ValidationError):
		ClassificationResult(
			location="Lilongwe",
			flagged=False,
			confidence=1.1,
			type="none",
			reason="not allowed",
		)


def test_empty_comments_use_unparsable_fallback():
	request = _request()
	first_result = ClassificationResult(
		comment_id="comment_1",
		post_id="post_1",
		page_id="page_1",
		location="Lilongwe",
		flagged=False,
		confidence=0.8,
		type="none",
		reason="empty",
	)
	second_result = first_result.model_copy(update={"comment_id": None})

	validated = _validate_results(
		{"request": request, "results": [first_result, second_result]}
	)["results"]

	assert validated == []


def test_model_output_cannot_change_ids_or_result_count():
	request = _request()
	result = ClassificationResult(
		comment_id="wrong",
		post_id="post_1",
		page_id="page_1",
		location="Lilongwe",
		flagged=False,
		confidence=0.9,
		type="none",
		reason="benign",
	)

	with pytest.raises(AIOutputError):
		_validate_results({"request": request, "results": [result]})


def test_provider_errors_are_returned_without_details(monkeypatch):
	client = _client(monkeypatch)
	monkeypatch.setattr(
		ai,
		"classify_comments",
		lambda request, **_kwargs: (_ for _ in ()).throw(
			ai.AIProviderError("provider failure details")
		),
	)

	monkeypatch.setenv("META_ACCESS_TOKEN", "server-token")
	monkeypatch.setattr(ai, "get_page", lambda page_id, token: {"name": "Lilongwe"})
	monkeypatch.setattr(
		ai,
		"get_page_posts",
		lambda page_id, token, max_items=None, since=None, until=None: [],
	)
	monkeypatch.setattr(
		ai,
		"_build_page_request",
		lambda page_id, page_name, access_token, settings: _request(),
	)

	response = client.post("/api/ai/classify/page_1")

	assert response.status_code == 503
	assert response.json() == {
		"detail": "The classification provider is temporarily unavailable"
	}