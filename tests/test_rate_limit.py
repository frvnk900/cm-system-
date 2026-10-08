from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import ai
from app.api.schema.fetch_state_model import PageFetchCache, ServiceState
from app.core.settings import default_runtime_settings
from app.services.get_pages_ import (
	MetaAPIError,
	MetaAuthError,
	MetaRateLimitError,
	meta_error,
)


SETTINGS = default_runtime_settings()


@pytest.fixture
def database():
	engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
	PageFetchCache.__table__.create(engine)
	ServiceState.__table__.create(engine)
	session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
	yield session
	session.close()


def _stored_page(token: str = "saved-token") -> SimpleNamespace:
	return SimpleNamespace(
		name="Lilongwe", access_token=token, is_active=True, ig_user_id=None, instagram_enabled=True
	)


def _recent() -> str:
	return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


class Meta:
	"""Fake Meta: counts every call so tests can assert how many were made."""

	def __init__(self, monkeypatch, posts=None, comments=None, fail_with=None, fail_tokens=()):
		self.calls: list[tuple[str, str]] = []
		self.posts = posts if posts is not None else [
			{"id": "post_1", "message": "Hi", "created_time": _recent(), "updated_time": _recent()}
		]
		self.comments = comments if comments is not None else [
			{"id": "c1", "message": "bad service", "created_time": _recent()}
		]
		self.fail_with = fail_with
		self.fail_tokens = set(fail_tokens)
		monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
		monkeypatch.setattr(ai, "get_page", self.get_page)
		monkeypatch.setattr(ai, "get_page_posts", self.get_posts)
		monkeypatch.setattr(ai, "get_all_comments_from_a_post", self.get_comments)

	def get_page(self, page_id, token):
		self.calls.append(("page lookup", token))
		return {"name": "Lilongwe", "access_token": "fresh-token"}

	def get_posts(self, page_id, token, **_kwargs):
		self.calls.append(("posts", token))
		if token in self.fail_tokens:
			raise MetaAuthError("Session has expired")
		if self.fail_with is not None:
			raise self.fail_with
		return self.posts

	def get_comments(self, post_id, token, **_kwargs):
		self.calls.append(("comments", token))
		return self.comments


def test_meta_errors_are_classified():
	assert isinstance(meta_error(403, {"error": {"code": 4, "message": "Application request limit reached"}}), MetaRateLimitError)
	assert isinstance(meta_error(400, {"error": {"code": 80001, "message": "page limit"}}), MetaRateLimitError)
	assert isinstance(meta_error(429, {}), MetaRateLimitError)
	assert isinstance(meta_error(400, {"error": {"code": 190, "message": "Session has expired"}}), MetaAuthError)
	other = meta_error(400, {"error": {"code": 100, "message": "Object does not exist"}})
	assert type(other) is MetaAPIError and "does not exist" in str(other)


def test_saved_page_token_is_used_without_a_user_token_lookup(monkeypatch, database):
	meta = Meta(monkeypatch)

	request = ai.collect_page_comments(database, "page_1", _stored_page(), SETTINGS)

	assert [c.comment_id for c in request.comments] == ["c1"]
	assert meta.calls == [("posts", "saved-token"), ("comments", "saved-token")]


def test_posts_with_no_recent_activity_are_skipped(monkeypatch, database):
	quiet = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
	posts = [
		{"id": "busy", "message": "A", "created_time": quiet, "updated_time": _recent()},
		{"id": "quiet_1", "message": "B", "created_time": quiet, "updated_time": quiet},
		{"id": "quiet_2", "message": "C", "created_time": quiet, "updated_time": quiet},
	]
	meta = Meta(monkeypatch, posts=posts)

	ai.collect_page_comments(database, "page_1", _stored_page(), SETTINGS)

	# One comments request (the post with activity), not three.
	assert [call for call, _ in meta.calls] == ["posts", "comments"]


def test_moderation_and_sentiment_share_one_fetch(monkeypatch, database):
	meta = Meta(monkeypatch)
	page = _stored_page()

	first = ai.collect_page_comments(database, "page_1", page, SETTINGS)
	calls_after_first = len(meta.calls)
	second = ai.collect_page_comments(database, "page_1", page, SETTINGS)

	assert len(meta.calls) == calls_after_first  # second call made no Meta requests
	assert second.model_dump() == first.model_dump()


def test_empty_pages_are_cached_too(monkeypatch, database):
	meta = Meta(monkeypatch, posts=[])
	page = _stored_page()

	assert ai.collect_page_comments(database, "page_1", page, SETTINGS) is None
	assert ai.collect_page_comments(database, "page_1", page, SETTINGS) is None
	assert len(meta.calls) == 1


def test_changed_settings_are_not_served_from_cache(monkeypatch, database):
	meta = Meta(monkeypatch)
	page = _stored_page()
	ai.collect_page_comments(database, "page_1", page, SETTINGS)
	calls = len(meta.calls)

	ai.collect_page_comments(database, "page_1", page, SETTINGS.model_copy(update={"lookback_days": 5}))

	assert len(meta.calls) > calls


def test_rejected_saved_token_is_refreshed_saved_and_retried(monkeypatch, database):
	meta = Meta(monkeypatch, fail_tokens={"saved-token"})
	page = _stored_page()

	request = ai.collect_page_comments(database, "page_1", page, SETTINGS)

	assert request is not None
	assert page.access_token == "fresh-token"  # repaired for next time
	assert meta.calls == [
		("posts", "saved-token"),
		("page lookup", "user-token"),
		("posts", "fresh-token"),
		("comments", "fresh-token"),
	]


def test_rate_limit_pauses_all_meta_calls(monkeypatch, database):
	meta = Meta(monkeypatch, fail_with=MetaRateLimitError("(#4) Application request limit reached"))

	with pytest.raises(HTTPException) as first:
		ai.collect_page_comments(database, "page_1", _stored_page(), SETTINGS)
	assert first.value.status_code == 429
	assert "Retry-After" in first.value.headers
	calls = len(meta.calls)

	# Every other location now gets 429 immediately, with no Meta request at all.
	meta.fail_with = None
	with pytest.raises(HTTPException) as second:
		ai.collect_page_comments(database, "page_2", _stored_page(), SETTINGS)
	assert second.value.status_code == 429
	assert len(meta.calls) == calls


def test_calls_resume_after_the_cooldown(monkeypatch, database):
	meta = Meta(monkeypatch)
	expired = datetime.now(timezone.utc) - timedelta(minutes=1)
	database.add(ServiceState(key="meta_cooldown_until", value=expired.isoformat()))
	database.commit()

	assert ai.collect_page_comments(database, "page_1", _stored_page(), SETTINGS) is not None
	assert meta.calls
