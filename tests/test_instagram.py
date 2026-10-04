from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.db.db import get_db
from app.api.routes import ai
from app.api.routes.comments import comments
from app.api.schema.ai_model import ClassificationResult, SentimentOutput
from app.core.settings import default_runtime_settings
from app.services import instagram_
from app.services.get_pages_ import MetaAPIError


def _fake_instagram(monkeypatch, recent: str, old: str) -> None:
	monkeypatch.setattr(
		ai,
		"get_instagram_media",
		lambda ig_id, token, newer_than, max_items: [
			{"id": "media_1", "caption": "New drop", "permalink": "https://instagram.com/p/x", "comments_count": 2}
		],
	)

	def fake_ig_comments(media, token, newer_than, max_items):
		items = [
			{"id": "ig_c1", "message": "You are scammers", "from": {"name": "angry_user"},
			 "created_time": recent, "permalink_url": media["permalink"], "platform": "instagram"},
			{"id": "ig_c2", "message": "old one", "from": None,
			 "created_time": old, "permalink_url": media["permalink"], "platform": "instagram"},
		]
		# The real service filters by time; mirror that here.
		return [c for c in items if datetime.fromisoformat(c["created_time"]) >= newer_than]

	monkeypatch.setattr(ai, "get_instagram_comments", fake_ig_comments)


def test_instagram_comments_are_collected_and_tagged(monkeypatch):
	now = datetime.now(timezone.utc)
	recent = (now - timedelta(hours=1)).isoformat()
	old = (now - timedelta(days=10)).isoformat()
	monkeypatch.setattr(ai, "get_page_posts", lambda *args, **kwargs: [])
	_fake_instagram(monkeypatch, recent, old)

	request = ai._build_page_request(
		"page_1", "Lilongwe", "token", default_runtime_settings(), instagram_id="ig_1"
	)

	assert request is not None
	assert [c.comment_id for c in request.comments] == ["ig_c1"]
	comment = request.comments[0]
	assert comment.platform == "instagram"
	assert comment.author == "angry_user"
	assert request.posts[0].post_id == "media_1"

	result = ClassificationResult(
		comment_id="ig_c1", post_id="media_1", page_id="page_1", location="Lilongwe",
		flagged=True, confidence=0.8, type="defamation", reason="Accusation.",
	)
	formatted = ai._format_results([result], request)[0]
	assert formatted.platform == "instagram"
	assert formatted.author == "angry_user"


def test_facebook_and_instagram_are_merged_newest_first(monkeypatch):
	now = datetime.now(timezone.utc)
	recent = (now - timedelta(hours=1)).isoformat()
	older = (now - timedelta(hours=5)).isoformat()
	monkeypatch.setattr(
		ai, "get_page_posts",
		lambda *args, **kwargs: [{"id": "post_1", "message": "Hi", "created_time": older}],
	)
	monkeypatch.setattr(
		ai, "get_all_comments_from_a_post",
		lambda *args, **kwargs: [{"id": "fb_c1", "message": "bad", "created_time": older}],
	)
	_fake_instagram(monkeypatch, recent, (now - timedelta(days=10)).isoformat())

	request = ai._build_page_request(
		"page_1", "Lilongwe", "token", default_runtime_settings(), instagram_id="ig_1"
	)

	assert [(c.comment_id, c.platform) for c in request.comments] == [
		("ig_c1", "instagram"),
		("fb_c1", "facebook"),
	]


def test_instagram_failure_does_not_block_facebook(monkeypatch):
	recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
	monkeypatch.setattr(
		ai, "get_page_posts",
		lambda *args, **kwargs: [{"id": "post_1", "message": "Hi", "created_time": recent}],
	)
	monkeypatch.setattr(
		ai, "get_all_comments_from_a_post",
		lambda *args, **kwargs: [{"id": "fb_c1", "message": "bad", "created_time": recent}],
	)

	def broken_media(*args, **kwargs):
		raise MetaAPIError("instagram permission missing")

	monkeypatch.setattr(ai, "get_instagram_media", broken_media)

	request = ai._build_page_request(
		"page_1", "Lilongwe", "token", default_runtime_settings(), instagram_id="ig_1"
	)

	assert [c.comment_id for c in request.comments] == ["fb_c1"]
	assert request.comments[0].platform == "facebook"


def test_instagram_is_skipped_when_switched_off_or_not_linked():
	linked = SimpleNamespace(ig_user_id="ig_1", instagram_enabled=True)
	switched_off = SimpleNamespace(ig_user_id="ig_1", instagram_enabled=False)
	not_linked = SimpleNamespace(ig_user_id=None, instagram_enabled=True)

	assert ai.instagram_id_for(linked) == "ig_1"
	assert ai.instagram_id_for(switched_off) is None
	assert ai.instagram_id_for(not_linked) is None
	assert ai.instagram_id_for(None) is None


def test_import_reads_instagram_in_the_same_request_and_survives_missing_permission(monkeypatch):
	calls = []

	def fake_get_pages(token, fields=instagram_.PAGE_FIELDS):
		calls.append(fields)
		if "instagram_business_account" in fields:
			raise MetaAPIError("(#10) permission required")
		return [{"id": "1", "name": "Page"}]

	monkeypatch.setattr(instagram_, "get_pages", fake_get_pages)

	pages = instagram_.get_pages_with_instagram("token")

	assert pages == [{"id": "1", "name": "Page"}]
	assert len(calls) == 2  # tried with Instagram, then fell back without
	assert instagram_.instagram_account_of(pages[0]) is None
	assert instagram_.instagram_account_of(
		{"instagram_business_account": {"id": 5, "username": "shop"}}
	) == {"id": "5", "username": "shop"}


def test_sentiment_route_includes_instagram_and_delete_uses_page_token(monkeypatch):
	recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
	stored_page = SimpleNamespace(is_active=True, ig_user_id="ig_1", instagram_enabled=True)

	class FakeSession:
		def scalar(self, *_args, **_kwargs):
			return stored_page

	application = FastAPI()
	application.include_router(comments.router)
	application.dependency_overrides[get_db] = lambda: FakeSession()
	client = TestClient(application)

	monkeypatch.setenv("META_ACCESS_TOKEN", "user-token")
	monkeypatch.setattr(comments, "load_runtime_settings", lambda _db: default_runtime_settings())
	monkeypatch.setattr(comments, "get_page", lambda page_id, token: {"name": "Lilongwe", "access_token": "page-token"})
	monkeypatch.setattr(comments, "load_sentiments", lambda _db, _c: {})
	monkeypatch.setattr(comments, "save_sentiments", lambda *args: None)
	monkeypatch.setattr(ai, "get_page_posts", lambda *args, **kwargs: [])
	_fake_instagram(monkeypatch, recent, (datetime.now(timezone.utc) - timedelta(days=10)).isoformat())
	monkeypatch.setattr(
		comments, "classify_sentiment",
		lambda request, settings, on_batch=None: {
			"ig_c1": SentimentOutput(comment_id="ig_c1", sentiment="negative", confidence=0.9, reason="Accusation."),
		},
	)

	rows = client.get("/comments/page/page_1/sentiment").json()

	assert [(r["comment_id"], r["platform"], r["sentiment"]) for r in rows] == [
		("ig_c1", "instagram", "negative")
	]

	# Deleting an Instagram comment is the same call, made with the page token.
	used = {}
	monkeypatch.setattr(
		comments, "delete_comment",
		lambda comment_id, token: used.update(id=comment_id, token=token) or {"success": True},
	)
	response = client.delete("/comments/ig_c1?page_id=page_1")

	assert response.status_code == 200
	assert used == {"id": "ig_c1", "token": "page-token"}
