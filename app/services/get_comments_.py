from datetime import datetime, timezone
from typing import Any

import httpx

from app.services.get_pages_ import (
	META_GRAPH_URL,
	META_HTTP_TIMEOUT,
	META_MAX_PAGES,
	MetaAPIError,
	_graph_get,
	meta_error,
)
from app.services.get_posts_ import get_page_posts


COMMENT_FIELDS = "id,message,from,created_time,attachment,parent,permalink_url"


def parse_meta_time(value: Any) -> datetime | None:
	"""Parse a Graph API timestamp such as 2026-09-29T10:45:38+0000."""
	if not value:
		return None
	try:
		return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
			timezone.utc
		)
	except ValueError:
		return None


def get_all_comments_from_a_post(
	post_id: str,
	page_access_token: str,
	max_items: int | None = None,
	newer_than: datetime | None = None,
) -> list[dict[str, Any]]:
	"""Return comments on a post, following Graph API pagination.

	With ``newer_than``, comments are read newest first and reading stops at
	the first comment written before that time.
	"""
	comments: list[dict[str, Any]] = []
	url: str | None = f"{META_GRAPH_URL}/{post_id}/comments"
	params: dict[str, Any] | None = {
		"access_token": page_access_token,
		"fields": COMMENT_FIELDS,
		"limit": 100,
	}
	if newer_than is not None:
		params["order"] = "reverse_chronological"

	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		seen_urls: set[str] = set()
		page_count = 0
		while url:
			if url in seen_urls or page_count >= META_MAX_PAGES:
				raise MetaAPIError("Meta Graph API pagination did not terminate")
			seen_urls.add(url)
			page_count += 1
			data = _graph_get(client, url, params)
			for comment in data.get("data", []):
				if newer_than is not None:
					created_at = parse_meta_time(comment.get("created_time"))
					if created_at is not None and created_at < newer_than:
						return comments
				comments.append(comment)
				if max_items is not None and len(comments) >= max_items:
					return comments
			url = data.get("paging", {}).get("next")
			params = None

	return comments


def get_all_comments_from_page(
	page_id: str, page_access_token: str
) -> list[dict[str, Any]]:
	"""Return page comments with their page and post ids attached."""
	comments: list[dict[str, Any]] = []

	for post in get_page_posts(page_id, page_access_token):
		post_id = post.get("id")
		if not post_id:
			continue

		for comment in get_all_comments_from_a_post(post_id, page_access_token):
			comments.append({**comment, "page_id": page_id, "post_id": post_id})

	return comments


def get_comment(comment_id: str, page_access_token: str) -> dict[str, Any]:
	"""Return a single comment from Meta by its comment id."""
	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		return _graph_get(
			client,
			f"{META_GRAPH_URL}/{comment_id}",
			{"access_token": page_access_token, "fields": COMMENT_FIELDS},
		)


def delete_comment(comment_id: str, page_access_token: str) -> dict[str, Any]:
	"""Delete a comment and return Meta's response."""
	with httpx.Client(timeout=10.0) as client:
		response = client.delete(
			f"{META_GRAPH_URL}/{comment_id}",
			params={"access_token": page_access_token},
		)
		data = response.json()
		if response.is_error or "error" in data:
			raise meta_error(response.status_code, data, response.text)
		return data
