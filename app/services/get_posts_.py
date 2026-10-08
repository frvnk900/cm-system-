from typing import Any

import httpx

from app.services.get_pages_ import (
	META_GRAPH_URL,
	META_HTTP_TIMEOUT,
	META_MAX_PAGES,
	MetaAPIError,
	_graph_get,
)


# updated_time moves whenever a post gets a new comment, so posts with no
# recent activity can be skipped without fetching their comments.
POST_FIELDS = "id,message,created_time,updated_time,permalink_url,full_picture,status_type"


def get_page_posts(
	page_id: str,
	page_access_token: str,
	max_items: int | None = None,
	since: int | None = None,
	until: int | None = None,
) -> list[dict[str, Any]]:
	"""Return every post on the given page, following Graph API pagination."""
	posts: list[dict[str, Any]] = []
	url: str | None = f"{META_GRAPH_URL}/{page_id}/posts"
	params: dict[str, Any] | None = {
		"access_token": page_access_token,
		"fields": POST_FIELDS,
		"limit": 100,
	}
	if since is not None:
		params["since"] = since
	if until is not None:
		params["until"] = until

	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		seen_urls: set[str] = set()
		page_count = 0
		while url:
			if url in seen_urls or page_count >= META_MAX_PAGES:
				raise MetaAPIError("Meta Graph API pagination did not terminate")
			seen_urls.add(url)
			page_count += 1
			data = _graph_get(client, url, params)
			posts.extend(data.get("data", []))
			if max_items is not None and len(posts) >= max_items:
				return posts[:max_items]
			# The "next" URL already carries the token and query params.
			url = data.get("paging", {}).get("next")
			params = None

	return posts


def get_post(post_id: str, page_access_token: str) -> dict[str, Any]:
	"""Return a single post from Meta by its post id."""
	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		return _graph_get(
			client,
			f"{META_GRAPH_URL}/{post_id}",
			{"access_token": page_access_token, "fields": POST_FIELDS},
		)
