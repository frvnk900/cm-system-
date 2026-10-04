"""Instagram Graph API: business accounts linked to Facebook Pages.

Instagram uses the linked Page's access token. Comments are normalised to the
same shape as Facebook comments (id, message, from.name, created_time,
permalink_url) plus ``platform="instagram"``. Deleting an Instagram comment
uses the same Graph call as Facebook: DELETE /{comment-id} with the Page token.
"""

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx

from app.services.get_comments_ import parse_meta_time
from app.services.get_pages_ import (
	META_GRAPH_URL,
	META_HTTP_TIMEOUT,
	META_MAX_PAGES,
	PAGE_FIELDS,
	MetaAPIError,
	_graph_get,
	get_pages,
)


logger = logging.getLogger(__name__)
PAGE_FIELDS_WITH_INSTAGRAM = f"{PAGE_FIELDS},instagram_business_account{{id,username}}"


def get_pages_with_instagram(user_access_token: str) -> list[dict[str, Any]]:
	"""Pages the user manages, each with its linked Instagram account if any.

	One request covers both. If the token lacks Instagram permissions the
	pages are still returned, just without Instagram accounts.
	"""
	try:
		return get_pages(user_access_token, PAGE_FIELDS_WITH_INSTAGRAM)
	except MetaAPIError as error:
		logger.warning("Instagram accounts could not be listed with pages: %s", error)
		return get_pages(user_access_token)


def instagram_account_of(page_data: dict[str, Any]) -> dict[str, str] | None:
	"""{"id", "username"} from a page returned by get_pages_with_instagram."""
	account = page_data.get("instagram_business_account")
	if not account or not account.get("id"):
		return None
	return {"id": str(account["id"]), "username": str(account.get("username") or "")}


def _paginate(client: httpx.Client, url: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
	"""Yield items across pages of a Graph API edge."""
	seen_urls: set[str] = set()
	page_count = 0
	next_url: str | None = url
	next_params: dict[str, Any] | None = params
	while next_url:
		if next_url in seen_urls or page_count >= META_MAX_PAGES:
			raise MetaAPIError("Meta Graph API pagination did not terminate")
		seen_urls.add(next_url)
		page_count += 1
		data = _graph_get(client, next_url, next_params)
		yield from data.get("data", [])
		next_url = data.get("paging", {}).get("next")
		next_params = None


def get_instagram_media(
	ig_user_id: str,
	page_access_token: str,
	newer_than: datetime,
	max_items: int,
) -> list[dict[str, Any]]:
	"""Return media (newest first) published after ``newer_than``."""
	media: list[dict[str, Any]] = []
	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		for item in _paginate(
			client,
			f"{META_GRAPH_URL}/{ig_user_id}/media",
			{
				"access_token": page_access_token,
				"fields": "id,caption,timestamp,permalink,comments_count",
				"limit": 50,
			},
		):
			published = parse_meta_time(item.get("timestamp"))
			if published is not None and published < newer_than:
				break  # media is returned newest first
			media.append(item)
			if len(media) >= max_items:
				break
	return media


def get_instagram_comments(
	media: dict[str, Any],
	page_access_token: str,
	newer_than: datetime,
	max_items: int,
) -> list[dict[str, Any]]:
	"""Return top-level comments on a media item written after ``newer_than``."""
	if not media.get("comments_count"):
		return []  # saves a request for posts nobody commented on
	comments: list[dict[str, Any]] = []
	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		for item in _paginate(
			client,
			f"{META_GRAPH_URL}/{media['id']}/comments",
			{
				"access_token": page_access_token,
				"fields": "id,text,username,timestamp",
				"limit": 50,
			},
		):
			written = parse_meta_time(item.get("timestamp"))
			if written is None or written < newer_than:
				continue  # comment order isn't guaranteed; filter each one
			comments.append(
				{
					"id": item.get("id"),
					"message": item.get("text") or "",
					"from": {"name": item.get("username")} if item.get("username") else None,
					"created_time": item.get("timestamp"),
					"permalink_url": media.get("permalink") or "",
					"platform": "instagram",
				}
			)
			if len(comments) >= max_items:
				break
	return comments
