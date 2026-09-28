import os
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

META_GRAPH_VERSION = os.getenv(
	"META_GRAPH_VERSION",
	os.getenv("META_GRAPH_API_VERSION", "v23.0"),
)
META_GRAPH_URL = f"https://graph.facebook.com/{META_GRAPH_VERSION}"
META_HTTP_TIMEOUT = float(os.getenv("META_HTTP_TIMEOUT", "30"))
META_MAX_PAGES = int(os.getenv("META_MAX_PAGES", "100"))
PAGE_FIELDS = "id,name,access_token,category,tasks"


class MetaAPIError(Exception):
	"""Raised when the Meta Graph API returns an error."""


class MetaAPIRequestError(MetaAPIError):
	"""Raised when a Meta Graph API request cannot complete."""


def _graph_get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
	try:
		response = client.get(url, params=params)
	except httpx.TimeoutException as error:
		raise MetaAPIRequestError("Meta Graph API request timed out") from error
	except httpx.RequestError as error:
		raise MetaAPIRequestError("Meta Graph API request failed") from error

	data = response.json()
	if response.is_error or "error" in data:
		message = data.get("error", {}).get("message", response.text)
		raise MetaAPIError(f"Meta Graph API error ({response.status_code}): {message}")
	return data


def get_pages(user_access_token: str) -> list[dict[str, Any]]:
	"""Return every page the user manages, following Graph API pagination."""
	pages: list[dict[str, Any]] = []
	url: str | None = f"{META_GRAPH_URL}/me/accounts"
	params: dict[str, Any] | None = {
		"access_token": user_access_token,
		"fields": PAGE_FIELDS,
		"limit": 100,
	}

	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		seen_urls: set[str] = set()
		page_count = 0
		while url:
			if url in seen_urls or page_count >= META_MAX_PAGES:
				raise MetaAPIError("Meta Graph API pagination did not terminate")
			seen_urls.add(url)
			page_count += 1
			data = _graph_get(client, url, params)
			pages.extend(data.get("data", []))
			# The "next" URL already carries the token and query params.
			url = data.get("paging", {}).get("next")
			params = None

	return pages


def get_page(page_id: str, access_token: str) -> dict[str, Any]:
	"""Return a single page from Meta by its page id."""
	with httpx.Client(timeout=META_HTTP_TIMEOUT) as client:
		return _graph_get(
			client,
			f"{META_GRAPH_URL}/{page_id}",
			{"access_token": access_token, "fields": "id,name,access_token,category"},
		)
