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


class MetaRateLimitError(MetaAPIError):
	"""Meta refused the call because a request limit was reached."""


class MetaAuthError(MetaAPIError):
	"""The access token was rejected (expired, revoked or invalid)."""


# Graph API error codes: 4 app limit, 17 user limit, 32 page limit, 613 custom
# limit, 80000-80014 business-use-case limits (Pages, Instagram).
RATE_LIMIT_CODES = {4, 17, 32, 613} | set(range(80000, 80015))
AUTH_ERROR_CODES = {102, 190}


def meta_error(status_code: int, payload: Any, fallback_text: str = "") -> MetaAPIError:
	"""Build the right MetaAPIError subclass from a Graph API error response."""
	error = payload.get("error", {}) if isinstance(payload, dict) else {}
	message = f"Meta Graph API error ({status_code}): {error.get('message', fallback_text)}"
	code = error.get("code")
	if status_code == 429 or code in RATE_LIMIT_CODES:
		return MetaRateLimitError(message)
	if code in AUTH_ERROR_CODES:
		return MetaAuthError(message)
	return MetaAPIError(message)


def _graph_get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
	try:
		response = client.get(url, params=params)
	except httpx.TimeoutException as error:
		raise MetaAPIRequestError("Meta Graph API request timed out") from error
	except httpx.RequestError as error:
		raise MetaAPIRequestError("Meta Graph API request failed") from error

	data = response.json()
	if response.is_error or "error" in data:
		raise meta_error(response.status_code, data, response.text)
	return data


def get_pages(user_access_token: str, fields: str = PAGE_FIELDS) -> list[dict[str, Any]]:
	"""Return every page the user manages, following Graph API pagination."""
	pages: list[dict[str, Any]] = []
	url: str | None = f"{META_GRAPH_URL}/me/accounts"
	params: dict[str, Any] | None = {
		"access_token": user_access_token,
		"fields": fields,
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
