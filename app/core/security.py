"""API key protection for the API routes."""

import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.settings import get_settings


# Declared as a security scheme so /docs shows an "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(api_key_header)) -> None:
	"""Reject requests without the configured API key.

	While API_KEY is not set the API stays open, so the key can be given to
	clients (the Sheet) before it is switched on.
	"""
	expected = get_settings().api_key
	if not expected:
		return
	if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
		raise HTTPException(status_code=401, detail="Missing or invalid API key")
