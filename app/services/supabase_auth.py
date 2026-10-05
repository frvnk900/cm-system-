"""Supabase Auth (GoTrue) calls used by the admin portal's accounts.

Only this module talks to Supabase Auth. The service role key never leaves
the server.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.settings import get_settings


logger = logging.getLogger(__name__)
TIMEOUT = 20.0


class SupabaseAuthError(Exception):
	"""Supabase Auth refused or could not complete a request."""

	def __init__(self, message: str, code: str = "", status: int = 0) -> None:
		super().__init__(message)
		self.code = code
		self.status = status


class SupabaseNotConfigured(SupabaseAuthError):
	"""SUPABASE_URL / keys are missing."""


def is_configured() -> bool:
	settings = get_settings()
	return bool(
		settings.supabase_url
		and settings.supabase_anon_key
		and settings.supabase_service_role_key
	)


def _request(
	method: str,
	path: str,
	*,
	key: str | None,
	bearer: str | None = None,
	json: dict[str, Any] | None = None,
	params: dict[str, Any] | None = None,
) -> dict[str, Any]:
	settings = get_settings()
	if not settings.supabase_url or not key:
		raise SupabaseNotConfigured("Supabase Auth is not configured on the server")
	headers = {"apikey": key}
	if bearer:
		headers["Authorization"] = f"Bearer {bearer}"
	try:
		response = httpx.request(
			method,
			f"{settings.supabase_url.rstrip('/')}/auth/v1{path}",
			headers=headers,
			json=json,
			params=params,
			timeout=TIMEOUT,
		)
	except httpx.HTTPError as error:
		raise SupabaseAuthError("Could not reach the sign-in service") from error

	try:
		data = response.json()
	except ValueError:
		data = {}
	if response.is_error:
		code = str(data.get("error_code") or data.get("error") or "")
		message = str(
			data.get("msg") or data.get("error_description") or data.get("message") or response.text
		)
		raise SupabaseAuthError(message, code=code, status=response.status_code)
	return data if isinstance(data, dict) else {}


def sign_in(email: str, password: str) -> dict[str, Any]:
	"""Check an email + password. Returns {"access_token", "user": {...}}."""
	return _request(
		"POST",
		"/token",
		key=get_settings().supabase_anon_key,
		params={"grant_type": "password"},
		json={"email": email, "password": password},
	)


def get_user(access_token: str) -> dict[str, Any]:
	"""The user an access token (from an emailed link) belongs to."""
	return _request(
		"GET", "/user", key=get_settings().supabase_anon_key, bearer=access_token
	)


def set_password(access_token: str, new_password: str) -> None:
	"""Set the password of the user who owns ``access_token``."""
	_request(
		"PUT",
		"/user",
		key=get_settings().supabase_anon_key,
		bearer=access_token,
		json={"password": new_password},
	)


def invite(email: str, redirect_to: str) -> None:
	"""Email an invitation link. Raises with code "email_exists" if already a user."""
	service_key = get_settings().supabase_service_role_key
	_request(
		"POST",
		"/invite",
		key=service_key,
		bearer=service_key,
		params={"redirect_to": redirect_to},
		json={"email": email},
	)


def send_password_link(email: str, redirect_to: str) -> None:
	"""Email a link that lets the user choose a new password."""
	_request(
		"POST",
		"/recover",
		key=get_settings().supabase_anon_key,
		params={"redirect_to": redirect_to},
		json={"email": email},
	)
