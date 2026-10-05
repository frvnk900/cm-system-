from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.db.db import get_db
from app.api.routes import admin
from app.api.schema.admin_user_model import AdminUser
from app.services.supabase_auth import SupabaseAuthError


SETUP_PASSWORD = "setup-pass"
API = admin.ADMIN_PATH + "/api"


class FakeSupabase:
	"""Stands in for Supabase Auth: remembers passwords and 'sent' emails."""

	def __init__(self) -> None:
		self.passwords: dict[str, str] = {}
		self.tokens: dict[str, str] = {}   # access token -> email
		self.invites: list[tuple[str, str]] = []
		self.password_links: list[tuple[str, str]] = []
		self.existing_users: set[str] = set()

	def is_configured(self) -> bool:
		return True

	def invite(self, email: str, redirect_to: str) -> None:
		if email in self.existing_users:
			raise SupabaseAuthError("already registered", code="email_exists", status=422)
		self.existing_users.add(email)
		self.invites.append((email, redirect_to))
		self.tokens[f"link-{email}"] = email

	def send_password_link(self, email: str, redirect_to: str) -> None:
		self.password_links.append((email, redirect_to))
		self.tokens[f"link-{email}"] = email

	def get_user(self, access_token: str) -> dict:
		if access_token not in self.tokens:
			raise SupabaseAuthError("invalid token", code="bad_jwt", status=401)
		return {"email": self.tokens[access_token]}

	def set_password(self, access_token: str, new_password: str) -> None:
		email = self.tokens[access_token]
		if self.passwords.get(email) == new_password:
			raise SupabaseAuthError("same", code="same_password", status=422)
		self.passwords[email] = new_password

	def sign_in(self, email: str, password: str) -> dict:
		if self.passwords.get(email) != password:
			raise SupabaseAuthError("Invalid login credentials", code="invalid_credentials", status=400)
		self.tokens[f"session-{email}"] = email
		return {"access_token": f"session-{email}", "user": {"email": email}}


@pytest.fixture
def portal(monkeypatch):
	"""A fresh portal: empty admin list, fake Supabase, setup password configured."""
	engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
	AdminUser.__table__.create(engine)
	Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

	def override_db():
		database = Session()
		try:
			yield database
		finally:
			database.close()

	fake = FakeSupabase()
	monkeypatch.setattr(admin, "supabase_auth", fake)
	monkeypatch.setattr(admin, "get_settings", lambda: SimpleNamespace(
		admin_password=SETUP_PASSWORD, admin_session_secret="test-secret",
		admin_session_hours=12, public_base_url=None,
	))
	admin._login_failures.clear()

	application = FastAPI()
	application.include_router(admin.router)
	application.dependency_overrides[get_db] = override_db
	return SimpleNamespace(app=application, supabase=fake, new_client=lambda: TestClient(application))


def _create_admin(portal, email: str, password: str, via: TestClient) -> None:
	"""Invite ``email`` using an already signed-in client, then choose a password."""
	assert via.post(f"{API}/admins", json={"email": email}).status_code == 200
	visitor = portal.new_client()
	response = visitor.post(f"{API}/password/set", json={"access_token": f"link-{email}", "password": password})
	assert response.status_code == 200, response.text


def _signed_in(portal, email: str, password: str) -> TestClient:
	client = portal.new_client()
	assert client.post(f"{API}/login", json={"email": email, "password": password}).status_code == 200
	return client


def _first_admin(portal, email="owner@example.com", password="owner-password") -> TestClient:
	setup = portal.new_client()
	assert setup.post(f"{API}/login", json={"password": SETUP_PASSWORD}).status_code == 200
	_create_admin(portal, email, password, via=setup)
	return _signed_in(portal, email, password)


def test_portal_is_closed_without_a_session(portal):
	client = portal.new_client()

	for path in ("/admins", "/settings", "/pages", "/me"):
		assert client.get(API + path).status_code == 401


def test_first_admin_is_created_with_the_setup_password_which_then_stops_working(portal):
	setup = portal.new_client()
	assert setup.get(f"{API}/auth/status").json()["setup"] is True
	assert setup.post(f"{API}/login", json={"password": "wrong"}).status_code == 401
	assert setup.post(f"{API}/login", json={"password": SETUP_PASSWORD}).status_code == 200
	assert setup.get(f"{API}/me").json() == {"ok": True, "email": None, "setup": True}

	# Invite yourself; the emailed link points back at the portal.
	assert setup.post(f"{API}/admins", json={"email": "Owner@Example.com"}).status_code == 200
	assert portal.supabase.invites == [("owner@example.com", "http://testserver" + admin.ADMIN_PATH)]
	listed = setup.get(f"{API}/admins").json()["admins"]
	assert [(a["email"], a["status"]) for a in listed] == [("owner@example.com", "invited")]
	# Still invited only, so the setup password keeps working (a typo can't lock you out).
	assert portal.new_client().post(f"{API}/login", json={"password": SETUP_PASSWORD}).status_code == 200

	# Open the link and choose a password: now signed in as that account.
	owner = portal.new_client()
	done = owner.post(f"{API}/password/set", json={"access_token": "link-owner@example.com", "password": "owner-password"})
	assert done.status_code == 200
	assert owner.get(f"{API}/me").json() == {"ok": True, "email": "owner@example.com", "setup": False}

	# The shared password is now off, and the old setup session is dead.
	assert portal.new_client().get(f"{API}/auth/status").json()["setup"] is False
	assert portal.new_client().post(f"{API}/login", json={"password": SETUP_PASSWORD}).status_code == 403
	assert setup.get(f"{API}/admins").status_code == 401


def test_sign_in_needs_an_invited_email_and_the_right_password(portal):
	_first_admin(portal)

	wrong_password = portal.new_client().post(f"{API}/login", json={"email": "owner@example.com", "password": "nope"})
	stranger = portal.new_client().post(f"{API}/login", json={"email": "stranger@example.com", "password": "x"})

	assert wrong_password.status_code == 401
	assert stranger.status_code == 401
	# Same message either way: it doesn't reveal who is an admin.
	assert wrong_password.json() == stranger.json() == {"detail": "Wrong email or password"}


def test_a_valid_supabase_user_who_was_not_invited_cannot_get_in(portal):
	_first_admin(portal)
	portal.supabase.passwords["outsider@example.com"] = "their-password"
	portal.supabase.tokens["outsider-link"] = "outsider@example.com"

	sign_in = portal.new_client().post(f"{API}/login", json={"email": "outsider@example.com", "password": "their-password"})
	set_password = portal.new_client().post(f"{API}/password/set", json={"access_token": "outsider-link", "password": "new-password"})

	assert sign_in.status_code == 401
	assert set_password.status_code == 403
	assert portal.supabase.passwords["outsider@example.com"] == "their-password"  # untouched


def test_an_admin_can_add_and_remove_another_admin(portal):
	owner = _first_admin(portal)
	_create_admin(portal, "second@example.com", "second-password", via=owner)
	second = _signed_in(portal, "second@example.com", "second-password")
	assert second.get(f"{API}/me").json()["email"] == "second@example.com"

	assert owner.post(f"{API}/admins", json={"email": "second@example.com"}).status_code == 409
	assert owner.delete(f"{API}/admins/owner@example.com").status_code == 400  # not yourself
	assert owner.delete(f"{API}/admins/second@example.com").status_code == 200

	# Removed admins are locked out immediately, even with a live session.
	assert second.get(f"{API}/admins").status_code == 401
	assert portal.new_client().post(f"{API}/login", json={"email": "second@example.com", "password": "second-password"}).status_code == 401


def test_the_last_admin_cannot_be_removed(portal):
	owner = _first_admin(portal)
	_create_admin(portal, "second@example.com", "second-password", via=owner)
	second = _signed_in(portal, "second@example.com", "second-password")
	assert second.delete(f"{API}/admins/owner@example.com").status_code == 200

	# Only "second" is left; a pending invite doesn't count as an admin who can sign in.
	assert second.post(f"{API}/admins", json={"email": "pending@example.com"}).status_code == 200
	assert second.delete(f"{API}/admins/second@example.com").status_code == 400
	assert second.delete(f"{API}/admins/pending@example.com").status_code == 200


def test_change_password_needs_the_current_one(portal):
	owner = _first_admin(portal)

	wrong = owner.post(f"{API}/account/password", json={"current_password": "nope", "new_password": "brand-new-password"})
	short = owner.post(f"{API}/account/password", json={"current_password": "owner-password", "new_password": "short"})
	changed = owner.post(f"{API}/account/password", json={"current_password": "owner-password", "new_password": "brand-new-password"})

	assert wrong.status_code == 400
	assert short.status_code == 422
	assert changed.status_code == 200
	assert portal.new_client().post(f"{API}/login", json={"email": "owner@example.com", "password": "owner-password"}).status_code == 401
	assert portal.new_client().post(f"{API}/login", json={"email": "owner@example.com", "password": "brand-new-password"}).status_code == 200


def test_forgot_password_only_emails_admins_and_never_says_which(portal):
	_first_admin(portal)

	known = portal.new_client().post(f"{API}/password/forgot", json={"email": "owner@example.com"})
	unknown = portal.new_client().post(f"{API}/password/forgot", json={"email": "stranger@example.com"})

	assert known.json() == unknown.json() == {"ok": True}
	assert [email for email, _ in portal.supabase.password_links] == ["owner@example.com"]


def test_inviting_someone_who_already_has_a_login_sends_a_password_link_instead(portal):
	owner = _first_admin(portal)
	portal.supabase.existing_users.add("existing@example.com")

	response = owner.post(f"{API}/admins", json={"email": "existing@example.com"})

	assert response.status_code == 200
	assert response.json()["outcome"] == "password link sent"
	assert portal.supabase.password_links[-1][0] == "existing@example.com"


def test_repeated_failed_sign_ins_are_locked_out(portal):
	_first_admin(portal)
	client = portal.new_client()

	codes = [client.post(f"{API}/login", json={"email": "owner@example.com", "password": "nope"}).status_code for _ in range(6)]

	assert codes == [401] * 5 + [429]
