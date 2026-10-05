import hmac
import logging
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.db.admin_users_ import (
	add_admin,
	count_active_admins,
	get_admin,
	list_admins,
	normalize_email,
	record_sign_in,
	remove_admin,
)
from app.api.db.app_settings_ import (
	load_runtime_settings,
	reset_runtime_settings,
	save_runtime_settings,
)
from app.api.db.db import get_db
from app.api.db.save_page_creds_ import save_page_credentials
from app.api.db.system_prompt_ import (
	latest_prompt_version,
	list_prompt_versions,
	reset_system_prompt,
	save_system_prompt,
)
from app.api.routes.ai import SEVERITY_BY_CATEGORY
from app.api.schema.classified_comment_model import ClassifiedComment
from app.api.schema.pages_model import Page, PageCreate
from app.api.schema.system_prompt_model import SystemPromptVersion
from app.core.settings import RuntimeSettings, default_runtime_settings, get_settings
from app.services import supabase_auth
from app.services.get_pages_ import MetaAPIError
from app.services.instagram_ import get_pages_with_instagram, instagram_account_of
from app.services.supabase_auth import SupabaseAuthError, SupabaseNotConfigured
from app.services.prompt.system_prompt import SYSTEM_PROMPT


logger = logging.getLogger(__name__)
ADMIN_PATH = get_settings().admin_path
router = APIRouter(prefix=ADMIN_PATH, tags=["admin"], include_in_schema=False)

ADMIN_PAGE = Path(__file__).resolve().parents[2] / "admin" / "index.html"
SESSION_COOKIE = "admin_session"
# Used only when ADMIN_SESSION_SECRET is unset; sessions then end on restart.
_FALLBACK_SECRET = secrets.token_urlsafe(32)
MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60
PASSWORD_MIN = 8
PASSWORD_MAX = 72
_login_failures: dict[str, deque[float]] = defaultdict(deque)


# ---------- auth ----------

def _serializer() -> URLSafeTimedSerializer:
	secret = get_settings().admin_session_secret or _FALLBACK_SECRET
	return URLSafeTimedSerializer(secret, salt="admin-session")


class AdminIdentity(BaseModel):
	"""Who is signed in. ``email`` is None for the one-time setup session."""

	email: str | None = None

	@property
	def is_setup(self) -> bool:
		return self.email is None

	@property
	def label(self) -> str:
		return self.email or "first-time setup"


def setup_login_allowed(database: Session) -> bool:
	"""The shared ADMIN_PASSWORD works only until one admin account is active."""
	return bool(get_settings().admin_password) and count_active_admins(database) == 0


def require_admin(request: Request, database: Session = Depends(get_db)) -> AdminIdentity:
	token = request.cookies.get(SESSION_COOKIE)
	if not token:
		raise HTTPException(status_code=401, detail="Not signed in")
	try:
		payload = _serializer().loads(token, max_age=get_settings().admin_session_hours * 3600)
	except BadSignature as error:
		raise HTTPException(status_code=401, detail="Session expired") from error

	email = payload.get("email") if isinstance(payload, dict) else None
	if email:
		# Checked on every request, so a removed admin is locked out at once.
		if get_admin(database, email) is None:
			raise HTTPException(status_code=401, detail="This account no longer has access")
		return AdminIdentity(email=email)
	# A setup session (shared password) ends as soon as a real account exists.
	if not setup_login_allowed(database):
		raise HTTPException(status_code=401, detail="Setup is finished; sign in with your account")
	return AdminIdentity()


def _recent_failures(client_ip: str) -> deque[float]:
	failures = _login_failures[client_ip]
	cutoff = time.monotonic() - LOGIN_LOCKOUT_SECONDS
	while failures and failures[0] < cutoff:
		failures.popleft()
	return failures


def _client_ip(request: Request) -> str:
	return request.client.host if request.client else "unknown"


def _guard_attempts(request: Request) -> deque[float]:
	"""Shared limiter for sign-in, password links and set-password attempts."""
	failures = _recent_failures(_client_ip(request))
	if len(failures) >= MAX_LOGIN_FAILURES:
		raise HTTPException(status_code=429, detail="Too many attempts; try again later")
	return failures


def _start_session(response: Response, request: Request, email: str | None) -> None:
	response.set_cookie(
		SESSION_COOKIE,
		_serializer().dumps({"email": email}),
		max_age=get_settings().admin_session_hours * 3600,
		httponly=True,
		samesite="strict",
		secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
		path=ADMIN_PATH,
	)


def _portal_url(request: Request) -> str:
	"""Public address of the portal, used as the target of emailed links."""
	base = get_settings().public_base_url
	if not base:
		scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
		# The Host header has already been checked by TrustedHostMiddleware.
		base = f"{scheme}://{request.headers.get('host', request.url.netloc)}"
	return base.rstrip("/") + ADMIN_PATH


class LoginRequest(BaseModel):
	email: str | None = None
	password: str


class EmailRequest(BaseModel):
	email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SetPasswordRequest(BaseModel):
	access_token: str = Field(min_length=10)
	password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


class ChangePasswordRequest(BaseModel):
	current_password: str
	new_password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


@router.get("")
def admin_page() -> FileResponse:
	return FileResponse(ADMIN_PAGE)


@router.get("/api/auth/status")
def auth_status(database: Session = Depends(get_db)) -> dict:
	"""What the sign-in screen should show."""
	return {
		"setup": setup_login_allowed(database),
		"accounts_ready": supabase_auth.is_configured(),
		"password_min": PASSWORD_MIN,
	}


@router.post("/api/login")
def login(
	body: LoginRequest,
	request: Request,
	response: Response,
	database: Session = Depends(get_db),
) -> dict:
	failures = _guard_attempts(request)

	if not body.email:
		# One-time setup with the shared password, until an account exists.
		admin_password = get_settings().admin_password
		if not setup_login_allowed(database) or not admin_password:
			raise HTTPException(status_code=403, detail="Sign in with your email and password")
		if not hmac.compare_digest(body.password.encode(), admin_password.encode()):
			failures.append(time.monotonic())
			logger.warning("Failed setup login from %s", _client_ip(request))
			raise HTTPException(status_code=401, detail="Wrong password")
		failures.clear()
		_start_session(response, request, None)
		return {"ok": True, "setup": True}

	email = normalize_email(body.email)
	admin = get_admin(database, email)
	try:
		# Same error whether the email is unknown or the password is wrong.
		if admin is None:
			raise SupabaseAuthError("not an admin", code="invalid_credentials", status=400)
		supabase_auth.sign_in(email, body.password)
	except SupabaseNotConfigured as error:
		raise HTTPException(status_code=503, detail=str(error)) from error
	except SupabaseAuthError as error:
		if error.status and error.status >= 500:
			raise HTTPException(status_code=503, detail="The sign-in service is unavailable") from error
		failures.append(time.monotonic())
		logger.warning("Failed admin login for %s from %s", email, _client_ip(request))
		raise HTTPException(status_code=401, detail="Wrong email or password") from error

	failures.clear()
	record_sign_in(database, admin)
	_start_session(response, request, email)
	return {"ok": True, "email": email}


@router.post("/api/password/forgot")
def forgot_password(
	body: EmailRequest, request: Request, database: Session = Depends(get_db)
) -> dict:
	"""Email a set-password link, if the address belongs to an admin."""
	failures = _guard_attempts(request)
	failures.append(time.monotonic())  # limits how many emails one visitor can trigger
	email = normalize_email(body.email)
	if get_admin(database, email) is not None:
		try:
			supabase_auth.send_password_link(email, _portal_url(request))
		except SupabaseAuthError as error:
			logger.warning("Password link for %s failed: %s", email, error)
	# Same answer either way, so this can't be used to discover who is an admin.
	return {"ok": True}


@router.post("/api/password/set")
def set_password(
	body: SetPasswordRequest,
	request: Request,
	response: Response,
	database: Session = Depends(get_db),
) -> dict:
	"""Finish an invitation or password reset: choose a password and sign in."""
	failures = _guard_attempts(request)
	try:
		user = supabase_auth.get_user(body.access_token)
		email = normalize_email(str(user.get("email") or ""))
		admin = get_admin(database, email) if email else None
		if admin is None:
			raise HTTPException(status_code=403, detail="This account does not have admin access")
		supabase_auth.set_password(body.access_token, body.password)
	except SupabaseNotConfigured as error:
		raise HTTPException(status_code=503, detail=str(error)) from error
	except SupabaseAuthError as error:
		failures.append(time.monotonic())
		if error.code == "same_password":
			raise HTTPException(status_code=400, detail="Choose a password you have not used before") from error
		if error.code == "weak_password":
			raise HTTPException(status_code=400, detail=str(error)) from error
		raise HTTPException(
			status_code=400, detail="This link is invalid or has expired. Ask for a new one."
		) from error

	record_sign_in(database, admin)
	_start_session(response, request, email)
	return {"ok": True, "email": email}


@router.post("/api/logout")
def logout(response: Response) -> dict:
	response.delete_cookie(SESSION_COOKIE, path=ADMIN_PATH)
	return {"ok": True}


@router.get("/api/me")
def me(identity: AdminIdentity = Depends(require_admin)) -> dict:
	return {"ok": True, "email": identity.email, "setup": identity.is_setup}


# ---------- admins ----------

def _send_invitation(email: str, request: Request) -> str:
	"""Invite a new person, or send a set-password link if they already have a login."""
	target = _portal_url(request)
	try:
		supabase_auth.invite(email, target)
		return "invited"
	except SupabaseAuthError as error:
		if error.code != "email_exists" and "already" not in str(error).lower():
			raise
	supabase_auth.send_password_link(email, target)
	return "password link sent"


@router.get("/api/admins")
def list_admin_users(
	identity: AdminIdentity = Depends(require_admin), database: Session = Depends(get_db)
) -> dict:
	return {
		"you": identity.email,
		"setup": identity.is_setup,
		"accounts_ready": supabase_auth.is_configured(),
		"admins": [
			{
				"email": admin.email,
				"status": "active" if admin.activated_at else "invited",
				"invited_by": admin.invited_by,
				"created_at": admin.created_at.isoformat() if admin.created_at else None,
				"last_login_at": admin.last_login_at.isoformat() if admin.last_login_at else None,
			}
			for admin in list_admins(database)
		],
	}


@router.post("/api/admins")
def invite_admin(
	body: EmailRequest,
	request: Request,
	identity: AdminIdentity = Depends(require_admin),
	database: Session = Depends(get_db),
) -> dict:
	"""Add an admin: they get an email with a link to choose their password."""
	email = normalize_email(body.email)
	if get_admin(database, email) is not None:
		raise HTTPException(status_code=409, detail="That email is already an admin")
	if not supabase_auth.is_configured():
		raise HTTPException(status_code=503, detail="Supabase Auth is not configured on the server")
	try:
		outcome = _send_invitation(email, request)
	except SupabaseAuthError as error:
		logger.warning("Invitation for %s failed: %s", email, error)
		raise HTTPException(status_code=502, detail=f"The invitation could not be sent: {error}") from error
	add_admin(database, email, invited_by=identity.label)
	logger.info("Admin %s invited by %s", email, identity.label)
	return {"email": email, "outcome": outcome}


@router.post("/api/admins/{email}/resend")
def resend_invitation(
	email: str,
	request: Request,
	identity: AdminIdentity = Depends(require_admin),
	database: Session = Depends(get_db),
) -> dict:
	admin = get_admin(database, email)
	if admin is None:
		raise HTTPException(status_code=404, detail="No such admin")
	try:
		outcome = _send_invitation(admin.email, request)
	except SupabaseAuthError as error:
		raise HTTPException(status_code=502, detail=f"The email could not be sent: {error}") from error
	return {"email": admin.email, "outcome": outcome}


@router.delete("/api/admins/{email}")
def remove_admin_user(
	email: str,
	identity: AdminIdentity = Depends(require_admin),
	database: Session = Depends(get_db),
) -> dict:
	admin = get_admin(database, email)
	if admin is None:
		raise HTTPException(status_code=404, detail="No such admin")
	if admin.email == identity.email:
		raise HTTPException(status_code=400, detail="You can't remove your own account")
	if admin.activated_at is not None and count_active_admins(database) <= 1:
		raise HTTPException(status_code=400, detail="You can't remove the last admin")
	remove_admin(database, admin)
	logger.info("Admin %s removed by %s", admin.email, identity.label)
	return {"removed": admin.email}


@router.post("/api/account/password")
def change_password(
	body: ChangePasswordRequest,
	request: Request,
	identity: AdminIdentity = Depends(require_admin),
) -> dict:
	"""Change the signed-in admin's own password (needs the current one)."""
	if identity.is_setup:
		raise HTTPException(status_code=400, detail="Create your account first")
	failures = _guard_attempts(request)
	try:
		session = supabase_auth.sign_in(identity.email, body.current_password)
	except SupabaseNotConfigured as error:
		raise HTTPException(status_code=503, detail=str(error)) from error
	except SupabaseAuthError as error:
		failures.append(time.monotonic())
		raise HTTPException(status_code=400, detail="Your current password is wrong") from error
	try:
		supabase_auth.set_password(str(session.get("access_token") or ""), body.new_password)
	except SupabaseAuthError as error:
		if error.code == "same_password":
			raise HTTPException(status_code=400, detail="The new password must be different") from error
		raise HTTPException(status_code=400, detail=f"The password could not be changed: {error}") from error
	return {"ok": True}


# ---------- settings ----------

@router.get("/api/settings", dependencies=[Depends(require_admin)])
def read_settings(database: Session = Depends(get_db)) -> dict:
	return {
		"settings": load_runtime_settings(database).model_dump(),
		"defaults": default_runtime_settings().model_dump(),
	}


@router.put("/api/settings", dependencies=[Depends(require_admin)])
def update_settings(body: RuntimeSettings, database: Session = Depends(get_db)) -> dict:
	return {"settings": save_runtime_settings(database, body).model_dump()}


@router.post("/api/settings/reset", dependencies=[Depends(require_admin)])
def reset_settings(database: Session = Depends(get_db)) -> dict:
	return {"settings": reset_runtime_settings(database).model_dump()}


def _mask(value: str | None) -> str | None:
	if not value:
		return None
	return f"{value[:4]}…{value[-4:]}" if len(value) > 12 else "set"


@router.get("/api/environment", dependencies=[Depends(require_admin)])
def read_environment() -> dict:
	settings = get_settings()
	return {
		"secrets": [
			{"name": "OPENAI_API_KEY", "value": _mask(settings.openai_api_key)},
			{"name": "GEMINI_API_KEY", "value": _mask(settings.gemini_api_key)},
			{"name": "META_ACCESS_TOKEN", "value": _mask(settings.meta_access_token)},
			{
				"name": "ADMIN_SESSION_SECRET",
				"value": "set" if settings.admin_session_secret else None,
			},
		],
		"server": [
			{"name": "META_GRAPH_VERSION", "value": settings.meta_graph_version},
			{"name": "META_HTTP_TIMEOUT", "value": settings.meta_http_timeout},
			{"name": "ADMIN_SESSION_HOURS", "value": settings.admin_session_hours},
		],
	}


# ---------- system prompt ----------

class PromptUpdate(BaseModel):
	content: str = Field(min_length=20, max_length=20000)


def _prompt_state(database: Session) -> dict:
	latest = latest_prompt_version(database)
	return {
		"content": latest.content if latest else SYSTEM_PROMPT,
		"is_default": latest is None,
		"updated_at": latest.created_at.isoformat() if latest else None,
		"versions": [
			{
				"id": version.id,
				"created_at": version.created_at.isoformat(),
				"preview": version.content[:120],
			}
			for version in list_prompt_versions(database)
		],
	}


@router.get("/api/prompt", dependencies=[Depends(require_admin)])
def read_prompt(database: Session = Depends(get_db)) -> dict:
	return _prompt_state(database)


@router.put("/api/prompt", dependencies=[Depends(require_admin)])
def update_prompt(body: PromptUpdate, database: Session = Depends(get_db)) -> dict:
	save_system_prompt(database, body.content.strip())
	return _prompt_state(database)


@router.get("/api/prompt/default", dependencies=[Depends(require_admin)])
def read_default_prompt() -> dict:
	return {"content": SYSTEM_PROMPT}


@router.get("/api/prompt/versions/{version_id}", dependencies=[Depends(require_admin)])
def read_prompt_version(version_id: int, database: Session = Depends(get_db)) -> dict:
	version = database.get(SystemPromptVersion, version_id)
	if version is None:
		raise HTTPException(status_code=404, detail="Version not found")
	return {"id": version.id, "content": version.content, "created_at": version.created_at.isoformat()}


@router.post("/api/prompt/reset", dependencies=[Depends(require_admin)])
def reset_prompt(database: Session = Depends(get_db)) -> dict:
	reset_system_prompt(database)
	return _prompt_state(database)


# ---------- pages ----------

class PageUpdate(BaseModel):
	is_active: bool | None = None
	location_id: int | None = None
	instagram_enabled: bool | None = None


@router.get("/api/pages", dependencies=[Depends(require_admin)])
def list_pages(database: Session = Depends(get_db)) -> list[dict]:
	checked = dict(
		database.execute(
			select(ClassifiedComment.page_id, func.count()).group_by(ClassifiedComment.page_id)
		).all()
	)
	flagged = dict(
		database.execute(
			select(ClassifiedComment.page_id, func.count())
			.where(ClassifiedComment.result.is_not(None))
			.group_by(ClassifiedComment.page_id)
		).all()
	)
	return [
		{
			"page_id": page.meta_page_id,
			"name": page.name,
			"location_id": page.location_id,
			"is_active": page.is_active,
			"ig_user_id": page.ig_user_id,
			"ig_username": page.ig_username,
			"instagram_enabled": page.instagram_enabled,
			"checked": checked.get(page.meta_page_id, 0),
			"flagged": flagged.get(page.meta_page_id, 0),
		}
		for page in database.scalars(select(Page).order_by(Page.name))
	]


@router.post("/api/pages/import", dependencies=[Depends(require_admin)])
def import_pages(database: Session = Depends(get_db)) -> dict:
	"""Pull every page the Meta token manages, with its linked Instagram account."""
	user_token = get_settings().meta_access_token
	if not user_token:
		raise HTTPException(status_code=500, detail="META_ACCESS_TOKEN is not set in .env")
	try:
		meta_pages = get_pages_with_instagram(user_token)
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail=str(error)) from error
	instagram_linked = 0
	for page_data in meta_pages:
		meta_page_id = str(page_data["id"])
		existing = database.scalar(select(Page).where(Page.meta_page_id == meta_page_id))
		instagram = instagram_account_of(page_data)
		instagram_linked += instagram is not None
		save_page_credentials(
			database,
			PageCreate(
				id=int(meta_page_id),
				name=page_data["name"],
				meta_page_id=meta_page_id,
				location_id=existing.location_id if existing else None,
				access_token=page_data.get("access_token", user_token),
				ig_user_id=instagram["id"] if instagram else None,
				ig_username=instagram["username"] if instagram else None,
			),
		)
	return {"imported": len(meta_pages), "instagram_linked": instagram_linked}


@router.patch("/api/pages/{page_id}", dependencies=[Depends(require_admin)])
def update_page(page_id: str, body: PageUpdate, database: Session = Depends(get_db)) -> dict:
	page = database.scalar(select(Page).where(Page.meta_page_id == page_id))
	if page is None:
		raise HTTPException(status_code=404, detail="Page not found")
	for field, value in body.model_dump(exclude_unset=True).items():
		setattr(page, field, value)
	database.commit()
	return {
		"page_id": page.meta_page_id,
		"is_active": page.is_active,
		"location_id": page.location_id,
		"instagram_enabled": page.instagram_enabled,
	}


# ---------- results & cache ----------

@router.get("/api/results", dependencies=[Depends(require_admin)])
def list_results(
	page_id: str | None = None,
	type: str | None = None,
	limit: int = 300,
	database: Session = Depends(get_db),
) -> list[dict]:
	query = select(ClassifiedComment).where(ClassifiedComment.result.is_not(None))
	if page_id:
		query = query.where(ClassifiedComment.page_id == page_id)
	query = query.order_by(ClassifiedComment.classified_at.desc()).limit(min(limit, 1000))
	rows = []
	for row in database.scalars(query):
		result = row.result or {}
		if type and result.get("type") != type:
			continue
		rows.append(
			{
				"comment_id": row.comment_id,
				"platform": row.platform or "facebook",
				"page_id": row.page_id,
				"location": row.location,
				"type": result.get("type"),
				"severity": SEVERITY_BY_CATEGORY.get(result.get("type"), "medium"),
				"confidence": result.get("confidence"),
				"reason": result.get("reason"),
				"comment": row.comment_text,
				"author": row.author,
				"comment_link": row.comment_link,
				"time_posted": row.time_posted,
				"classified_at": row.classified_at.isoformat() if row.classified_at else None,
			}
		)
	return rows


@router.delete("/api/results/{comment_id}", dependencies=[Depends(require_admin)])
def recheck_comment(comment_id: str, database: Session = Depends(get_db)) -> dict:
	"""Forget one comment so the next run sends it to the AI again."""
	deleted = database.execute(
		delete(ClassifiedComment).where(ClassifiedComment.comment_id == comment_id)
	).rowcount
	database.commit()
	return {"deleted": deleted}


@router.delete("/api/cache", dependencies=[Depends(require_admin)])
def clear_cache(page_id: str | None = None, database: Session = Depends(get_db)) -> dict:
	"""Forget checked comments (one page or all) so they are re-checked."""
	statement = delete(ClassifiedComment)
	if page_id:
		statement = statement.where(ClassifiedComment.page_id == page_id)
	deleted = database.execute(statement).rowcount
	database.commit()
	return {"deleted": deleted}
