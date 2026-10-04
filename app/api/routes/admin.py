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
from app.services.get_pages_ import MetaAPIError
from app.services.instagram_ import get_pages_with_instagram, instagram_account_of
from app.services.prompt.system_prompt import SYSTEM_PROMPT


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], include_in_schema=False)

ADMIN_PAGE = Path(__file__).resolve().parents[2] / "admin" / "index.html"
SESSION_COOKIE = "admin_session"
# Used only when ADMIN_SESSION_SECRET is unset; sessions then end on restart.
_FALLBACK_SECRET = secrets.token_urlsafe(32)
MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60
_login_failures: dict[str, deque[float]] = defaultdict(deque)


# ---------- auth ----------

def _serializer() -> URLSafeTimedSerializer:
	secret = get_settings().admin_session_secret or _FALLBACK_SECRET
	return URLSafeTimedSerializer(secret, salt="admin-session")


def require_admin(request: Request) -> None:
	token = request.cookies.get(SESSION_COOKIE)
	if not token:
		raise HTTPException(status_code=401, detail="Not logged in")
	try:
		_serializer().loads(token, max_age=get_settings().admin_session_hours * 3600)
	except BadSignature as error:
		raise HTTPException(status_code=401, detail="Session expired") from error


def _recent_failures(client_ip: str) -> deque[float]:
	failures = _login_failures[client_ip]
	cutoff = time.monotonic() - LOGIN_LOCKOUT_SECONDS
	while failures and failures[0] < cutoff:
		failures.popleft()
	return failures


class LoginRequest(BaseModel):
	password: str


@router.get("")
def admin_page() -> FileResponse:
	return FileResponse(ADMIN_PAGE)


@router.post("/api/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict:
	admin_password = get_settings().admin_password
	if not admin_password:
		raise HTTPException(status_code=503, detail="ADMIN_PASSWORD is not set in .env")

	client_ip = request.client.host if request.client else "unknown"
	failures = _recent_failures(client_ip)
	if len(failures) >= MAX_LOGIN_FAILURES:
		raise HTTPException(status_code=429, detail="Too many attempts; try again later")

	if not hmac.compare_digest(body.password.encode(), admin_password.encode()):
		failures.append(time.monotonic())
		logger.warning("Failed admin login from %s", client_ip)
		raise HTTPException(status_code=401, detail="Wrong password")

	failures.clear()
	response.set_cookie(
		SESSION_COOKIE,
		_serializer().dumps("admin"),
		max_age=get_settings().admin_session_hours * 3600,
		httponly=True,
		samesite="strict",
		secure=request.url.scheme == "https",
		path="/admin",
	)
	return {"ok": True}


@router.post("/api/logout")
def logout(response: Response) -> dict:
	response.delete_cookie(SESSION_COOKIE, path="/admin")
	return {"ok": True}


@router.get("/api/me", dependencies=[Depends(require_admin)])
def me() -> dict:
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
