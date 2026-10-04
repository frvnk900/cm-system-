from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.api.db.db import SessionLocal, create_tables
from app.api.routes.admin import router as admin_router
from app.api.routes.ai import router as ai_router
from app.api.routes.comments.comments import router as comments_router
from app.api.routes.pages.pages import router as pages_router
from app.api.routes.posts.posts import router as posts_router
from app.api.schema.pages_model import Page
from app.core.security import require_api_key
from app.core.settings import get_settings


load_dotenv()
logger = logging.getLogger(__name__)


def _environment_list(name: str, defaults: str) -> list[str]:
	values = os.getenv(name, defaults)
	return [value.strip() for value in values.split(",") if value.strip()]


cors_origins = _environment_list(
	"CORS_ORIGINS",
	"http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173, *",
)
trusted_hosts = _environment_list(
	"TRUSTED_HOSTS",
	"localhost,127.0.0.1,testserver,*.vercel.app",
)



@asynccontextmanager
async def lifespan(_: FastAPI):
	create_tables()
	yield


app = FastAPI(lifespan=lifespan)
# The admin portal has its own password login.
app.include_router(admin_router)
# Everything else needs the API key (once API_KEY is set).
api_protection = [Depends(require_api_key)]
app.include_router(ai_router, dependencies=api_protection)
app.include_router(comments_router, dependencies=api_protection)
app.include_router(pages_router, dependencies=api_protection)
app.include_router(posts_router, dependencies=api_protection)
app.add_middleware(
	TrustedHostMiddleware,
	allowed_hosts=trusted_hosts,
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=cors_origins,
	allow_credentials=True,
	allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
	allow_headers=["Content-Type"],
)


LANDING_PAGE = Path(__file__).resolve().parent / "landing" / "index.html"
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Brand images change rarely; let browsers keep them for a day.
ASSET_HEADERS = {"Cache-Control": "public, max-age=86400"}


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
def favicon() -> FileResponse:
	return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png", headers=ASSET_HEADERS)


@app.get("/logo.png", include_in_schema=False)
def logo() -> FileResponse:
	return FileResponse(STATIC_DIR / "logo.png", media_type="image/png", headers=ASSET_HEADERS)


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
	"""Public landing page; its Sign in button points at the admin portal."""
	html = LANDING_PAGE.read_text(encoding="utf-8")
	return HTMLResponse(html.replace("__ADMIN_PATH__", get_settings().admin_path))


@app.get("/health", tags=["health"])
def health() -> dict:
	"""Machine-readable status: is the API up, and what is it monitoring."""
	try:
		with SessionLocal() as database:
			active = select(func.count()).select_from(Page).where(Page.is_active.is_(True))
			locations = database.scalar(active)
			instagram_accounts = database.scalar(
				active.where(Page.ig_user_id.is_not(None), Page.instagram_enabled.is_(True))
			)
	except SQLAlchemyError:
		logger.warning("Health check could not reach the database", exc_info=True)
		return {"status": "degraded"}
	return {
		"status": "ok",
		"locations": locations,
		"instagram_accounts": instagram_accounts,
	}


@app.get("/robots.txt", include_in_schema=False)
def robots() -> PlainTextResponse:
	"""Keep search engines away from the whole service."""
	return PlainTextResponse("User-agent: *\nDisallow: /\n")


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(
		"app.main:app",
		host=os.getenv("HOST", "127.0.0.1"),
		port=int(os.getenv("PORT", "8000")),
		reload=True,
		reload_includes=[".env"],
	)
