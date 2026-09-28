from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.db.db import create_tables
from app.api.routes.admin import router as admin_router
from app.api.routes.ai import router as ai_router
from app.api.routes.comments.comments import router as comments_router
from app.api.routes.pages.pages import router as pages_router
from app.api.routes.posts.posts import router as posts_router


load_dotenv()


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
app.include_router(admin_router)
app.include_router(ai_router)
app.include_router(comments_router)
app.include_router(pages_router)
app.include_router(posts_router)
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


@app.get("/", tags=["health"])
def root() -> dict:
	"""Entry page: confirms the API is up and points to the main pages."""
	return {"status": "ok", "docs": "/docs", "admin": "/admin"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(
		"app.main:app",
		host=os.getenv("HOST", "127.0.0.1"),
		port=int(os.getenv("PORT", "8000")),
		reload=True,
		reload_includes=[".env"],
	)
