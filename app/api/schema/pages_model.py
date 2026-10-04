from datetime import datetime

from sqlalchemy import BIGINT, Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import BaseModel, ConfigDict

from app.api.db.db import Base


class Page(Base):
	"""Database model for a connected Meta page."""

	__tablename__ = "pages"

	id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
	name: Mapped[str] = mapped_column(String(255), nullable=False)
	meta_page_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	location_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True, index=True)
	access_token: Mapped[str] = mapped_column(String, nullable=False)
	is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
	# Instagram business account linked to this Page (found on import).
	ig_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
	ig_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
	instagram_enabled: Mapped[bool] = mapped_column(
		Boolean, default=True, server_default="true", nullable=False
	)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True),
		server_default=func.now(),
		nullable=False,
	)


class PageCreate(BaseModel):
	"""Request model for storing a connected Meta page."""

	id: int
	name: str
	meta_page_id: str
	location_id: int | None = None
	access_token: str
	is_active: bool = True
	ig_user_id: str | None = None
	ig_username: str | None = None


class PageImport(BaseModel):
	"""Request model for importing all managed pages from Meta Graph API."""

	access_token: str
	location_id: int | None = None


class PageResponse(BaseModel):
	"""API response model for a stored Meta page (the access token is never returned)."""

	model_config = ConfigDict(from_attributes=True)

	id: int
	name: str
	meta_page_id: str
	location_id: int | None
	is_active: bool
	created_at: datetime
