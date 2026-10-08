from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.api.db.db import Base


class PageFetchCache(Base):
    """What was last fetched from Meta for a page, reused for a few minutes.

    Moderation and sentiment run back to back for each location; the second
    one reads this instead of asking Meta for the same posts and comments.
    """

    __tablename__ = "page_fetch_cache"

    page_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    # The settings the fetch was made with; a different signature is a miss.
    signature: Mapped[str] = mapped_column(String(255), nullable=False)
    # {"request": <ClassificationRequest as JSON> or None when nothing was found}
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ServiceState(Base):
    """Small key/value store for service-wide state (e.g. the Meta cooldown)."""

    __tablename__ = "service_state"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
