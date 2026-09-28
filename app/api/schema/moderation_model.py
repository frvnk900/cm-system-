from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BIGINT, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.api.db.db import Base


class Moderation(Base):
    """Database record for a moderated Meta comment."""

    __tablename__ = "moderation"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("pages.id"),
        nullable=False,
    )
    post_id: Mapped[str] = mapped_column(String(255), nullable=False)
    comment_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    comment_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    author_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    action: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    initiator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    detected_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    actioned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ModerationResponse(BaseModel):
    """API response schema for a moderation record."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    page_id: int
    post_id: str
    comment_id: str
    comment_text: str | None
    author_name: str | None
    author_id: str | None
    action: str | None
    reason: str | None
    category: str | None
    confidence: float | None
    initiator: str | None
    detected_by: str | None
    status: str
    created_at: datetime
    actioned_at: datetime | None
