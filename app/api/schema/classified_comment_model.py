from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.api.db.db import Base


class ClassifiedComment(Base):
    """A comment the AI has already checked, so it is not sent again."""

    __tablename__ = "classified_comments"

    comment_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    # Hash of the comment text; an edited comment gets checked again.
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # The classification result, or NULL when the comment was clean.
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    platform: Mapped[str | None] = mapped_column(String(20), nullable=True)
    page_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    comment_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    comment_link: Mapped[str | None] = mapped_column(Text, nullable=True)
    time_posted: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
