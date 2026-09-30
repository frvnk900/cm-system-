from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.api.db.db import Base


class CommentSentimentRecord(Base):
    """Cached sentiment for a comment, so it is only sent to the AI once."""

    __tablename__ = "comment_sentiments"

    comment_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    # Hash of the comment text; an edited comment gets classified again.
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
