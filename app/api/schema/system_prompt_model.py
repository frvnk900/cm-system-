from datetime import datetime

from sqlalchemy import BIGINT, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.api.db.db import Base


class SystemPromptVersion(Base):
    """One saved version of the AI system prompt; the newest one is used."""

    __tablename__ = "system_prompt_versions"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
