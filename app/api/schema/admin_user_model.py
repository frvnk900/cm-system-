from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.api.db.db import Base


class AdminUser(Base):
    """Who may use the admin portal. Passwords live in Supabase Auth, not here."""

    __tablename__ = "admin_users"

    # Always stored lower-case.
    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    invited_by: Mapped[str | None] = mapped_column(String(320), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Set once the person has chosen a password; until then they are "invited".
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
