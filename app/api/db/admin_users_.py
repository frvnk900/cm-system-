from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schema.admin_user_model import AdminUser


def normalize_email(email: str) -> str:
	return email.strip().lower()


def get_admin(database: Session, email: str) -> AdminUser | None:
	return database.get(AdminUser, normalize_email(email))


def list_admins(database: Session) -> list[AdminUser]:
	return list(database.scalars(select(AdminUser).order_by(AdminUser.created_at)))


def count_active_admins(database: Session) -> int:
	"""Admins who have set a password and can therefore sign in."""
	return database.scalar(
		select(func.count()).select_from(AdminUser).where(AdminUser.activated_at.is_not(None))
	) or 0


def add_admin(database: Session, email: str, invited_by: str | None) -> AdminUser:
	admin = AdminUser(email=normalize_email(email), invited_by=invited_by)
	database.add(admin)
	database.commit()
	return admin


def remove_admin(database: Session, admin: AdminUser) -> None:
	database.delete(admin)
	database.commit()


def record_sign_in(database: Session, admin: AdminUser) -> None:
	"""Mark the admin active (first time) and note the sign-in time."""
	now = datetime.now(timezone.utc)
	if admin.activated_at is None:
		admin.activated_at = now
	admin.last_login_at = now
	database.commit()
