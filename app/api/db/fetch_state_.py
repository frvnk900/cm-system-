from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.api.schema.fetch_state_model import PageFetchCache, ServiceState


COOLDOWN_KEY = "meta_cooldown_until"


def _aware(value: datetime) -> datetime:
	return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def load_page_cache(
	database: Session, page_id: str, signature: str, max_age: timedelta
) -> dict[str, Any] | None:
	"""The cached fetch for a page if it is fresh and made with the same settings."""
	row = database.get(PageFetchCache, page_id)
	if row is None or row.signature != signature:
		return None
	if datetime.now(timezone.utc) - _aware(row.fetched_at) > max_age:
		return None
	return row.payload


def save_page_cache(
	database: Session, page_id: str, signature: str, request: dict[str, Any] | None
) -> None:
	database.merge(
		PageFetchCache(
			page_id=page_id,
			signature=signature,
			payload={"request": request},
			fetched_at=datetime.now(timezone.utc),
		)
	)
	database.commit()


def meta_cooldown_until(database: Session) -> datetime | None:
	"""When Meta calls may resume, or None if they are allowed now."""
	row = database.get(ServiceState, COOLDOWN_KEY)
	if row is None:
		return None
	try:
		until = _aware(datetime.fromisoformat(row.value))
	except ValueError:
		return None
	return until if until > datetime.now(timezone.utc) else None


def start_meta_cooldown(database: Session, minutes: int) -> datetime:
	"""Pause Meta calls for a while after a rate-limit error."""
	until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
	database.merge(ServiceState(key=COOLDOWN_KEY, value=until.isoformat()))
	database.commit()
	return until
