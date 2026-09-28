from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schema.pages_model import Page


def get_page_access_token(database: Session, page_id: str) -> str | None:
	"""Return the stored Meta page token for a page id."""
	page = database.scalar(select(Page).where(Page.meta_page_id == page_id))
	return page.access_token if page is not None else None


def get_all_page_access_tokens(database: Session) -> list[str]:
	"""Return all stored page tokens for Meta object lookup."""
	return list(database.scalars(select(Page.access_token)).all())