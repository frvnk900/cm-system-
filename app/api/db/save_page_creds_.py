from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.schema.pages_model import Page, PageCreate


def save_page_credentials(database: Session, page_data: PageCreate) -> Page:
	"""Save one connected Meta page, replacing an existing duplicate."""
	existing_page = database.get(Page, page_data.id)
	existing_page = database.scalar(
		select(Page).where(Page.meta_page_id == page_data.meta_page_id)
	) or existing_page

	if existing_page is None:
		page = Page(**page_data.model_dump())
		database.add(page)
	else:
		# Keep is_active as set in the admin portal when re-importing.
		for field, value in page_data.model_dump(exclude={"is_active"}).items():
			setattr(existing_page, field, value)
		page = existing_page

	try:
		database.commit()
	except IntegrityError:
		database.rollback()
		raise

	database.refresh(page)
	return page
