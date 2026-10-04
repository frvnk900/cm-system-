import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.db.db import get_db
from app.api.db.save_page_creds_ import save_page_credentials
from app.api.schema.pages_model import Page, PageCreate, PageResponse
from app.services.get_pages_ import MetaAPIError, get_page, get_pages
from app.services.instagram_ import get_pages_with_instagram, instagram_account_of


load_dotenv()

router = APIRouter(prefix="/pages", tags=["pages"])


def _get_access_token() -> str:
	access_token = os.getenv("META_ACCESS_TOKEN")
	if not access_token:
		raise HTTPException(
			status_code=500,
			detail="META_ACCESS_TOKEN is not configured",
		)
	return access_token


def _without_token(page: dict) -> dict:
	"""Page access tokens stay on the server; they are never returned."""
	return {key: value for key, value in page.items() if key != "access_token"}


@router.get("", response_model=list[dict])
def list_pages() -> list[dict]:
	"""Return every page managed by the configured Meta user token (no tokens)."""
	try:
		return [_without_token(page) for page in get_pages(_get_access_token())]
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error


@router.post("/save", response_model=list[PageResponse])
def save_pages_to_db(
	database: Session = Depends(get_db),
) -> list[PageResponse]:
	"""Import all managed Meta pages using the configured server token."""
	user_access_token = _get_access_token()

	try:
		stored_pages = []
		for page_data in get_pages_with_instagram(user_access_token):
			meta_page_id = str(page_data["id"])
			existing = database.scalar(
				select(Page).where(Page.meta_page_id == meta_page_id)
			)
			instagram = instagram_account_of(page_data)
			stored_pages.append(
				save_page_credentials(
					database,
					PageCreate(
						id=int(meta_page_id),
						name=page_data["name"],
						meta_page_id=meta_page_id,
						location_id=existing.location_id if existing else None,
						access_token=page_data.get(
							"access_token", user_access_token
						),
						ig_user_id=instagram["id"] if instagram else None,
						ig_username=instagram["username"] if instagram else None,
					),
				),
			)

		return stored_pages
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error


@router.get("/active", response_model=list[dict])
def list_active_pages(database: Session = Depends(get_db)) -> list[dict]:
	"""Return saved pages enabled in the admin portal (no tokens)."""
	pages = database.scalars(
		select(Page).where(Page.is_active.is_(True)).order_by(Page.name)
	)
	return [
		{"name": page.name, "page_id": page.meta_page_id, "location_id": page.location_id}
		for page in pages
	]


@router.get("/{page_id}", response_model=dict)
def read_page(
	page_id: str,
) -> dict:
	"""Return one Meta page by its page id (no token)."""
	try:
		return _without_token(get_page(page_id, _get_access_token()))
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error
