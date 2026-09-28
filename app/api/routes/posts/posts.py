from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.db.db import get_db
from app.api.db.page_access_ import get_page_access_token
from app.services.get_pages_ import MetaAPIError
from app.services.get_posts_ import get_page_posts, get_post


router = APIRouter(prefix="/posts", tags=["posts"])


@router.get("/page/{page_id}", response_model=list[dict])
def list_page_posts(
	page_id: str, database: Session = Depends(get_db)
) -> list[dict]:
	"""Return every post published on a page."""
	page_access_token = get_page_access_token(database, page_id)
	if not page_access_token:
		raise HTTPException(status_code=404, detail="Page is not saved in the database")
	try:
		return get_page_posts(page_id, page_access_token)
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error


@router.get("/{post_id}", response_model=dict)
def read_post(
	post_id: str,
	database: Session = Depends(get_db),
	page_id: str = Query(...),
) -> dict:
	"""Return one post by its Meta post id."""
	page_access_token = get_page_access_token(database, page_id)
	if not page_access_token:
		raise HTTPException(status_code=404, detail="Page is not saved in the database")
	try:
		return get_post(post_id, page_access_token)
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error
