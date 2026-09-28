import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.db.db import get_db
from app.api.db.page_access_ import (
	get_all_page_access_tokens,
	get_page_access_token,
)
from app.services.get_comments_ import (
	delete_comment,
	get_all_comments_from_a_post,
	get_all_comments_from_page,
	get_comment,
)
from app.services.get_pages_ import MetaAPIError, MetaAPIRequestError


router = APIRouter(prefix="/comments", tags=["comments"])
logger = logging.getLogger(__name__)


def _get_tokens(database: Session, page_id: str | None) -> list[str]:
	if page_id:
		token = get_page_access_token(database, page_id)
		if token:
			return [token]
		raise HTTPException(status_code=404, detail="Page is not saved in the database")

	tokens = get_all_page_access_tokens(database)
	if not tokens:
		raise HTTPException(status_code=404, detail="No saved page tokens found")
	return tokens


def _get_comment_from_saved_pages(comment_id: str, database: Session) -> dict:
	last_error: MetaAPIError | None = None
	for token in _get_tokens(database, None):
		try:
			return get_comment(comment_id, token)
		except MetaAPIRequestError:
			raise
		except MetaAPIError as error:
			last_error = error
	if last_error:
		raise last_error
	raise HTTPException(status_code=404, detail="Comment was not found")


@router.get("/page/{page_id}", response_model=list[dict])
def list_page_comments(
	page_id: str, database: Session = Depends(get_db)
) -> list[dict]:
	"""Return all comments from every post on a page."""
	page_access_token = get_page_access_token(database, page_id)
	if not page_access_token:
		raise HTTPException(status_code=404, detail="Page is not saved in the database")
	try:
		return get_all_comments_from_page(page_id, page_access_token)
	except MetaAPIRequestError as error:
		logger.warning("Meta page comment request failed: %s", error)
		raise HTTPException(status_code=504, detail="Meta comment request timed out or failed") from error
	except MetaAPIError as error:
		logger.warning("Meta page comment retrieval failed: %s", error)
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error


@router.get("/post/{post_id}", response_model=list[dict])
def list_post_comments(
	post_id: str,
	database: Session = Depends(get_db),
	page_id: str | None = Query(None),
) -> list[dict]:
	"""Return every comment on a post."""
	try:
		for page_access_token in _get_tokens(database, page_id):
			try:
				return get_all_comments_from_a_post(post_id, page_access_token)
			except MetaAPIError:
				continue
		raise HTTPException(status_code=404, detail="Post was not found")
	except MetaAPIRequestError as error:
		logger.warning("Meta post comment request failed: %s", error)
		raise HTTPException(status_code=504, detail="Meta comment request timed out or failed") from error
	except MetaAPIError as error:
		logger.warning("Meta post comment retrieval failed: %s", error)
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error


@router.get("/{comment_id}", response_model=dict)
def read_comment(
	comment_id: str,
	database: Session = Depends(get_db),
	page_id: str = Query(...),
) -> dict:
	"""Return one comment by its Meta comment id."""
	try:
		return get_comment(comment_id, _get_tokens(database, page_id)[0])
	except MetaAPIRequestError as error:
		logger.warning("Meta single comment request failed: %s", error)
		raise HTTPException(status_code=504, detail="Meta comment request timed out or failed") from error
	except MetaAPIError as error:
		logger.warning("Meta single comment retrieval failed: %s", error)
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error


@router.delete("/{comment_id}", response_model=dict)
def remove_comment(
	comment_id: str,
	database: Session = Depends(get_db),
	page_id: str | None = Query(None),
) -> dict:
	"""Delete one comment by its Meta comment id."""
	try:
		if page_id:
			return delete_comment(comment_id, _get_tokens(database, page_id)[0])
		for page_access_token in _get_tokens(database, None):
			try:
				return delete_comment(comment_id, page_access_token)
			except MetaAPIError:
				continue
		raise HTTPException(status_code=404, detail="Comment was not found")
	except MetaAPIError as error:
		raise HTTPException(status_code=502, detail="Meta content could not be retrieved") from error
