import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.db.app_settings_ import load_runtime_settings
from app.api.db.comment_sentiments_ import load_sentiments, save_sentiments
from app.api.db.db import get_db
from app.api.db.page_access_ import (
	get_all_page_access_tokens,
	get_page_access_token,
)
from app.api.routes.ai import _build_page_request, _get_meta_access_token
from app.api.schema.ai_model import CommentSentiment
from app.api.schema.pages_model import Page
from app.services.ai_classification import (
	AIConfigurationError,
	AIOutputError,
	AIProviderError,
)
from app.services.sentiment import classify_sentiment
from app.services.get_comments_ import (
	delete_comment,
	get_all_comments_from_a_post,
	get_all_comments_from_page,
	get_comment,
)
from app.services.get_pages_ import MetaAPIError, MetaAPIRequestError, get_page


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


@router.get("/page/{page_id}/sentiment", response_model=list[CommentSentiment])
def list_page_comment_sentiment(
	page_id: str,
	sentiment: Literal["positive", "neutral", "negative", "unknown"] | None = Query(
		None, description="Only return comments with this sentiment"
	),
	database: Session = Depends(get_db),
) -> list[CommentSentiment]:
	"""Return recent comments (positive and negative), each labelled by the AI.

	Uses the same comment window as moderation (admin settings). Sentiments are
	cached, so a comment is only sent to the AI once. Any returned comment can
	be deleted with DELETE /comments/{comment_id}?page_id={page_id}.
	"""
	access_token = _get_meta_access_token()
	stored_page = database.scalar(select(Page).where(Page.meta_page_id == page_id))
	if stored_page is not None and not stored_page.is_active:
		raise HTTPException(status_code=409, detail="This page is disabled in the admin portal")
	settings = load_runtime_settings(database)
	try:
		page = get_page(page_id, access_token)
		page_name = str(page.get("name") or page_id)
		page_token = str(page.get("access_token") or access_token)
		request = _build_page_request(page_id, page_name, page_token, settings)
		if request is None:
			return []

		cached = load_sentiments(database, request.comments)
		to_classify = [
			comment
			for comment in request.comments
			if comment.comment_id
			and comment.comment_id not in cached
			and comment.body.strip()
		]
		logger.info(
			"Sentiment cache: cached=%d new=%d", len(cached), len(to_classify)
		)
		fresh = {}
		if to_classify:
			fresh = classify_sentiment(
				request.model_copy(update={"comments": to_classify}),
				settings,
				on_batch=lambda comments, results: save_sentiments(
					database, comments, results
				),
			)
	except MetaAPIRequestError as error:
		logger.warning("Meta sentiment request failed: %s", error)
		raise HTTPException(status_code=504, detail="Meta content request timed out or failed") from error
	except MetaAPIError as error:
		logger.warning("Meta sentiment retrieval failed: %s", error)
		raise HTTPException(status_code=502, detail=f"Meta content could not be retrieved: {error}") from error
	except AIConfigurationError as error:
		raise HTTPException(status_code=500, detail=str(error)) from error
	except AIProviderError as error:
		raise HTTPException(status_code=503, detail="The AI provider is temporarily unavailable") from error
	except AIOutputError as error:
		raise HTTPException(status_code=502, detail=str(error)) from error

	labelled: list[CommentSentiment] = []
	for comment in request.comments:  # newest first
		if not comment.comment_id:
			continue
		if comment.comment_id in cached:
			row = cached[comment.comment_id]
			label, confidence, reason = row.sentiment, row.confidence, row.reason or ""
		elif comment.comment_id in fresh:
			result = fresh[comment.comment_id]
			label, confidence, reason = result.sentiment, result.confidence, result.reason
		elif not comment.body.strip():
			label, confidence, reason = "neutral", 0.0, "No text (image or sticker only)"
		else:
			label, confidence, reason = "unknown", 0.0, "Not classified this run; try again"
		if sentiment is not None and label != sentiment:
			continue
		labelled.append(
			CommentSentiment(
				comment_id=comment.comment_id,
				post_id=comment.post_id,
				page_id=comment.page_id,
				location=request.location,
				sentiment=label,
				confidence=confidence,
				reason=reason,
				comment=comment.body,
				author=comment.author or "",
				comment_link=comment.comment_link or "",
				time_posted=comment.time_posted or "",
			)
		)
	return labelled


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


def _fresh_page_token(page_id: str) -> str | None:
	"""Page token derived from META_ACCESS_TOKEN now, so it can't be stale."""
	user_token = os.getenv("META_ACCESS_TOKEN")
	if not user_token:
		return None
	try:
		return get_page(page_id, user_token).get("access_token")
	except MetaAPIError as error:
		logger.warning("Could not get a fresh page token for %s: %s", page_id, error)
		return None


@router.delete("/{comment_id}", response_model=dict)
def remove_comment(
	comment_id: str,
	database: Session = Depends(get_db),
	page_id: str | None = Query(None),
) -> dict:
	"""Delete one comment by its Meta comment id."""
	try:
		if page_id:
			token = _fresh_page_token(page_id) or _get_tokens(database, page_id)[0]
			return delete_comment(comment_id, token)
		for page_access_token in _get_tokens(database, None):
			try:
				return delete_comment(comment_id, page_access_token)
			except MetaAPIError:
				continue
		raise HTTPException(status_code=404, detail="Comment was not found")
	except MetaAPIError as error:
		logger.warning("Meta comment delete failed: %s", error)
		# Facebook's reason (expired token, missing permission, already deleted…).
		raise HTTPException(
			status_code=502, detail=f"Facebook refused the delete: {error}"
		) from error
