import logging
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.db.app_settings_ import load_runtime_settings
from app.api.db.classified_comments_ import body_hash, load_classified, save_classified
from app.api.db.db import get_db
from app.api.db.system_prompt_ import load_system_prompt
from app.api.schema.pages_model import Page
from app.core.settings import RuntimeSettings
from app.api.schema.ai_model import (
	ClassificationRequest,
	ClassificationResult,
	CommentContext,
	PostContext,
)
from app.services.ai_classification import (
	AIConfigurationError,
	AIOutputError,
	AIProviderError,
	classify_comments,
)
from app.services.get_comments_ import get_all_comments_from_a_post
from app.services.get_pages_ import MetaAPIError, MetaAPIRequestError, get_page
from app.services.get_posts_ import get_page_posts


load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai"])
SEVERITY_BY_CATEGORY = {
	"threat": "critical",
	"doxxing": "critical",
	"hate_speech": "high",
	"harassment": "high",
	"defamation": "high",
	"profanity": "medium",
	"misinformation": "medium",
	"spam": "low",
	"negative_sentiment": "low",
}


def _get_meta_access_token() -> str:
	access_token = os.getenv("META_ACCESS_TOKEN")
	if not access_token:
		raise HTTPException(
			status_code=500,
			detail="META_ACCESS_TOKEN is not configured",
		)
	return access_token


def _build_page_request(
	page_id: str, page_name: str, access_token: str, settings: RuntimeSettings
) -> ClassificationRequest | None:
	"""Collect recent posts and comments; None when there is nothing to classify."""
	max_items = settings.max_comments_per_run
	now = datetime.now(timezone.utc)
	window_start = now - timedelta(days=settings.lookback_days)
	posts_from_meta = get_page_posts(
		page_id,
		access_token,
		max_items=max_items,
		since=int(window_start.timestamp()),
		until=int(now.timestamp()),
	)
	logger.info("Meta posts fetched: posts=%d", len(posts_from_meta))
	posts: list[PostContext] = []
	comments: list[CommentContext] = []

	for post_number, post in enumerate(posts_from_meta, start=1):
		if len(comments) >= max_items:
			break
		created_time = post.get("created_time")
		if not created_time:
			continue
		try:
			created_at = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
		except (AttributeError, ValueError):
			logger.warning("Skipping post with invalid created_time")
			continue
		created_at_utc = created_at.astimezone(timezone.utc)
		if not window_start <= created_at_utc <= now:
			continue
		post_id = str(post["id"]) if post.get("id") is not None else None
		posts.append(
			PostContext(
				post_id=post_id,
				page_id=page_id,
				body=str(post.get("message") or ""),
			)
		)
		if post_id is None:
			continue

		remaining_comments = max_items - len(comments)
		for comment in get_all_comments_from_a_post(
			post_id, access_token, max_items=remaining_comments
		):
			comment_author = comment.get("from")
			comments.append(
				CommentContext(
					comment_id=(
						str(comment["id"])
						if comment.get("id") is not None
						else None
					),
					post_id=post_id,
					page_id=page_id,
					author=(
						str(comment_author.get("name"))
						if isinstance(comment_author, dict)
						and comment_author.get("name") is not None
						else None
					),
					body=str(comment.get("message") or ""),
					comment_link=str(comment.get("permalink_url") or ""),
					time_posted=str(comment.get("created_time") or ""),
				)
			)
		if post_number == 1 or post_number % 10 == 0:
			logger.info(
				"Meta comment collection progress: posts_processed=%d comments=%d",
				post_number,
				len(comments),
			)

	logger.info(
		"Meta comment collection completed: posts=%d comments=%d",
		len(posts),
		len(comments),
	)
	# ClassificationRequest needs at least one post and one comment.
	if not posts or not comments:
		return None
	return ClassificationRequest(
		location=page_name,
		posts=posts,
		comments=comments,
	)


def _format_results(
	results: list[ClassificationResult], request: ClassificationRequest
) -> list[ClassificationResult]:
	comments_by_key = {
		(comment.comment_id, comment.post_id): comment for comment in request.comments
	}
	logged_at = datetime.now(timezone.utc).isoformat()
	formatted: list[ClassificationResult] = []
	for result in results:
		comment = comments_by_key.get((result.comment_id, result.post_id))
		formatted.append(
			result.model_copy(
				update={
					"logged_at": logged_at,
					"platform": "facebook",
					"page_name": request.location,
					"comment": comment.body if comment else "",
					"comment_link": comment.comment_link if comment else "",
					"time_posted": comment.time_posted if comment else "",
					"author": comment.author if comment else "",
					"severity": SEVERITY_BY_CATEGORY.get(result.type, "medium"),
					"category": result.type,
					"status": "pending",
					"parent_post_id": result.post_id or "",
				}
			)
		)
	return formatted


def _classify_new_comments(
	request: ClassificationRequest, database: Session, settings: RuntimeSettings
) -> list[ClassificationResult]:
	"""Reuse stored results and send only new or edited comments to the AI."""
	cached = load_classified(database, request.comments)
	results: list[ClassificationResult] = []
	new_comments: list[CommentContext] = []
	for comment in request.comments:
		row = cached.get(comment.comment_id) if comment.comment_id else None
		if row is not None and row.body_hash == body_hash(comment.body):
			if row.result is not None:
				results.append(ClassificationResult(**row.result))
		else:
			new_comments.append(comment)
	logger.info(
		"Classification cache: cached=%d new=%d",
		len(request.comments) - len(new_comments),
		len(new_comments),
	)

	if new_comments:
		results.extend(
			classify_comments(
				request.model_copy(update={"comments": new_comments}),
				settings=settings,
				system_prompt=load_system_prompt(database),
				on_batch=lambda comments, batch_results: save_classified(
					database, comments, batch_results, request.location
				),
			)
		)
	return results


@router.post("/classify/{page_id}", response_model=list[ClassificationResult])
def classify_page(
	page_id: str, database: Session = Depends(get_db)
) -> list[ClassificationResult]:
	"""Fetch a page's content and classify comments not checked before."""
	access_token = _get_meta_access_token()
	stored_page = database.scalar(select(Page).where(Page.meta_page_id == page_id))
	if stored_page is not None and not stored_page.is_active:
		raise HTTPException(
			status_code=409,
			detail="This page is disabled in the admin portal",
		)
	settings = load_runtime_settings(database)
	try:
		page = get_page(page_id, access_token)
		page_name = str(page.get("name") or page_id)
		page_access_token = str(page.get("access_token") or access_token)
		request = _build_page_request(page_id, page_name, page_access_token, settings)
		if request is None:
			return []
		return _format_results(
			_classify_new_comments(request, database, settings), request
		)
	except MetaAPIRequestError as error:
		logger.warning("Meta content request failed: %s", error)
		raise HTTPException(
			status_code=504,
			detail="Meta content request timed out or failed",
		) from error
	except MetaAPIError as error:
		logger.warning("Meta content retrieval failed: %s", error)
		raise HTTPException(
			status_code=502,
			detail="Meta content could not be retrieved",
		) from error
	except AIConfigurationError as error:
		raise HTTPException(status_code=500, detail=str(error)) from error
	except AIProviderError as error:
		raise HTTPException(
			status_code=503,
			detail="The classification provider is temporarily unavailable",
		) from error
	except AIOutputError as error:
		raise HTTPException(status_code=502, detail=str(error)) from error
	except Exception as error:
		logger.exception("Unexpected AI classification failure")
		raise HTTPException(
			status_code=500,
			detail="Classification failed unexpectedly",
		) from error
