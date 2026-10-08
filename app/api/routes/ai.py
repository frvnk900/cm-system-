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
from app.api.db.fetch_state_ import (
	load_page_cache,
	meta_cooldown_until,
	save_page_cache,
	start_meta_cooldown,
)
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
from app.services.get_comments_ import get_all_comments_from_a_post, parse_meta_time
from app.services.get_pages_ import (
	MetaAPIError,
	MetaAPIRequestError,
	MetaAuthError,
	MetaRateLimitError,
	get_page,
)
from app.services.get_posts_ import get_page_posts
from app.services.instagram_ import get_instagram_comments, get_instagram_media


load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai"])
# Upper bound on posts read per page per run (newest first).
MAX_POSTS_SCANNED = 100
# Moderation and sentiment for a page share one Meta fetch for this long.
FETCH_CACHE_TTL = timedelta(minutes=10)
# After Meta reports a request limit, make no Meta calls for this long.
RATE_LIMIT_COOLDOWN_MINUTES = 20
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


def instagram_id_for(stored_page: Page | None) -> str | None:
	"""The page's Instagram account id, if linked and switched on in the portal."""
	if stored_page is None or not stored_page.instagram_enabled:
		return None
	return stored_page.ig_user_id


def _collect_instagram(
	instagram_id: str,
	page_id: str,
	access_token: str,
	post_window_start: datetime,
	comment_window_start: datetime,
	max_items: int,
	posts_by_id: dict[str, PostContext],
	recent: list[tuple[datetime, dict, str]],
) -> None:
	"""Add recent Instagram comments; a failure here never blocks Facebook."""
	try:
		media_items = get_instagram_media(
			instagram_id, access_token, post_window_start, MAX_POSTS_SCANNED
		)
		for media in media_items:
			media_id = str(media["id"])
			new_comments = get_instagram_comments(
				media, access_token, comment_window_start, max_items
			)
			for comment in new_comments:
				written_at = parse_meta_time(comment.get("created_time"))
				if written_at is not None:
					recent.append((written_at, comment, media_id))
			if new_comments:
				posts_by_id[media_id] = PostContext(
					post_id=media_id,
					page_id=page_id,
					body=str(media.get("caption") or ""),
				)
		logger.info("Instagram media scanned: media=%d", len(media_items))
	except MetaRateLimitError:
		raise  # a request limit must pause everything, not be skipped
	except MetaAPIError as error:
		logger.warning("Instagram comments skipped for page %s: %s", page_id, error)


def _build_page_request(
	page_id: str,
	page_name: str,
	access_token: str,
	settings: RuntimeSettings,
	instagram_id: str | None = None,
) -> ClassificationRequest | None:
	"""Collect comments written recently, on any post from the post window.

	Comments are selected by when they were written (``lookback_days``), not by
	the age of their post, so new comments on older posts are still checked.
	With ``instagram_id``, the linked Instagram account is included too.
	Returns None when there is nothing to classify.
	"""
	max_items = settings.max_comments_per_run
	now = datetime.now(timezone.utc)
	comment_window_start = now - timedelta(days=settings.lookback_days)
	post_window_start = now - timedelta(days=settings.post_lookback_days)
	posts_from_meta = get_page_posts(
		page_id,
		access_token,
		max_items=MAX_POSTS_SCANNED,
		since=int(post_window_start.timestamp()),
		until=int(now.timestamp()),
	)
	logger.info("Meta posts fetched: posts=%d", len(posts_from_meta))
	posts_by_id: dict[str, PostContext] = {}
	# (written at, comment, post id) for every comment inside the window.
	recent: list[tuple[datetime, dict, str]] = []
	quiet_posts = 0

	for post_number, post in enumerate(posts_from_meta, start=1):
		created_at = parse_meta_time(post.get("created_time"))
		if post.get("id") is None or created_at is None:
			continue
		if not post_window_start <= created_at <= now:
			continue
		# A new comment moves the post's updated_time, so a post untouched since
		# the comment window opened has nothing new: skip its comments request.
		updated_at = parse_meta_time(post.get("updated_time"))
		if updated_at is not None and updated_at < comment_window_start:
			quiet_posts += 1
			continue
		post_id = str(post["id"])
		new_comments = get_all_comments_from_a_post(
			post_id, access_token, max_items=max_items, newer_than=comment_window_start
		)
		for comment in new_comments:
			written_at = parse_meta_time(comment.get("created_time"))
			if written_at is not None and written_at >= comment_window_start:
				recent.append((written_at, comment, post_id))
		if new_comments:
			posts_by_id[post_id] = PostContext(
				post_id=post_id,
				page_id=page_id,
				body=str(post.get("message") or ""),
			)
		if post_number == 1 or post_number % 10 == 0:
			logger.info(
				"Meta comment collection progress: posts_scanned=%d new_comments=%d",
				post_number,
				len(recent),
			)

	if instagram_id:
		_collect_instagram(
			instagram_id,
			page_id,
			access_token,
			post_window_start,
			comment_window_start,
			max_items,
			posts_by_id,
			recent,
		)

	# Newest comments first, across all posts, up to the per-run limit.
	recent.sort(key=lambda item: item[0], reverse=True)
	comments: list[CommentContext] = []
	for _, comment, post_id in recent[:max_items]:
		comment_author = comment.get("from")
		comments.append(
			CommentContext(
				comment_id=(
					str(comment["id"]) if comment.get("id") is not None else None
				),
				post_id=post_id,
				page_id=page_id,
				platform=comment.get("platform", "facebook"),
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
	# Only posts that still have a selected comment are sent as context.
	used_post_ids = {comment.post_id for comment in comments}
	posts = [post for post_id, post in posts_by_id.items() if post_id in used_post_ids]

	logger.info(
		"Meta comment collection completed: posts_scanned=%d quiet_posts_skipped=%d "
		"posts_with_new_comments=%d new_comments=%d instagram=%d",
		len(posts_from_meta),
		quiet_posts,
		len(posts),
		len(comments),
		sum(comment.platform == "instagram" for comment in comments),
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
					"platform": comment.platform if comment else "facebook",
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


def _page_credentials(
	database: Session, page_id: str, stored_page: Page | None, refresh: bool = False
) -> tuple[str, str]:
	"""Return (page name, page access token).

	The token saved at import is used when there is one: it needs no extra
	request, and its calls count against the page's own limit rather than the
	small app-wide limit that user-token calls hit. With ``refresh`` (or no
	saved token) a new one is fetched with the user token and saved.
	"""
	if stored_page is not None and stored_page.access_token and not refresh:
		return stored_page.name, stored_page.access_token
	user_token = _get_meta_access_token()
	page = get_page(page_id, user_token)
	page_token = str(page.get("access_token") or user_token)
	if stored_page is not None:
		stored_page.access_token = page_token
		database.commit()
		logger.info("Saved a fresh page token for %s", page_id)
	return str(page.get("name") or page_id), page_token


def _rate_limited(until: datetime) -> HTTPException:
	return HTTPException(
		status_code=429,
		detail=f"Meta request limit reached; paused until {until:%H:%M} UTC",
		headers={"Retry-After": str(max(int((until - datetime.now(timezone.utc)).total_seconds()), 1))},
	)


def collect_page_comments(
	database: Session, page_id: str, stored_page: Page | None, settings: RuntimeSettings
) -> ClassificationRequest | None:
	"""Recent comments for a page, fetched from Meta at most once per few minutes.

	Moderation and sentiment both call this for the same page back to back; the
	second call reads the cached fetch. While Meta is rate-limiting us, no Meta
	calls are made at all (HTTP 429) until the cooldown ends.
	"""
	instagram_id = instagram_id_for(stored_page)
	signature = ":".join(
		str(part)
		for part in (
			settings.lookback_days,
			settings.post_lookback_days,
			settings.max_comments_per_run,
			instagram_id or "",
		)
	)
	cached = load_page_cache(database, page_id, signature, FETCH_CACHE_TTL)
	if cached is not None:
		logger.info("Using cached Meta fetch for page %s", page_id)
		return ClassificationRequest(**cached["request"]) if cached.get("request") else None

	paused_until = meta_cooldown_until(database)
	if paused_until is not None:
		raise _rate_limited(paused_until)

	try:
		name, token = _page_credentials(database, page_id, stored_page)
		try:
			request = _build_page_request(page_id, name, token, settings, instagram_id=instagram_id)
		except MetaAuthError:
			if stored_page is None:
				raise
			# The saved token stopped working: get a new one, save it, try once more.
			logger.warning("Saved page token for %s was rejected; refreshing it", page_id)
			name, token = _page_credentials(database, page_id, stored_page, refresh=True)
			request = _build_page_request(page_id, name, token, settings, instagram_id=instagram_id)
	except MetaRateLimitError as error:
		until = start_meta_cooldown(database, RATE_LIMIT_COOLDOWN_MINUTES)
		logger.warning("Meta request limit reached (%s); pausing until %s", error, until)
		raise _rate_limited(until) from error

	save_page_cache(
		database, page_id, signature, request.model_dump(mode="json") if request else None
	)
	return request


@router.post("/classify/{page_id}", response_model=list[ClassificationResult])
def classify_page(
	page_id: str, database: Session = Depends(get_db)
) -> list[ClassificationResult]:
	"""Fetch a page's content and classify comments not checked before."""
	stored_page = database.scalar(select(Page).where(Page.meta_page_id == page_id))
	if stored_page is not None and not stored_page.is_active:
		raise HTTPException(
			status_code=409,
			detail="This page is disabled in the admin portal",
		)
	settings = load_runtime_settings(database)
	try:
		request = collect_page_comments(database, page_id, stored_page, settings)
		if request is None:
			return []
		return _format_results(
			_classify_new_comments(request, database, settings), request
		)
	except HTTPException:
		raise  # e.g. 429 while Meta is rate-limiting
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
