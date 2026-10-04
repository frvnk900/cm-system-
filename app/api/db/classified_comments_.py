import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schema.ai_model import (
	ClassificationOutput,
	ClassificationResult,
	CommentContext,
)
from app.api.schema.classified_comment_model import ClassifiedComment


OUTPUT_FIELDS = set(ClassificationOutput.model_fields)


def body_hash(body: str) -> str:
	return hashlib.sha256(body.strip().encode("utf-8")).hexdigest()


def load_classified(
	database: Session, comments: list[CommentContext]
) -> dict[str, ClassifiedComment]:
	"""Return cached classifications keyed by comment id."""
	comment_ids = [comment.comment_id for comment in comments if comment.comment_id]
	if not comment_ids:
		return {}
	rows = database.scalars(
		select(ClassifiedComment).where(ClassifiedComment.comment_id.in_(comment_ids))
	)
	return {row.comment_id: row for row in rows}


def save_classified(
	database: Session,
	comments: list[CommentContext],
	results: list[ClassificationResult],
	location: str,
) -> None:
	"""Record every checked comment; comments without a result were clean."""
	results_by_id = {result.comment_id: result for result in results}
	for comment in comments:
		if not comment.comment_id:
			continue
		result = results_by_id.get(comment.comment_id)
		database.merge(
			ClassifiedComment(
				comment_id=comment.comment_id,
				body_hash=body_hash(comment.body),
				result=result.model_dump(include=OUTPUT_FIELDS) if result else None,
				platform=comment.platform,
				page_id=comment.page_id,
				location=location,
				comment_text=comment.body,
				author=comment.author,
				comment_link=comment.comment_link,
				time_posted=comment.time_posted,
			)
		)
	database.commit()
