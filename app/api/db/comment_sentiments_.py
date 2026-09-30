from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.db.classified_comments_ import body_hash
from app.api.schema.ai_model import CommentContext, SentimentOutput
from app.api.schema.comment_sentiment_model import CommentSentimentRecord


def load_sentiments(
	database: Session, comments: list[CommentContext]
) -> dict[str, CommentSentimentRecord]:
	"""Cached sentiments that still match the comment text, keyed by comment id."""
	by_id = {comment.comment_id: comment for comment in comments if comment.comment_id}
	if not by_id:
		return {}
	rows = database.scalars(
		select(CommentSentimentRecord).where(CommentSentimentRecord.comment_id.in_(by_id))
	)
	return {
		row.comment_id: row
		for row in rows
		if row.body_hash == body_hash(by_id[row.comment_id].body)
	}


def save_sentiments(
	database: Session,
	comments: list[CommentContext],
	results: list[SentimentOutput],
) -> None:
	"""Store the sentiment of each classified comment."""
	comments_by_id = {comment.comment_id: comment for comment in comments}
	for result in results:
		comment = comments_by_id.get(result.comment_id)
		if comment is None:
			continue
		database.merge(
			CommentSentimentRecord(
				comment_id=result.comment_id,
				body_hash=body_hash(comment.body),
				sentiment=result.sentiment,
				confidence=result.confidence,
				reason=result.reason,
				page_id=comment.page_id,
			)
		)
	database.commit()
