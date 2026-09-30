"""Positive / neutral / negative sentiment for every comment."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from app.api.schema.ai_model import (
	ClassificationRequest,
	CommentContext,
	SentimentBatch,
	SentimentOutput,
)
from app.core.settings import RuntimeSettings
from app.services.ai_classification import (
	AIOutputError,
	AIProviderError,
	complete_with_fallback,
)


logger = logging.getLogger(__name__)
POST_CONTEXT_CHARS = 500

SENTIMENT_PROMPT = """You label the sentiment of comments on a business's social media page.

Input JSON: {"location": page name, "posts": [{post_id, text}], "comments": [{comment_id, post_id, text}]}.
Use each comment's post only as context.

For EVERY comment return exactly one object:
- comment_id: copied exactly from the input
- sentiment: "positive", "neutral" or "negative"
- confidence: 0.0-1.0
- reason: one short sentence

positive = praise, thanks, support, excitement, recommendations.
negative = complaints, criticism, anger, accusations, insults, or sarcasm aimed at the business.
neutral = questions, tagging friends, off-topic, purely informational, or unclear.

Judge the comment as written. Sarcastic praise is negative. When unsure, use neutral with lower confidence.
Never invent or change a comment_id, and do not skip any comment."""


def classify_sentiment(
	request: ClassificationRequest,
	settings: RuntimeSettings,
	on_batch: Callable[[list[CommentContext], list[SentimentOutput]], None] | None = None,
) -> dict[str, SentimentOutput]:
	"""Classify every comment in ``request``; returns results keyed by comment id.

	A failed batch is logged and skipped so the others still return; the error
	is raised only when every batch fails. ``on_batch`` receives each successful
	batch so results can be cached as they arrive.
	"""
	results: dict[str, SentimentOutput] = {}
	errors: list[AIProviderError | AIOutputError] = []
	comments = request.comments
	batch_count = 0

	for start in range(0, len(comments), settings.batch_size):
		chunk = comments[start : start + settings.batch_size]
		batch_count += 1
		post_ids = {comment.post_id for comment in chunk}
		payload = {
			"location": request.location,
			"posts": [
				{"post_id": post.post_id, "text": post.body[:POST_CONTEXT_CHARS]}
				for post in request.posts
				if post.post_id in post_ids
			],
			"comments": [
				{"comment_id": c.comment_id, "post_id": c.post_id, "text": c.body}
				for c in chunk
			],
		}
		try:
			batch = complete_with_fallback(
				settings,
				SENTIMENT_PROMPT,
				json.dumps(payload, ensure_ascii=True),
				SentimentBatch,
				"sentiment",
			)
		except (AIProviderError, AIOutputError) as error:
			logger.warning("Sentiment batch %d failed and was skipped: %s", batch_count, error)
			errors.append(error)
			continue

		wanted = {comment.comment_id for comment in chunk}
		valid = [
			result
			for result in batch.results
			if result.comment_id in wanted and result.comment_id not in results
		]
		for result in valid:
			results[result.comment_id] = result
		if len(valid) < len(chunk):
			logger.warning(
				"Sentiment batch %d: %d of %d comments were not labelled",
				batch_count,
				len(chunk) - len(valid),
				len(chunk),
			)
		if on_batch is not None:
			on_batch(chunk, valid)

	if errors and len(errors) == batch_count:
		raise errors[-1]
	logger.info(
		"Sentiment completed: labelled=%d of %d, failed_batches=%d",
		len(results),
		len(comments),
		len(errors),
	)
	return results
