from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import (
	APIConnectionError,
	APIStatusError,
	APITimeoutError,
	AuthenticationError,
	LengthFinishReasonError,
	OpenAI,
	RateLimitError,
)

from app.api.schema.ai_model import (
	ClassificationBatch,
	ClassificationRequest,
	ClassificationResult,
	CommentContext,
)
from app.core.settings import RuntimeSettings, default_runtime_settings, get_settings
from app.services.prompt.system_prompt import SYSTEM_PROMPT


logger = logging.getLogger(__name__)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class AIConfigurationError(Exception):
	"""Raised when an AI provider configuration is incomplete."""


class AIProviderError(Exception):
	"""Raised when an AI provider cannot complete a request."""


class AIOutputError(Exception):
	"""Raised when the model output violates the classification contract."""


class ClassificationState(TypedDict, total=False):
	request: ClassificationRequest
	settings: RuntimeSettings
	system_prompt: str
	prepared_input: str
	results: list[ClassificationResult]


def _prepare_input(state: ClassificationState) -> dict[str, str]:
	request = state["request"]
	logger.info(
		"AI classification input prepared: posts=%d comments=%d",
		len(request.posts),
		len(request.comments),
	)
	return {
		"prepared_input": json.dumps(
			request.model_dump(mode="json"), ensure_ascii=True
		)
	}


class _Provider(TypedDict):
	name: str
	key_env: str
	model: str
	base_url: str | None
	extra_params: dict[str, Any]


def _providers(settings: RuntimeSettings) -> list[_Provider]:
	"""Providers in the order the admin portal says to try them."""
	available: dict[str, _Provider] = {
		"openai": {
			"name": "OpenAI",
			"key_env": "OPENAI_API_KEY",
			"model": settings.openai_model,
			"base_url": None,
			"extra_params": {},
		},
		"gemini": {
			"name": "Gemini",
			"key_env": "GEMINI_API_KEY",
			"model": settings.gemini_model,
			"base_url": GEMINI_BASE_URL,
			# Gemini's thinking tokens count against max_completion_tokens;
			# keep them small so the structured output is not truncated.
			"extra_params": {"reasoning_effort": settings.gemini_reasoning_effort},
		},
	}
	return [available[name] for name in settings.provider_order]


def _classify_with(
	provider: _Provider, state: ClassificationState
) -> list[ClassificationResult]:
	name = provider["name"]
	settings = state["settings"]
	api_key = getattr(get_settings(), provider["key_env"].lower())
	if not api_key:
		raise AIConfigurationError(f"{provider['key_env']} is not configured")

	logger.info(
		"%s classification request started: model=%s comments=%d",
		name,
		provider["model"],
		len(state["request"].comments),
	)
	client = OpenAI(
		api_key=api_key,
		base_url=provider["base_url"],
		timeout=settings.ai_timeout_seconds,
	)
	try:
		completion = client.chat.completions.parse(
			model=provider["model"],
			messages=[
				{"role": "system", "content": state.get("system_prompt") or SYSTEM_PROMPT},
				{"role": "user", "content": state["prepared_input"]},
			],
			response_format=ClassificationBatch,
			max_completion_tokens=settings.max_completion_tokens,
			**provider["extra_params"],
		)
	except LengthFinishReasonError as error:
		logger.warning("%s classification output hit the token limit", name)
		raise AIOutputError("The model response was cut off by the token limit") from error
	except AuthenticationError as error:
		logger.error("%s authentication failed; check %s", name, provider["key_env"])
		raise AIConfigurationError(f"{provider['key_env']} is invalid or expired") from error
	except (
		APITimeoutError,
		RateLimitError,
		APIConnectionError,
		APIStatusError,
	) as error:
		logger.warning("%s classification request failed: %s", name, type(error).__name__)
		raise AIProviderError("The classification provider is temporarily unavailable") from error

	message = completion.choices[0].message
	if message.refusal or message.parsed is None:
		logger.warning("%s classification returned no usable structured output", name)
		raise AIOutputError("The model did not return a valid classification")

	logger.info(
		"%s classification response received: results=%d",
		name,
		len(message.parsed.results),
	)
	return [
		ClassificationResult(**output.model_dump())
		for output in message.parsed.results
	]


def _classify_input(state: ClassificationState) -> dict[str, list[ClassificationResult]]:
	providers = _providers(state["settings"])
	for position, provider in enumerate(providers):
		is_last = position == len(providers) - 1
		try:
			return {"results": _classify_with(provider, state)}
		except (AIConfigurationError, AIProviderError) as error:
			if is_last:
				raise
			logger.warning(
				"%s unavailable (%s); falling back to %s",
				provider["name"],
				error,
				providers[position + 1]["name"],
			)
	raise AIConfigurationError("No classification provider is configured")


def _validate_results(state: ClassificationState) -> dict[str, list[ClassificationResult]]:
	request = state["request"]
	results = state.get("results", [])
	post_pages = {post.post_id: post.page_id for post in request.posts if post.post_id}
	validated: list[ClassificationResult] = []
	matched_comments: set[int] = set()

	for result in results:
		matching_comments = []
		for index, comment in enumerate(request.comments):
			expected_page_id = comment.page_id
			if expected_page_id is None and comment.post_id:
				expected_page_id = post_pages.get(comment.post_id)
			if (
				result.comment_id == comment.comment_id
				and result.post_id == comment.post_id
				and result.page_id == expected_page_id
				and result.location == request.location
			):
				matching_comments.append((index, expected_page_id))

		if len(matching_comments) != 1 or matching_comments[0][0] in matched_comments:
			raise AIOutputError("The model changed an input identifier or location")
		matched_comments.add(matching_comments[0][0])

		if result.type == "none":
			continue
		if not result.flagged:
			raise AIOutputError("The model returned inconsistent classification fields")
		validated.append(result)

	logger.info(
		"AI classification results validated: results=%d flagged=%d",
		len(validated),
		sum(result.flagged for result in validated),
	)
	return {"results": validated}


def _build_graph():
	graph = StateGraph(ClassificationState)
	graph.add_node("prepare", _prepare_input)
	graph.add_node("classify", _classify_input)
	graph.add_node("validate", _validate_results)
	graph.add_edge(START, "prepare")
	graph.add_edge("prepare", "classify")
	graph.add_edge("classify", "validate")
	graph.add_edge("validate", END)
	return graph.compile()


CLASSIFICATION_GRAPH = _build_graph()


def classify_comments(
	request: ClassificationRequest,
	settings: RuntimeSettings | None = None,
	on_batch: Callable[[list[CommentContext], list[ClassificationResult]], None]
	| None = None,
	system_prompt: str | None = None,
) -> list[ClassificationResult]:
	"""Run the typed LangGraph classification workflow.

	A failed batch is logged and skipped so the other batches still return;
	the error is raised only when every batch fails. ``on_batch`` is called
	after each successful batch with its comments and results.
	"""
	settings = settings or default_runtime_settings()
	batch_size = settings.batch_size
	logger.info(
		"AI classification workflow started: posts=%d comments=%d",
		len(request.posts),
		len(request.comments),
	)
	seen_bodies: set[str] = set()
	processable_comments = []
	fallback_results: dict[int, ClassificationResult] = {}
	for index, comment in enumerate(request.comments):
		normalized_body = comment.body.strip()
		is_duplicate = bool(normalized_body) and normalized_body in seen_bodies
		if normalized_body:
			seen_bodies.add(normalized_body)
		if not normalized_body or is_duplicate:
			fallback_results[index] = ClassificationResult(
				comment_id=comment.comment_id,
				post_id=comment.post_id,
				page_id=comment.page_id,
				location=request.location,
				flagged=False,
				confidence=0.0,
				type="none",
				reason="unparsable input",
			)
		else:
			processable_comments.append((index, comment))

	model_results: dict[int, ClassificationResult] = {}
	batch_errors: list[AIProviderError | AIOutputError] = []
	batch_count = 0
	try:
		for chunk_start in range(
			0, len(processable_comments), batch_size
		):
			comment_chunk = processable_comments[
				chunk_start : chunk_start + batch_size
			]
			chunk_request = request.model_copy(
				update={
					"comments": [comment for _, comment in comment_chunk]
				}
			)
			batch_number = chunk_start // batch_size + 1
			batch_count += 1
			logger.info(
				"Classifying comment batch: batch=%d size=%d",
				batch_number,
				len(comment_chunk),
			)
			try:
				state: dict[str, Any] = CLASSIFICATION_GRAPH.invoke(
					{
						"request": chunk_request,
						"settings": settings,
						"system_prompt": system_prompt or SYSTEM_PROMPT,
					}
				)
			except (AIProviderError, AIOutputError) as error:
				logger.warning(
					"Comment batch %d failed and was skipped: %s", batch_number, error
				)
				batch_errors.append(error)
				continue
			matched_indices: set[int] = set()
			batch_results: dict[int, ClassificationResult] = {}
			post_pages = {
				post.post_id: post.page_id
				for post in chunk_request.posts
				if post.post_id
			}
			for result in state["results"]:
				matching_comments = []
				for original_index, comment in comment_chunk:
					if original_index in matched_indices:
						continue
					expected_page_id = comment.page_id
					if expected_page_id is None and comment.post_id:
						expected_page_id = post_pages.get(comment.post_id)
					if (
						result.comment_id == comment.comment_id
						and result.post_id == comment.post_id
						and result.page_id == expected_page_id
					):
						matching_comments.append(original_index)
				if len(matching_comments) != 1:
					break
				matched_indices.add(matching_comments[0])
				batch_results[matching_comments[0]] = result
			else:
				model_results.update(batch_results)
				if on_batch is not None:
					on_batch(
						chunk_request.comments,
						[batch_results[index] for index in sorted(batch_results)],
					)
				continue
			error = AIOutputError(
				"The model returned an unknown or duplicate comment result"
			)
			logger.warning(
				"Comment batch %d failed and was skipped: %s", batch_number, error
			)
			batch_errors.append(error)
		if batch_errors and len(batch_errors) == batch_count:
			raise batch_errors[-1]
	except (AIConfigurationError, AIProviderError, AIOutputError):
		logger.warning("AI classification workflow stopped with a handled error")
		raise
	except Exception:
		logger.exception("AI classification workflow failed")
		raise

	results = [model_results[index] for index in sorted(model_results)]
	logger.info(
		"AI classification workflow completed: results=%d failed_batches=%d/%d",
		len(results),
		len(batch_errors),
		batch_count,
	)
	return results