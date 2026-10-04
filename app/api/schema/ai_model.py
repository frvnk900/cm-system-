from typing import Literal

from pydantic import BaseModel, Field


ClassificationType = Literal[
	"harassment",
	"profanity",
	"threat",
	"hate_speech",
	"defamation",
	"misinformation",
	"spam",
	"doxxing",
	"negative_sentiment",
	"none",
]


class PostContext(BaseModel):
	post_id: str | None = None
	page_id: str | None = None
	body: str


Platform = Literal["facebook", "instagram"]


class CommentContext(BaseModel):
	comment_id: str | None = None
	post_id: str | None = None
	page_id: str | None = None
	platform: Platform = "facebook"
	author: str | None = None
	body: str
	comment_link: str | None = None
	time_posted: str | None = None


class ClassificationRequest(BaseModel):
	location: str = Field(min_length=1)
	posts: list[PostContext] = Field(min_length=1)
	comments: list[CommentContext] = Field(min_length=1)


class ClassificationOutput(BaseModel):
	"""Fields the model produces; everything else is filled in by the API."""

	comment_id: str | None = None
	post_id: str | None = None
	page_id: str | None = None
	location: str
	flagged: bool
	confidence: float = Field(ge=0.0, le=1.0)
	type: ClassificationType
	reason: str = Field(min_length=1, max_length=500)


class ClassificationResult(ClassificationOutput):
	logged_at: str = ""
	platform: str = "facebook"
	page_name: str = ""
	comment: str = ""
	comment_link: str = ""
	time_posted: str = ""
	author: str = ""
	severity: str = ""
	category: str = ""
	status: str = "pending"
	parent_post_id: str = ""


class ClassificationBatch(BaseModel):
	results: list[ClassificationOutput]


Sentiment = Literal["positive", "neutral", "negative"]


class SentimentOutput(BaseModel):
	"""One comment's sentiment, as produced by the model."""

	comment_id: str
	sentiment: Sentiment
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str = Field(min_length=1, max_length=300)


class SentimentBatch(BaseModel):
	results: list[SentimentOutput]


class CommentSentiment(BaseModel):
	"""A comment with its sentiment, as returned by the API.

	``sentiment`` is "unknown" when the AI could not classify it this run.
	Delete it with DELETE /comments/{comment_id}?page_id={page_id}.
	"""

	comment_id: str
	post_id: str | None = None
	page_id: str | None = None
	location: str
	platform: str = "facebook"
	sentiment: Literal["positive", "neutral", "negative", "unknown"]
	confidence: float = 0.0
	reason: str = ""
	comment: str = ""
	author: str = ""
	comment_link: str = ""
	time_posted: str = ""