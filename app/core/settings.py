from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal["openai", "gemini"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


class Settings(BaseSettings):
	"""Server configuration read from .env; changes need a restart."""

	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	# Secrets: shown masked in the admin portal, edited only in .env.
	openai_api_key: str | None = None
	gemini_api_key: str | None = None
	meta_access_token: str | None = None
	admin_password: str | None = None
	admin_session_secret: str | None = None
	admin_session_hours: int = 12
	# Where the admin portal lives. Set ADMIN_PATH to something private
	# (e.g. /team-portal-7f3k) so it can't be found by guessing /admin.
	admin_path: str = "/admin"

	@field_validator("admin_path")
	@classmethod
	def _clean_admin_path(cls, value: str) -> str:
		path = "/" + value.strip().strip("/")
		if path == "/" or not all(part.replace("-", "").replace("_", "").isalnum() for part in path.strip("/").split("/")):
			raise ValueError("ADMIN_PATH must look like /my-portal (letters, digits, - and _)")
		return path

	# Server config: shown read-only in the admin portal.
	meta_graph_version: str = Field(
		"v23.0",
		validation_alias=AliasChoices("META_GRAPH_VERSION", "META_GRAPH_API_VERSION"),
	)
	meta_http_timeout: float = 30

	# Defaults for the runtime settings below.
	openai_model: str = "gpt-4o-mini"
	gemini_model: str = "gemini-3.8-flash"
	gemini_reasoning_effort: ReasoningEffort = "low"
	openai_timeout: float = 120


@lru_cache
def get_settings() -> Settings:
	return Settings()


class RuntimeSettings(BaseModel):
	"""Settings admins change in the portal; stored in the database."""

	# AI
	provider_order: list[ProviderName] = Field(min_length=1)
	openai_model: str = Field(min_length=1)
	gemini_model: str = Field(min_length=1)
	gemini_reasoning_effort: ReasoningEffort
	ai_timeout_seconds: float = Field(ge=10, le=600)
	batch_size: int = Field(ge=1, le=50)
	max_completion_tokens: int = Field(ge=1000, le=65000)

	# Fetching: comments written in the last `lookback_days` are checked,
	# found on posts published in the last `post_lookback_days`.
	lookback_days: int = Field(ge=1, le=30)
	post_lookback_days: int = Field(default=30, ge=1, le=365)
	max_comments_per_run: int = Field(ge=1, le=500)

	@field_validator("provider_order")
	@classmethod
	def _unique_providers(cls, value: list[ProviderName]) -> list[ProviderName]:
		if len(set(value)) != len(value):
			raise ValueError("each provider can appear only once")
		return value

	@model_validator(mode="after")
	def _post_window_covers_comments(self) -> "RuntimeSettings":
		if self.post_lookback_days < self.lookback_days:
			raise ValueError(
				"post_lookback_days must be at least lookback_days "
				"(a comment can't be older than its post)"
			)
		return self


def default_runtime_settings() -> RuntimeSettings:
	settings = get_settings()
	return RuntimeSettings(
		provider_order=["openai", "gemini"],
		openai_model=settings.openai_model,
		gemini_model=settings.gemini_model,
		gemini_reasoning_effort=settings.gemini_reasoning_effort,
		ai_timeout_seconds=settings.openai_timeout,
		batch_size=10,
		max_completion_tokens=8000,
		lookback_days=2,
		post_lookback_days=30,
		max_comments_per_run=50,
	)
