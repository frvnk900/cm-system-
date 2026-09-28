from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.schema.system_prompt_model import SystemPromptVersion
from app.services.prompt.system_prompt import SYSTEM_PROMPT


def latest_prompt_version(database: Session) -> SystemPromptVersion | None:
	return database.scalar(
		select(SystemPromptVersion).order_by(SystemPromptVersion.id.desc()).limit(1)
	)


def load_system_prompt(database: Session) -> str:
	"""Return the newest saved prompt, or the built-in one if none is saved."""
	version = latest_prompt_version(database)
	return version.content if version is not None else SYSTEM_PROMPT


def save_system_prompt(database: Session, content: str) -> SystemPromptVersion:
	version = SystemPromptVersion(content=content)
	database.add(version)
	database.commit()
	database.refresh(version)
	return version


def list_prompt_versions(database: Session, limit: int = 20) -> list[SystemPromptVersion]:
	return list(
		database.scalars(
			select(SystemPromptVersion).order_by(SystemPromptVersion.id.desc()).limit(limit)
		)
	)


def reset_system_prompt(database: Session) -> None:
	"""Delete all saved versions so the built-in prompt is used again."""
	database.execute(delete(SystemPromptVersion))
	database.commit()
