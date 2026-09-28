import logging

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.schema.app_settings_model import AppSettings
from app.core.settings import RuntimeSettings, default_runtime_settings


logger = logging.getLogger(__name__)
SETTINGS_ROW_ID = 1


def load_runtime_settings(database: Session) -> RuntimeSettings:
	"""Return stored settings layered over the .env defaults."""
	defaults = default_runtime_settings()
	row = database.get(AppSettings, SETTINGS_ROW_ID)
	if row is None or not row.data:
		return defaults
	known = {key: value for key, value in row.data.items() if key in RuntimeSettings.model_fields}
	try:
		return RuntimeSettings(**{**defaults.model_dump(), **known})
	except ValidationError:
		logger.warning("Stored admin settings are invalid; using defaults")
		return defaults


def save_runtime_settings(database: Session, settings: RuntimeSettings) -> RuntimeSettings:
	row = database.get(AppSettings, SETTINGS_ROW_ID)
	if row is None:
		row = AppSettings(id=SETTINGS_ROW_ID)
		database.add(row)
	row.data = settings.model_dump()
	database.commit()
	return settings


def reset_runtime_settings(database: Session) -> RuntimeSettings:
	row = database.get(AppSettings, SETTINGS_ROW_ID)
	if row is not None:
		database.delete(row)
		database.commit()
	return default_runtime_settings()
