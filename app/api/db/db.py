from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool


load_dotenv()

DATABASE_URL = (os.getenv("POSTGRES_DB_URL") or "").strip()
if not DATABASE_URL:
	raise RuntimeError("POSTGRES_DB_URL is not configured")
# Use psycopg 3 whatever scheme was pasted; psycopg2 is not installed.
for _prefix in ("postgresql://", "postgres://"):
	if DATABASE_URL.startswith(_prefix):
		DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len(_prefix):]
		break


class Base(DeclarativeBase):
	pass


from app.api.schema import pages_model  # noqa: E402,F401
from app.api.schema import moderation_model  # noqa: E402,F401
from app.api.schema import classified_comment_model  # noqa: E402,F401
from app.api.schema import app_settings_model  # noqa: E402,F401
from app.api.schema import system_prompt_model  # noqa: E402,F401
from app.api.schema import comment_sentiment_model  # noqa: E402,F401


# Serverless instances should not hold pooled connections; Supabase's
# pooler (port 6543) does the pooling. The pooler shares server connections
# between clients, so psycopg's automatic prepared statements must be off.
engine = create_engine(
	DATABASE_URL,
	poolclass=NullPool,
	connect_args={"prepare_threshold": None},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_tables() -> None:
	"""Create ORM tables and widen Meta IDs for existing databases."""
	Base.metadata.create_all(bind=engine)
	with engine.begin() as connection:
		connection.execute(
			text(
				"ALTER TABLE pages ADD COLUMN IF NOT EXISTS location_id BIGINT"
			)
		)
		connection.execute(
			text("ALTER TABLE pages ALTER COLUMN id TYPE BIGINT")
		)
		for column, column_type in (
			("ig_user_id", "VARCHAR(255)"),
			("ig_username", "VARCHAR(255)"),
			("instagram_enabled", "BOOLEAN NOT NULL DEFAULT TRUE"),
		):
			connection.execute(
				text(f"ALTER TABLE pages ADD COLUMN IF NOT EXISTS {column} {column_type}")
			)
		for column, column_type in (
			("platform", "VARCHAR(20)"),
			("page_id", "VARCHAR(255)"),
			("location", "VARCHAR(255)"),
			("comment_text", "TEXT"),
			("author", "VARCHAR(255)"),
			("comment_link", "TEXT"),
			("time_posted", "VARCHAR(64)"),
		):
			connection.execute(
				text(
					f"ALTER TABLE classified_comments "
					f"ADD COLUMN IF NOT EXISTS {column} {column_type}"
				)
			)
		# Clean comments were once stored as JSON 'null'; normalise to SQL NULL.
		connection.execute(
			text(
				"UPDATE classified_comments SET result = NULL "
				"WHERE result IS NOT NULL AND result::text = 'null'"
			)
		)
		# Rows cached before page_id/location columns existed.
		connection.execute(
			text(
				"UPDATE classified_comments "
				"SET page_id = result->>'page_id', location = result->>'location' "
				"WHERE page_id IS NULL AND result IS NOT NULL"
			)
		)


 


def get_db() -> Generator[Session, None, None]:
	"""Provide a SQLAlchemy session for FastAPI dependencies."""
	database = SessionLocal()
	try:
		yield database
	finally:
		database.close()
