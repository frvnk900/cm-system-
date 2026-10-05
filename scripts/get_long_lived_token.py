"""Exchange a short-lived Meta user token for a long-lived one (~60 days).

Reads META_APP_ID and META_APP_SECRET from .env. The short-lived token is
taken from --token, or META_ACCESS_TOKEN in .env, or asked for.

Usage (from the backend folder):
    python scripts/get_long_lived_token.py                 # exchange and print
    python scripts/get_long_lived_token.py --save          # also write it to .env
    python scripts/get_long_lived_token.py --token EAAB... --save

Page tokens fetched with a long-lived user token do not expire, so after
saving, run "Import pages from Meta" in /admin to store them.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
GRAPH_VERSION = "v26.0"


def graph_url() -> str:
	version = os.getenv("META_GRAPH_VERSION") or os.getenv("META_GRAPH_API_VERSION") or GRAPH_VERSION
	return f"https://graph.facebook.com/{version}"


def fail(message: str) -> None:
	print(f"ERROR: {message}", file=sys.stderr)
	sys.exit(1)


def describe_expiry(expires_at: int | None) -> str:
	if not expires_at:
		return "never"
	when = datetime.fromtimestamp(expires_at, timezone.utc)
	days = (when - datetime.now(timezone.utc)).days
	return f"{when:%Y-%m-%d %H:%M} UTC (in {days} days)"


def exchange(app_id: str, app_secret: str, short_token: str) -> str:
	response = httpx.get(
		f"{graph_url()}/oauth/access_token",
		params={
			"grant_type": "fb_exchange_token",
			"client_id": app_id,
			"client_secret": app_secret,
			"fb_exchange_token": short_token,
		},
		timeout=30,
	)
	data = response.json()
	if response.is_error or "access_token" not in data:
		message = data.get("error", {}).get("message", response.text)
		fail(f"Meta refused the exchange: {message}")
	return data["access_token"]


def inspect(token: str, app_id: str, app_secret: str) -> None:
	"""Print who the token belongs to, when it expires and its permissions."""
	response = httpx.get(
		f"{graph_url()}/debug_token",
		params={"input_token": token, "access_token": f"{app_id}|{app_secret}"},
		timeout=30,
	)
	info = response.json().get("data", {})
	me = httpx.get(
		f"{graph_url()}/me", params={"access_token": token, "fields": "name"}, timeout=30
	).json()
	print(f"  user:        {me.get('name', '?')}")
	print(f"  valid:       {info.get('is_valid')}")
	print(f"  expires:     {describe_expiry(info.get('expires_at'))}")
	print(f"  permissions: {', '.join(sorted(info.get('scopes', []))) or '?'}")

	pages = httpx.get(
		f"{graph_url()}/me/accounts",
		params={"access_token": token, "fields": "name", "limit": 100},
		timeout=30,
	).json()
	names = [page.get("name", "?").strip() for page in pages.get("data", [])]
	print(f"  pages:       {len(names)} ({', '.join(names[:5])}{', …' if len(names) > 5 else ''})")


def save_to_env(token: str) -> None:
	"""Replace META_ACCESS_TOKEN in .env (a .env.bak copy is kept)."""
	text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
	if ENV_PATH.exists():
		shutil.copyfile(ENV_PATH, ENV_PATH.with_suffix(".bak"))
	line = f"META_ACCESS_TOKEN={token}"
	pattern = re.compile(r"^[ \t]*META_ACCESS_TOKEN[ \t]*=.*$", re.MULTILINE)
	if pattern.search(text):
		text = pattern.sub(line, text, count=1)
	else:
		text = text.rstrip("\n") + ("\n" if text else "") + line + "\n"
	ENV_PATH.write_text(text, encoding="utf-8")
	print(f"\nSaved to {ENV_PATH} (previous version: {ENV_PATH.with_suffix('.bak').name})")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--token", help="short-lived user token (default: META_ACCESS_TOKEN in .env)")
	parser.add_argument("--save", action="store_true", help="write the long-lived token to .env")
	args = parser.parse_args()

	load_dotenv(ENV_PATH)
	app_id = (os.getenv("META_APP_ID") or "").strip()
	app_secret = (os.getenv("META_APP_SECRET") or "").strip()
	if not app_id or not app_secret:
		fail(f"META_APP_ID and META_APP_SECRET must be set in {ENV_PATH}")

	short_token = (args.token or os.getenv("META_ACCESS_TOKEN") or "").strip()
	if not short_token:
		short_token = getpass.getpass("Paste the short-lived token (hidden): ").strip()
	if not short_token:
		fail("No token given")

	print("Exchanging token…")
	long_token = exchange(app_id, app_secret, short_token)
	print("\nLong-lived token details:")
	inspect(long_token, app_id, app_secret)

	print("\nLong-lived token (put this in Vercel as META_ACCESS_TOKEN):\n")
	print(long_token)

	if args.save:
		save_to_env(long_token)
	else:
		print("\nRun again with --save to write it to .env.")
	print("Then run 'Import pages from Meta' in /admin so page tokens are refreshed.")


if __name__ == "__main__":
	main()
