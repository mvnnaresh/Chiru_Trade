"""Shared secret loading for live data adapters."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def load_upstox_access_token(
    *,
    root: str | Path = ".",
    env_var: str = "UPSTOX_ACCESS_TOKEN",
) -> str:
    """Return the Upstox access token from env or Streamlit secrets."""
    token = os.environ.get(env_var, "").strip()
    if token:
        return token
    if tomllib is None:
        return ""
    root_path = Path(root).resolve()
    secrets_path = root_path / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return ""
    try:
        payload = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    value = payload.get(env_var, "")
    return str(value).strip() if value is not None else ""
