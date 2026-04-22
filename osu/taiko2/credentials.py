"""Secret loading for osu! API credentials.

Lookup order (first hit wins):
  1. Process environment variables.
  2. `osu/taiko2/.env` (dotenv format: KEY=VALUE per line, '#' comments).
  3. `osu/taiko2/secrets.json` (flat JSON object).

Neither of the file forms is tracked in git. The secrets module never logs
or prints the values it loads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_TAIKO2_DIR = Path(__file__).resolve().parent
_ENV_PATH = _TAIKO2_DIR / ".env"
_JSON_PATH = _TAIKO2_DIR / "secrets.json"

_cache: dict[str, str] | None = None


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def _load_all() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache

    merged: dict[str, str] = {}
    if _JSON_PATH.exists():
        try:
            data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update({str(k): str(v) for k, v in data.items()})
        except (OSError, json.JSONDecodeError):
            pass
    if _ENV_PATH.exists():
        try:
            merged.update(_parse_env_file(_ENV_PATH))
        except OSError:
            pass
    _cache = merged
    return merged


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a secret by name, or `default` if unset."""
    val = os.environ.get(name)
    if val is not None:
        return val
    return _load_all().get(name, default)


def require(name: str) -> str:
    """Return a secret by name; raise `RuntimeError` if missing."""
    val = get(name)
    if not val:
        raise RuntimeError(
            f"Missing secret {name!r}. Set it as an environment variable, or "
            f"add it to {_ENV_PATH} or {_JSON_PATH}."
        )
    return val
