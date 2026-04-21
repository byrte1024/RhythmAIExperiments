"""Minimal osu! API v2 client: OAuth client-credentials token + batch fetch.

Credentials come from the `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` secrets
via `osu.taiko2.secrets`. Never accept them as call arguments — keeps keys
out of call sites, process lists, and logs.

Referenced against osu/taiko/datasets/fetch_ranked_status.py (now removed):
same endpoints, same batch size, same 429-retry shape.
"""
from __future__ import annotations

import time
from typing import Any

from .. import secrets

TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_BASE = "https://osu.ppy.sh/api/v2"
BATCH_SIZE = 50  # v2 /beatmaps accepts up to 50 ids[] per call


class OsuV2Client:
    """Thin wrapper around the subset of osu! API v2 this repo uses."""

    def __init__(self, timeout_s: float = 30.0):
        self._token: str | None = None
        self._timeout = timeout_s

    def _authenticate(self) -> str:
        import requests
        client_id = secrets.require("OSU_CLIENT_ID")
        client_secret = secrets.require("OSU_CLIENT_SECRET")
        resp = requests.post(
            TOKEN_URL,
            json={
                "client_id": int(client_id),
                "client_secret": client_secret,
                "grant_type": "client_credentials",
                "scope": "public",
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def token(self) -> str:
        if self._token is None:
            self._token = self._authenticate()
        return self._token

    def get_beatmaps(self, beatmap_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch up to `BATCH_SIZE` beatmaps by id. Retries once on 429."""
        import requests
        if not beatmap_ids:
            return []
        if len(beatmap_ids) > BATCH_SIZE:
            raise ValueError(f"max {BATCH_SIZE} ids per call, got {len(beatmap_ids)}")

        headers = {"Authorization": f"Bearer {self.token()}"}
        params = [("ids[]", str(b)) for b in beatmap_ids]
        resp = requests.get(
            f"{API_BASE}/beatmaps", headers=headers, params=params,
            timeout=self._timeout,
        )
        if resp.status_code == 429:
            time.sleep(60)
            resp = requests.get(
                f"{API_BASE}/beatmaps", headers=headers, params=params,
                timeout=self._timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("beatmaps", [])
